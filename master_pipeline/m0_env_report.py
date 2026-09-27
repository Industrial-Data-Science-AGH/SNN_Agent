#!/usr/bin/env python3
"""M0 krok 1: raport sprzętu i środowiska (pola `hardware` i `environment`).

Tylko odczyt: nie instaluje niczego, nie zmienia środowiska, nie zapisuje
numeru seryjnego. Bezpieczne do uruchomienia podczas trwającego treningu
(mały test MPS to jedna tablica 1024x1024).

Użycie:
    python m0_env_report.py                # JSON na stdout
    python m0_env_report.py --out env.json # dodatkowo zapis do pliku
    python m0_env_report.py --repo /ścieżka/do/SNN_Agent
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def sh(cmd: list[str], timeout: int = 20) -> str | None:
    """Uruchom polecenie, zwróć stdout albo None przy błędzie."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def sysctl(key: str) -> str | None:
    return sh(["sysctl", "-n", key])


def sysctl_int(key: str) -> int | None:
    v = sysctl(key)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def hardware_section() -> dict:
    is_mac = platform.system() == "Darwin"
    hw: dict = {
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if not is_mac:
        hw["note"] = "Nie macOS: ten raport nie opisuje docelowego M5 Max."
        hw["cpu_logical"] = os.cpu_count()
        return hw

    mem = sysctl_int("hw.memsize")
    mac_ver = platform.mac_ver()[0]
    translated = sysctl_int("sysctl.proc_translated")  # 1 = Rosetta
    hw.update(
        {
            "macos": mac_ver,
            "cpu_brand": sysctl("machdep.cpu.brand_string"),
            "model_identifier": sysctl("hw.model"),
            "cpu_physical": sysctl_int("hw.physicalcpu"),
            "cpu_logical": sysctl_int("hw.logicalcpu"),
            # perflevel0 = rdzenie wydajne (P), perflevel1 = energooszczędne (E)
            "cpu_perf_cores": sysctl_int("hw.perflevel0.physicalcpu"),
            "cpu_eff_cores": sysctl_int("hw.perflevel1.physicalcpu"),
            "ram_bytes": mem,
            "ram_gib": round(mem / 2**30, 1) if mem else None,
            "arm64_capable": sysctl_int("hw.optional.arm64"),
            "running_under_rosetta": bool(translated) if translated is not None else None,
        }
    )

    # Liczba rdzeni GPU z system_profiler (bez numeru seryjnego: bierzemy tylko wybrane pola)
    raw = sh(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30)
    if raw:
        try:
            gpus = json.loads(raw).get("SPDisplaysDataType", [])
            hw["gpu"] = [
                {
                    "name": g.get("sppci_model"),
                    "cores": g.get("sppci_cores"),
                    "vendor": g.get("spdisplays_vendor"),
                }
                for g in gpus
            ]
        except json.JSONDecodeError:
            hw["gpu"] = None
    return hw


def memory_baseline() -> dict:
    """Stan pamięci PRZED benchmarkiem (istotne, jeśli GA już działa)."""
    info: dict = {"swap": sysctl("vm.swapusage")}
    mp = sh(["memory_pressure", "-Q"])
    info["memory_pressure_free_pct_line"] = (
        next((ln for ln in mp.splitlines() if "free percentage" in ln), mp)
        if mp
        else None
    )
    return info


def torch_section() -> dict:
    try:
        import torch  # noqa: WPS433
    except ImportError:
        return {"torch_installed": False}

    info: dict = {
        "torch_installed": True,
        "torch": torch.__version__,
        "torch_threads_default": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    mps = getattr(torch.backends, "mps", None)
    info["mps_is_built"] = bool(mps and mps.is_built())
    info["mps_is_available"] = bool(mps and mps.is_available())

    if info["mps_is_available"]:
        try:
            a = torch.randn(1024, 1024, device="mps")
            b = (a @ a).sum()
            torch.mps.synchronize()
            info["mps_smoke_test"] = {"ok": True, "checksum_finite": bool(torch.isfinite(b).item())}
            info["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
            info["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
            rec = getattr(torch.mps, "recommended_max_memory", None)
            info["mps_recommended_max_bytes"] = rec() if callable(rec) else None
        except Exception as exc:  # noqa: BLE001
            info["mps_smoke_test"] = {"ok": False, "error": repr(exc)}
    return info


def module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def git_section(repo: str | None) -> dict:
    if not repo:
        return {}
    head = sh(["git", "-C", repo, "rev-parse", "HEAD"])
    branch = sh(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = sh(["git", "-C", repo, "status", "--porcelain"])
    return {
        "source_commit": head,
        "branch": branch,
        "worktree_dirty": bool(dirty) if dirty is not None else None,
    }


def warnings_for(hw: dict, tr: dict) -> list[str]:
    w: list[str] = []
    if hw.get("system") == "Darwin":
        if hw.get("machine") != "arm64":
            w.append("Python nie jest natywnym arm64 (machine != arm64).")
        if hw.get("running_under_rosetta"):
            w.append("Proces działa pod Rosettą: wyniki nie odpowiadają natywnemu ARM64.")
        ram = hw.get("ram_gib")
        if ram is not None and abs(ram - 128) > 1:
            w.append(f"RAM = {ram} GiB, profil zakłada 128 GB.")
    else:
        w.append("Raport nie pochodzi z macOS/M5 Max.")
    if tr.get("torch_installed") and not tr.get("mps_is_available"):
        w.append("MPS niedostępny: benchmark GPU nie może zostać wykonany.")
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0":
        w.append("PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 wyłącza limit MPS; nie zalecane.")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        w.append("MPS fallback włączony: raportuj go jawnie, wynik może być hybrydowy.")
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="zapisz JSON do pliku")
    ap.add_argument("--repo", help="ścieżka repo do odczytu commita")
    args = ap.parse_args()

    hw = hardware_section()
    tr = torch_section()
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hw,
        "memory_baseline": memory_baseline(),
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "numpy": module_version("numpy"),
            **tr,
            "env_vars": {
                k: os.environ.get(k)
                for k in (
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            **git_section(args.repo),
        },
    }
    report["warnings"] = warnings_for(hw, tr)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
