import os
import json
import time
import subprocess
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

def get_git_sha() -> str:
    """Pobiera aktualny hash commitu Git, by zapewnić odtwarzalność eksperymentu."""
    try:
        result = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
        return result.decode('ascii').strip()
    except Exception as e:
        print(f"[OSTRZEŻENIE] Nie udało się pobrać Git SHA: {e}")
        return "unknown"

def _atomic_write_json(path: str, obj: Any) -> None:
    """Zapisuje JSON atomowo: tmp w tym samym katalogu (ten sam filesystem,
    wiec os.replace jest atomowa podmiana na POSIX) + fsync + replace.
    Uzywane wszedzie tam, gdzie plik moze byc czytany jako 'jawny stan' do
    resume (M3 punkt 3) -- polowiczny zapis nie moze go uszkodzic."""
    tmp_path = path + f".tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, cls=SetEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class SetEncoder(json.JSONEncoder):
    """Pozwala na serializację obiektów typu 'set' jako listy (posortowane, jeśli to możliwe)."""
    def default(self, obj):
        if isinstance(obj, set):
            try:
                return sorted(obj)
            except TypeError:
                # elementy nieporównywalne (np. mieszane typy) — fallback bez sortowania
                return list(obj)
        return super().default(obj)

@dataclass
class RunTracker:
    """Odpowiada za tworzenie folderu eksperymentu i zarządzanie manifestem."""

    # Zmienne przekazywane przy tworzeniu obiektu
    config: Any
    device: str
    workers: int
    hw_benchmark: Dict[int, float] = field(default_factory=dict)
    base_dir: str = "runs"

    # Zmienne inicjalizowane automatycznie (nie podajemy ich w konstruktorze)
    start_time: float = field(init=False)
    timestamp: str = field(init=False)
    run_dir: str = field(init=False)
    stage_times: Dict[str, float] = field(default_factory=dict, init=False)
    metrics: Dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Uruchamia się automatycznie po przypisaniu pól przez @dataclass."""
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{self.timestamp}")
        self._setup_dir()

    def _setup_dir(self):
        """Tworzy strukturę katalogów i dokonuje początkowego zrzutu konfiguracji."""
        os.makedirs(self.run_dir, exist_ok=True)

        config_path = os.path.join(self.run_dir, "config.json")
        _atomic_write_json(config_path, asdict(self.config))

        self.update_manifest(status="RUNNING")
        print(f"[TRACKER] Rozpoczęto eksperyment. Katalog: {self.run_dir}")

    def log_stage_time(self, stage_name: str, duration: float):
        """Rejestruje czas trwania konkretnego etapu (np. GA, fine-tuning)."""
        self.stage_times[stage_name] = round(duration, 2)
        self.update_manifest(status="RUNNING")

    def log_metrics(self, dataset_name: str, new_metrics: dict):
        """Zapisuje metryki dla danego datasetu (np. 'spikes_ext', 'continuous')."""
        self.metrics[dataset_name] = new_metrics
        self.update_manifest(status="RUNNING")

    def update_manifest(self, status: str = "COMPLETED"):
        """Zapisuje kompletny stan eksperymentu do pliku manifest.json."""
        manifest = {
            "status": status,
            "git_sha": get_git_sha(),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "hardware": {
                "device": self.device,
                "workers": self.workers,
                "benchmark_results_sec": self.hw_benchmark
            },
            "execution_times_sec": self.stage_times,
            "metrics": self.metrics
        }

        if status != "RUNNING":
            manifest["total_wall_time_sec"] = round(time.time() - self.start_time, 2)

        # M3 punkt 3 (26.09.2026, Marcel): update_manifest() jest wolane po
        # KAZDYM etapie (log_stage_time, log_metrics), wiec manifest.json jest
        # jedynym jawnym stanem, z ktorego pipeline.py --resume odczytuje co
        # juz zrobiono. Wczesniej ten zapis nie byl atomowy -- crash/kill -9
        # w trakcie json.dump() zostawial obcięty/uszkodzony plik, co psulo
        # WLASNIE mechanizm resume, ktory ma chronic.
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        _atomic_write_json(manifest_path, manifest)

    def get_run_dir(self) -> str:
        return self.run_dir
    