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
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=4)
            
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

        with open(os.path.join(self.run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
            
    def get_run_dir(self) -> str:
        return self.run_dir
    