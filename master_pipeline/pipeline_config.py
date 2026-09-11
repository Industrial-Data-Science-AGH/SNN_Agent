import json
from dataclasses import dataclass, field, asdict
from typing import List, Union

@dataclass
class DataConfig:
    """Ścieżki i ustawienia dla wszystkich zbiorów danych i cech."""
    train: str = "architecture_14_neurons_patryk_09_07/spikes_v2/train"
    val: str = "architecture_14_neurons_patryk_09_07/spikes_v2/val"
    test: str = "architecture_14_neurons_patryk_09_07/spikes_v2/test"
    spikes_ext: str = "ga_neuron_search/spikes_ext"
    continuous_eval: str = "" #DODAJ PO KONTAKCIE Z KACPREM

@dataclass
class GAConfig:
    """Parametry przeszukiwania topologii i okablowania."""
    neurons_range: List[int] = field(default_factory=lambda: [4, 6, 8, 10])
    pop_size: int = 30
    generations: int = 20
    elite: int = 3
    screen_mult: int = 1
    screen_budget: float = 0.34
    fitness_metric: str = "clip_f1"
    feature_penalty: float = 0.005
    parsimony_eps: float = 0.02
    
    def __post_init__(self):
        if max(self.neurons_range) > 10:
            raise ValueError("Hardware constraint: max neurons na płytce Lu.i to 10!")

@dataclass
class TrainConfig:
    """Parametry proxy-treningu w GA oraz pełnego dotrenowania (HAT/QAT)."""
    proxy_epochs: int = 4
    winner_epochs: int = 60
    hat_frac: float = 0.4
    lr: float = 3e-3
    batch_size: int = 128
    pos_weight_grid: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    tune_k_range: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    fitness_seeds: int = 1

@dataclass
class PipelineConfig:
    """Główna klasa spinająca cały eksperyment MLOps."""
    data: DataConfig = field(default_factory=DataConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    device: str = "auto"
    workers: Union[int, str] = "auto"
    seed: int = 42
    
    @classmethod
    def from_json(cls, filepath: str) -> "PipelineConfig":
        """Ładuje konfigurację z pliku JSON i rozpakowuje do zagnieżdżonych klas."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
            
        return cls(
            data=DataConfig(**raw.get("data", {})),
            ga=GAConfig(**raw.get("ga", {})),
            train=TrainConfig(**raw.get("train", {})),
            device=raw.get("device", "auto"),
            workers=raw.get("workers", "auto"),
            seed=raw.get("seed", 42)
        )

    def to_json(self, filepath: str):
        """Zapisuje kompletną konfigurację na dysk (przydatne do manifestu runu)."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)
            