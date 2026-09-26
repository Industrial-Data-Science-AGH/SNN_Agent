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
    # 26.09.2026, Marcel: zbudowane samodzielnie z lokalnego korpusu VOICe
    # (dataset/clean/audio + annotation) + ESC-50, generatorem z
    # dataset/continuous/eval (branch feat/continuous-dataset). 3 warianty,
    # seedy 42/43/44 (= config.seed + si dla fitness_seeds<=3), po 5 zdarzen
    # kazdy, 600s. Audio NIE jest w gicie (deterministyczne, regenerowalne),
    # zamrozone sa tylko manifesty (.manifest.json, z audio.sha256 w srodku).
    # Komenda do odtworzenia:
    #   python -m dataset.continuous.eval.cli \
    #       --glass-annotation-dir dataset/clean/annotation \
    #       --glass-audio-root     dataset/clean/audio \
    #       --glass-allowed-stems  dataset/clean/target/synthetic_target_test.txt \
    #       --train-stems-files    dataset/clean/source/synthetic_source_training.txt \
    #                              dataset/clean/source/synthetic_source_validation.txt \
    #       --train-manifest       dataset/versions/v2.0.0/manifest.csv \
    #       --background-dir       data/ESC-50-master/audio \
    #       --seeds 42 43 44 --out-dir dataset/continuous/out
    continuous_eval: str = "dataset/continuous/out"

@dataclass
class GAConfig:
    """Parametry przeszukiwania topologii i okablowania."""
    neurons_range: List[int] = field(default_factory=lambda: [4, 6, 8])
    pop_size: int = 30
    generations: int = 20
    elite: int = 3
    screen_mult: int = 1
    screen_budget: float = 0.34
    # M1 (25.09.2026, Marcel): było "recal_fa" -- literowka. RealFitness.__init__
    # (fitness.py) ma twarda asercje `metric in ("ap", "clip_f1", "recall_fa")`,
    # wiec kazdy config.json polegajacy na tym defaultcie (bez jawnego
    # nadpisania) wywalilby sie AssertionError zanim cokolwiek zaczeloby sie
    # trenowac. To jest tez "cel wyboru modelu na walidacji" z M1 punkt 1 --
    # musi byc jednoznaczny i poprawny, zeby zamrozenie protokolu mialo sens.
    fitness_metric: str = "recall_fa"
    feature_penalty: float = 0.005
    parsimony_eps: float = 0.02

    def __post_init__(self):
        if max(self.neurons_range) > 8:
            raise ValueError("Hardware constraint: max neurons na płytce Lu.i to 8!")

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
    # M2 punkt 2/3 (26.09.2026, Marcel): NOWE pole, celowo oddzielone od
    # fitness_seeds. fitness_seeds sluzy tanim, proxy ocenom w run_ga_stage
    # (kilka epok, wielokrotnie w petli GA). winner_seeds steruje
    # winner.train_full -- pelny, drogi trening HAT->QAT (winner_epochs=60)
    # zwyciezcy topologii, gdzie 'seeds' niezaleznych przebiegow sluzy do
    # wyboru MEDIANY (odpornosc na przypadek), nie do usredniania. Gdyby to
    # bylo jedno pole, podniesienie fitness_seeds dla taniej fazy GA po
    # cichu potroilyby tez koszt drogiego treningu zwyciezcy.
    winner_seeds: int = 5
    num_samples: int = 6000

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
            