# Checkpointy

Ten katalog jest **śledzony w gicie**, wbrew regułom `**/models/` i `**/*.pt`
w `.gitignore` (wyjątek `!/models/` na końcu pliku).

## Po co

Do sierpnia 2026 w repo nie było ani jednego pliku `.pt`. `hw7_config.json`
deklarował `"source_checkpoint": "hw7_s2.pt"`, a `sweep_pw10_s1.pt` był cytowany
w notatkach z kampanii, ale żaden z nich nie istniał na dysku. Skutkiem było to,
że **żadnego wyniku nie dało się ponownie ocenić** — ani przemierzyć inną
metryką, ani sprawdzić round-tripu eksportu na sprzęt, ani porównać dwóch
modeli na tym samym zbiorze. Można było tylko trenować od nowa i dostać inną
liczbę, bo rozrzut między seedami (0.53-0.59 val-F1) jest większy niż różnice,
o które się spieramy.

Sieć ma kilkanaście neuronów, więc checkpoint waży kilkadziesiąt kB. Nie ma
powodu, żeby go nie trzymać.

## Konwencja nazw

```
models/hw_<wersja_zbioru>_s<seed>.pt      checkpoint (wagi + epoka + faza)
models/hw_<wersja_zbioru>_s<seed>.json    konfiguracja sprzętowa + metryki
models/train_log_<wersja>_s<seed>.csv     przebieg treningu per epoka
```

`v2` oznacza artefakt spike'owy zbudowany z `dataset/versions/v2.0.0`.

## Co musi być w pliku .json

`snn_hw_pipeline.py train` zapisuje blok `model_provenance` automatycznie:
seed, `pos_weight`, `hat_frac`, ścieżkę checkpointu, commit gita, czas oraz
**pochodzenie artefaktu treningowego przepisane z jego `channels.json`**
(wersja zbioru, sha256 manifestu, sha256 enkodera, kolejność strumienia).

Wcześniej ten blok był wpisywany ręcznie i niósł tyle co
`"trained_on": "spikes_manifest7"`, czyli nazwę katalogu bez informacji, że ten
katalog ma przeciek między splitami i `clock_tick` w klasie pozytywnej.

## Czego tu nie trzymać

Artefaktów spike'owych (`spikes_v2/` i podobnych). Są odtwarzalne z audio przez
`encoder_twin.py build-manifest` i ważą setki MB. Patrz `docs/DATASET_CONTRACT.md`.
