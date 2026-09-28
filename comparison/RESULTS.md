# Wyniki: SNN vs Fourier

Zbiór `dataset/versions/v2.0.0`, split `test` (1867 klipów, 2,76 h tła), metryka
`snn_pipeline/stream_eval.py`, ramka co 9984 µs. Próg detekcji i reguła k-of-w
wybierane na `val`, raport na `test` z zamrożoną parą i jedną regułą. Przedziały
ufności bootstrapowane po `group_id`. Protokół w `README.md`.

## Krzywa recall ↔ FA/h

| budżet FA/h | Fourier `mcu` | Fourier `full` | SNN (`models/WYNIKI.md`) |
|---|---|---|---|
| 1 | 1,5 % @ 0,4 | nie mieści się na teście | 0 % |
| 6 | 4,2 % @ 1,8 | 3,3 % @ 1,5 | **0 %** |
| 30 | 23,8 % @ 15,2 | 20,0 % @ 9,4 | — |
| 120 | 36,5 % @ 28,7 | 37,9 % @ 15,6 | — |
| 600 | 73,2 % @ 134,6 | **85,3 % @ 107,4** | 86,3 % @ 219 |

Zapis „X % @ Y" to recall przy zmierzonym łącznym FA/h, nie przy budżecie.
Budżet jest tylko ograniczeniem, przy którym wybrano punkt pracy, i wiąże
**każdy** rodzaj tła z osobna.

## Trzy rzeczy, które z tego wynikają

**1. Przewaga SNN nie leży w jakości.**
Fourier bez ograniczeń sprzętowych osiąga 85,3 % recall przy 107 FA/h. SNN
w najlepszym udokumentowanym przebiegu osiąga 86,3 % przy 219 FA/h. To jest ten
sam recall przy **połowie fałszywych alarmów**. Jeżeli projekt ma bronić tezy
„SNN jest lepszy", to nie na tej osi.

**2. Budżet mikrokontrolera kosztuje mniej więcej dwanaście punktów recall.**
Zejście z pełnego rFFT 512 na sześć pasm liczonych na gołym hopie zabiera
85,3 % → 73,2 % na szczycie krzywej, a przy dopasowanym recall (~37 %) mniej
więcej podwaja FA/h (15,6 → 28,7). To jest pierwsza liczba w tym projekcie,
która mówi, ile naprawdę kosztuje ograniczenie sprzętowe, a nie ile się go
obawiamy.

**3. Ścianą jest mowa i jest wspólna.**
W punkcie o najwyższym recall: Fourier `full` 529 FA/h na mowie, Fourier `mcu`
497, SNN 298–376. Trzy różne front endy, ten sam problem. To nie jest różnica
między architekturami, tylko brak rozdzielności mowa/szkło w danych, na których
uczymy. Dopóki to nie pęknie, żaden z tych trzech nie zbliży się do 6 FA/h.

Przy budżecie 6 FA/h wszystkie trzy podejścia są praktycznie przy zerze: 4,2 %,
3,3 % i 0 %. Próg akceptacji z `docs/DATASET_CONTRACT.md:297` (recall ≥ 0,70
przy ≤ 6 FA/h, żaden `kind` powyżej) **nie jest dziś osiągany przez nic**.

## Gdzie przewaga SNN jest realna

Cykle. Z pomiarów Kacpra na płytce (`comparison/mcu_budget.py`):

```
ATmega328P @ 16 MHz, ramka 9983 us = 159 728 cykli
  zmierzony ISR                  113 359 cykli
  zmierzona obróbka ramki         46 160 cykli
  wolne                              209 cykli   (99,87 % zajęte)
```

FFT 256-punktowe to szacunkowo 61 440–102 400 cykli (30 720 przy
optymistycznych 30 cyklach na motylek). Nie mieści się nawet blisko. Sześć binów
Goertzela to 11 520–18 432 cykli i mieści się, ale **tylko jako zamiennik**
obecnych siedmiu kanałów, bo wolnych cykli jest 209.

Czyli kolumna `full` z tabeli wyżej **nie jest opcją wdrożeniową na tym MCU**.
Jest sufitem, który mówi, ile Fourier w ogóle potrafi na tych danych.

## Czego ta tabela jeszcze nie dowodzi

Kolumna SNN pochodzi z `models/WYNIKI.md` na branchu `fah-metric-kn` (PR #49),
nie została przeliczona tą samą uprzężą. Zbiór i metryka są te same, co czyni
zestawienie sensownym, ale tamten przebieg używał dekodera k=1 na sztywno,
podczas gdy ta uprząż wybiera regułę na walidacji. **Domknięcie wymaga
checkpointu SNN i przepuszczenia go przez `comparison/evaluate.py`.**

Do tego czasu tabela pokazuje dwie kolumny zmierzone tak samo i jedną
zacytowaną.

## Ograniczenie pomiaru

W teście jest 2,76 h tła: `loud_event` 1,43 h, `stationary` 0,67 h, `animal`
0,44 h, `speech` 0,22 h. Jeden fałszywy alarm na mowie to 4,5 FA/h, więc budżet
1 FA/h znaczy w praktyce „zero fałszywek na mowie". Stąd biorą się szerokie
przedziały ufności przy niskich budżetach (przy 30 FA/h dolna granica CI
schodzi do zera dla obu wariantów). Poprawa to więcej godzin tła, nie inna
metryka.

## Odtworzenie

```bash
python -m comparison.extract  --variant mcu  --workers 4
python -m comparison.extract  --variant full --workers 4
python -m comparison.evaluate --variant mcu
python -m comparison.evaluate --variant full
python -m comparison.mcu_budget
```

Surowe wyniki: `comparison/results/{mcu,full}.json`.
