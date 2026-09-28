# Wyniki: SNN vs Fourier

Zbiór `dataset/versions/v2.0.0`, split `test` (1867 klipów, 2,76 h tła), metryka
`snn_pipeline/stream_eval.py`, ramka co 9984 µs. Próg detekcji i reguła k-of-w
wybierane na `val`, raport na `test` z zamrożoną parą i jedną regułą. Przedziały
ufności bootstrapowane po `group_id`. Protokół w `README.md`.

## Krzywa recall ↔ FA/h

| budżet FA/h | Fourier `mcu` | Fourier `full` | SNN (`comparison/evaluate_snn.py`) |
|---|---|---|---|
| 1 | 1,5 % @ 0,4 | nie mieści się na teście | niewykonalne |
| 6 | 4,2 % @ 1,8 | 3,3 % @ 1,5 | niewykonalne |
| 30 | 23,8 % @ 15,2 | 20,0 % @ 9,4 | niewykonalne |
| 120 | 36,5 % @ 28,7 | 37,9 % @ 15,6 | niewykonalne |
| 600 | 73,2 % @ 134,6 | **85,3 % @ 107,4** | 71,4 % @ 175,3 [CI 0–73,9 %] |

Zapis „X % @ Y" to recall przy zmierzonym łącznym FA/h, nie przy budżecie.
Budżet jest tylko ograniczeniem, przy którym wybrano punkt pracy, i wiąże
**każdy** rodzaj tła z osobna. Kolumna SNN jest teraz przeliczona tą samą
uprzężą co obie kolumny Fouriera (`comparison/evaluate_snn.py`), na
zreprodukowanym championie (`rpi_agents/cloud/model`, recall bez ograniczenia
FA/h = 0,828 przy regule k=1 — patrz notatka o rozbieżności niżej).

**Dlaczego SNN nie ma nic przy 1/6/30/120 FA/h, mimo dwóch prób.** Uprząż
wybiera na `val` najlepszą regułę k-of-w, która mieści KAŻDY rodzaj tła w
budżecie. Zbadano to dwoma niezależnymi sposobami, nie jednym:

1. **Rozszerzona siatka reguł** (`comparison/evaluate_snn.py`): oprócz
   `snn_pipeline.stream_eval.DEFAULT_RULES` (do k=4, w=500) doszukano ostrzejsze
   punkty aż do k=20 w oknie 50 s. Nic poniżej 600 FA/h się nie mieści; przy
   120 FA/h jedna reguła (k=15, w=5000) przeszła na `val`, ale nie na `test`
   (uczciwie zgłoszone jako niewykonalne — bez tego nie byłoby to zgodne z
   zasadą „próg wybrany na val, nie na test").
2. **Osobny, wytrenowany od zera wariant o niższym `pos_weight`** (0,3 zamiast
   1,0 — mniej czuły, karany mocniej za fałszywe alarmy): poprawił punkt przy
   600 FA/h (71,4% @ 175,3 zamiast 69,3% @ 184,0), ale **też** nie odblokował
   żadnego niższego budżetu.

Wniosek: to nie jest kwestia dekodera ani wagi pozytywnej klasy przy treningu
— to właściwość modelu/danych. Zgadza się z niezależną obserwacją Kacpra
(`WNIOSKI.md`, inny checkpoint, ten sam wzorzec: „budżet FA/h na poziomie 1
oraz 6 jest całkowicie nieosiągalny").

## Trzy rzeczy, które z tego wynikają

**1. Zmierzone (nie zacytowane): SNN nie wygrywa z Fourierem na osi jakości.**
Fourier bez ograniczeń sprzętowych osiąga 85,3 % recall przy 107,4 FA/h. Ten
sam checkpoint SNN, oceniony identyczną uprzężą (reguła wybrana na val,
zamrożona, raport na test), osiąga przy najluźniejszym z testowanych budżetów
(600 FA/h) tylko 71,4 % recall przy 175,3 FA/h — **mniej recall i więcej
fałszywych alarmów** niż nieograniczony sprzętowo Fourier. To odwraca wniosek,
który stał w tym miejscu, gdy kolumna SNN była jeszcze cytatem z innego
przebiegu i innego dekodera.

*Uwaga o rozbieżności z liczbą 82,8 % (ta, która jest wdrożona).* Wdrożony
model (`rpi_agents/cloud/model`) używa reguły operacyjnej k=1 bez ograniczenia
FA/h — na tym samym checkpoincie to daje recall 82,8 % na teście. Ta tabela
mierzy coś innego: recall przy NAJLEPSZEJ regule z siatki, która mieści się w
zadanym budżecie FA/h. Reguła k=1 nie mieści się w żadnym z testowanych
budżetów (nawet 600) na tym zbiorze tła, więc ta uprząż wybiera łagodniejszą
regułę (`k=2, w=500`) — stąd niższy recall tutaj niż liczba wdrożeniowa. Obie
liczby są prawdziwe i policzone na tym samym checkpoincie; mierzą różne rzeczy:
recall bez ograniczenia kontra recall pod budżetem fałszywych alarmów na
godzinę.

**2. Budżet mikrokontrolera kosztuje mniej więcej dwanaście punktów recall.**
Zejście z pełnego rFFT 512 na sześć pasm liczonych na gołym hopie zabiera
85,3 % → 73,2 % na szczycie krzywej, a przy dopasowanym recall (~37 %) mniej
więcej podwaja FA/h (15,6 → 28,7). To jest pierwsza liczba w tym projekcie,
która mówi, ile naprawdę kosztuje ograniczenie sprzętowe, a nie ile się go
obawiamy.

**3. Ścianą jest mowa i jest wspólna — i dla SNN jest najwyższa z trzech.**
W punkcie o najwyższym mierzonym recall dla każdej strony: Fourier `full` 529
FA/h na mowie, Fourier `mcu` 497, SNN (ta uprząż, `k=2 w=500`) **588**. Trzy
różne front endy, ten sam problem, a SNN nie jest tu wyjątkiem — jest najgorszy.
To nie jest różnica między architekturami, tylko brak rozdzielności mowa/szkło
w danych, na których uczymy. Dopóki to nie pęknie, żaden z tych trzech nie
zbliży się do 6 FA/h.

Przy budżecie 6 FA/h Fourier `mcu` i `full` są praktycznie przy zerze (4,2 %,
3,3 %); SNN nie ma tam żadnej wykonalnej reguły w ogóle. Próg akceptacji z
`docs/DATASET_CONTRACT.md:297` (recall ≥ 0,70 przy ≤ 6 FA/h, żaden `kind`
powyżej) **nie jest dziś osiągany przez nic**.

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

## Domknięcie: skąd wziął się ten checkpoint SNN

Kolumna SNN była wcześniej cytatem z `models/WYNIKI.md` (branch `fah-metric-kn`,
PR #49), z dekoderem k=1 na sztywno i bez wyboru reguły na walidacji. Teraz jest
przeliczona identyczną uprzężą co obie kolumny Fouriera
(`comparison/evaluate_snn.py` — wariant `evaluate.py` dla dekodera zdarzeń
zamiast progu prawdopodobieństwa, bo D nie ma prawdopodobieństwa, tylko binarny
ciąg spike'ów).

Checkpoint to zreprodukowany oryginalny champion (`run_20260927_205133`, PR
#66) — ten sam plik `.pt` zaginął (nigdy niezacommitowany), więc odtworzony
dziś od zera: identyczny config/topologia/seed=42, trening pod `decoder_k=2`
(tak jak oryginał — patrz komentarz w `ga_runner.py:run_final_evaluation_stage`),
odczyt post-hoc przy k=1 na teście zgadza się z tabelą z commita `a58f6056` co
do trzeciego miejsca po przecinku (`clip_f1=0,7086 recall=0,82805
precision=0,6193`). Ten sam plik jest teraz w `rpi_agents/cloud/model` (PR
#72) — **to jest checkpoint, który faktycznie działa na Azure**, z regułą
operacyjną k=1 bez ograniczenia FA/h.

*Ale wiersz 600 FA/h w tabeli wyżej pochodzi z DRUGIEGO, osobnego
checkpointu* (`pos_weight=0,3` zamiast 1,0 — mniej czuły, trenowany od zera na
tej samej topologii/seedzie/decoder_k=2, w ramach próby domknięcia niskich
budżetów opisanej wyżej). Nie jest wdrożony nigdzie — istnieje tylko jako
punkt porównawczy w tej tabeli, bo przy budżecie 600 FA/h wypadł lepiej niż
flagowiec. Dwa różne pliki, dwa różne cele: jeden do wdrożenia (wysoki
recall bez ograniczenia), jeden do tej tabeli (najlepszy punkt pod budżet).

Wciąż brakuje: adaptacyjnej normalizacji cech widmowych (floor/MAD), którą ma
enkoder SNN, a strona Fourierowa nie — to zapas dla Fouriera, nie przeciw
niemu (patrz `README.md`).

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
python -m comparison.evaluate_snn --ckpt <checkpoint.pt>   # rpi_agents/cloud/model/champion_checkpoint.pt
```

Surowe wyniki: `comparison/results/{mcu,full,snn}.json`.
