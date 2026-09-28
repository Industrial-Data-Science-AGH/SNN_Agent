# SNN vs Fourier — protokół porównania

Po decyzji opiekuna (27.09.2026) projekt jest prototypem programowym bez
sprzętu, a głównym wynikiem jest porównanie sieci impulsowej z klasycznym
podejściem widmowym. Ten katalog dostarcza **stronę fourierowską** i wspólną
ramę pomiarową. Strona SNN mieszka w `architecture_14_neurons_patryk_09_07/`
i `snn_runtime/`.

## Zasada: jedno kryterium, jedna oś czasu, jeden podział

Porównanie jest warte tyle, ile jego najsłabsze założenie, więc trzy rzeczy są
wspólne i nie wolno ich różnicować.

**Metryka.** `snn_pipeline/stream_eval.py`, ta sama funkcja, którą wybierany
jest model SNN: *recall przy ustalonym budżecie fałszywych alarmów na godzinę*,
z rozbiciem po rodzaju tła i z przedziałem ufności bootstrapowanym po
`group_id`. Nie accuracy, nie AUC ramkowe, nie F1. Detektor alarmowy ocenia się
tym, ile przegapi przy liczbie fałszywek, którą użytkownik zniesie.

**Oś czasu.** Ramka co 192 próbki przy ~19231 Hz, czyli co 9984 µs — dokładnie
siatka enkodera Lu.i. Gdyby obie strony liczyły ramki inaczej, reguła k-of-w
i FA/h przestałyby być porównywalne.

**Podział.** `dataset/versions/v2.0.0`, kolumna `split`. Sprawdzone przed
użyciem: 4878 grup, **zero grup w więcej niż jednym splicie**, etykiety zgodne
z katalogiem pochodzenia (4444 klipy z `glass/` jako pozytywy, wszystkie
`hard_negative/` jako negatywy). To istotne, bo poprzednia wersja zbioru miała
przeciek po grupach i odwrócone etykiety na 1653 klipach.

**Etykieta ramki** to etykieta klipu rozdana na jego ramki. To jest handicap
i jest celowy: `snn_hw_pipeline.py:319` robi dokładnie to samo dla SNN. Danie
Fourierowi prawdziwych granic zdarzeń z adnotacji VOICe, podczas gdy SNN uczy
się z etykiet klipowych, przechyliłoby porównanie w drugą stronę.

**Wybór punktu pracy** (próg detekcji i reguła k-of-w) odbywa się na `val`,
a raport powstaje raz, na `test`, z zamrożoną parą. `stream_report` dostaje na
teście **jedną** regułę, bo mając całą siatkę wybrałby najlepszą dla danych,
które zobaczył, czyli wybierałby punkt pracy na zbiorze testowym.

## Dwa warianty Fouriera, bo jeden byłby nieuczciwy

Zdanie „Fourier wypadł gorzej" jest wynikiem tylko wtedy, gdy nie znaczy
„nikt go nie dostroił". Dlatego są dwa.

| wariant | co widzi | po co |
|---|---|---|
| `full` | rFFT 512 punktów na ramkę, 24 pasma logarytmiczne, centroid, płaskość, rolloff, energia, plus przyrosty | sufit, jaki Fourier w ogóle osiąga na tych danych, bez żadnego budżetu sprzętowego |
| `mcu` | 6 magnitud pasmowych liczonych na gołym hopie 192 próbek, plus energia, plus przyrosty | to, co ATmega328P mogłaby realnie policzyć |

Oba dostają ten sam klasyfikator ramkowy i tę samą warstwę decyzyjną, więc
jedyna różnica między nimi to ilość widma, na którą stać sprzęt.

## Trzecia kolumna: koszt na MCU

Projekt twierdzi nie tylko, że SNN wykrywa szkło, ale że robi to tam, gdzie
Fourier się nie mieści. To jest teza o cyklach, więc ma osobne źródło
(`mcu_budget.py`) i osobny status dowodowy.

**Zmierzone** przez Kacpra na płytce
(`encoder/features-improvement/measurements.json`, sekcja `board`):
ATmega328P @ 16 MHz, ramka 9983 µs = 159 728 cykli, ISR per próbka 590,5 cykla
czyli **70,97% CPU**, obróbka ramki 2885 µs.

```
  zmierzony ISR                  113 359 cykli
  zmierzona obróbka ramki         46 160 cykli
  wolne                              209 cykli   (99,87 % zajęte)
```

**Oszacowane** tutaj, z pokazaną arytmetyką i jawnie oznaczone jako szacunek:
FFT 256-punktowe to 1024 motylki, czyli 61 440–102 400 cykli przy 60–100
cyklach na motylek (30 720 przy optymistycznych 30). Sześć binów Goertzela po
192 próbki to 11 520–18 432 cykli.

Wniosek, odporny na to, że mogę się mylić co do stałej: **przy obecnym
firmware nie mieści się nic**, bo wolnych cykli jest 209. Front end widmowy ma
sens wyłącznie jako **zamiennik** obecnych siedmiu kanałów, i wtedy budżet to
46 369 cykli — w którym sześć binów Goertzela mieści się spokojnie, a FFT 256
tylko przy szacunku, w który sam nie wierzę, i bez miejsca na cokolwiek dalej.
Dlatego `mcu` to sześć pasm, a nie FFT.

## Ograniczenie, które trzeba podać razem z wynikiem

W teście jest **2,76 h tła**, rozbite na `loud_event` 1,43 h, `stationary`
0,67 h, `animal` 0,44 h, `speech` 0,22 h. Przy takim materiale jeden fałszywy
alarm na mowie to już 4,5 FA/h, więc budżet 1 FA/h znaczy w praktyce „zero
fałszywek na mowie", a nie „jedna na godzinę". Rozdzielczość pomiaru jest
gruba i przedział ufności to pokaże. Sensowna poprawa to dłuższe tło, a nie
inna metryka.

## Jak odtworzyć

```bash
python -m comparison.extract  --variant mcu  --workers 4
python -m comparison.extract  --variant full --workers 4
python -m comparison.evaluate --variant mcu
python -m comparison.evaluate --variant full
python -m comparison.mcu_budget
```

Ekstrakcja przechodzi raz przez 20,8 h audio i zapisuje cechy ramkowe do
`comparison/cache/`; ocena czyta cache, więc powtórzenie eksperymentu z inną
regułą lub innym progiem trwa sekundy. Wyniki lądują w `comparison/results/`.
