# Kontrakt danych SNN

Ten dokument definiuje, czym jest **główny zbiór SNN**, jak się go buduje, jak
sprawdza i jak wersjonuje. Jest nadrzędny wobec wszystkiego, co robimy dalej:
każdy trening i każda metryka mają wskazywać wersję zbioru, na której powstały.

Kod: [`snn_pipeline/dataset_contract.py`](../snn_pipeline/dataset_contract.py) —
schemat, reguły grup i taksonomia w jednym miejscu.

---

## Dwie komendy

```bash
# zbuduj wersję (odmawia nadpisania istniejącej)
python snn_pipeline/build_dataset_version.py --version v1.0.0

# zwaliduj wersję (kod wyjścia != 0, gdy cokolwiek krytycznego nie przejdzie)
python snn_pipeline/validate_dataset.py --version v1.0.0
```

Wynik lądowuje w `dataset/versions/<wersja>/`:

| plik | co zawiera |
|---|---|
| `manifest.csv` | jeden wiersz na nagranie — pełen schemat niżej |
| `dataset.json` | metadane wersji: ziarno, commit, inwentarz źródeł, suma kontrolna manifestu |
| `stats.md` | statystyki per split, gotowe do wklejenia w raport |

---

## Schemat manifestu

| kolumna | znaczenie |
|---|---|
| `id` | stabilny identyfikator, liczony ze źródła i ścieżki (`sha1`, 12 znaków) |
| `filepath` | ścieżka względna do korzenia repozytorium |
| `sha256` | suma kontrolna zawartości pliku |
| `bytes` | rozmiar |
| `label` | `positive` \| `negative` |
| `kind` | `positive` \| `stationary` \| `loud_event` \| `speech` \| `animal` |
| `source` | `esc50` \| `datasec` \| `voice` |
| `subclass` | oryginalna klasa w źródle |
| `group_id` | **nagranie źródłowe** — jednostka podziału |
| `split` | `train` \| `val` \| `test` |
| `duration_s`, `sample_rate`, `channels`, `subtype` | zmierzone parametry audio |
| `license` | licencja źródła |

`id` jest liczone ze **ścieżki**, nie z zawartości — dzięki temu przetrwa
rekompresję pliku, a jednocześnie jest identyczne na każdej maszynie.
Za zgodność zawartości odpowiada osobna kolumna `sha256`.

---

## Reguła grupy — najważniejsza rzecz w tym dokumencie

**Grupa to nagranie źródłowe, a nie plik.** Wszystkie pliki z jednej grupy
trafiają do tego samego splitu.

Powód: kilkanaście plików potrafi pochodzić z jednego nagrania. Dzielą wtedy
akustykę pomieszczenia, tło i często tę samą próbkę zdarzenia. Podział po
plikach wsadza je po obu stronach egzaminu, a wynik testu przestaje mierzyć
umiejętność i zaczyna mierzyć pamięć.

| źródło | grupa | dlaczego tak |
|---|---|---|
| `esc50` | `esc50_<fold>_<clipId>` | ujęcia `A` i `B` to **to samo nagranie** pocięte na dwa pliki — take celowo pomijamy |
| `voice` | `voice_synthetic_NNN` | 8888 wycinków pochodzi z zaledwie **207 miksów**; identyfikator miksu jest w nazwie pliku |
| `datasec` | `datasec_<nazwa pliku>` | tu **celowo** jeden plik = jedna grupa: DataSEC nazywa pliki `<Klasa>-NNN.wav` i każdy jest osobnym, wyciętym przez autorów zdarzeniem (sprawdzone: 3226 plików = 3226 grup). Obcięcie końcówki `-NNN` zlepiłoby 109 niezależnych nagrań szkła w jedną grupę |
| dowolne z Freesound | `<źródło>_fs<idUploadu>` | jeden upload bywa pocięty na kilkanaście plików — upload `396289` dał 16, `483590` dwanaście |

Podział robi `GroupShuffleSplit` **osobno w obrębie każdego źródła**, żeby
proporcje 70/15/15 trzymały się w każdym z nich — inaczej jedno duże źródło
zdominowałoby losowanie i któreś ze źródeł mogłoby zniknąć z testu.

**Split jest zapisany w manifeście jako kolumna, nie odtwarzany z ziarna.**
Ziarno służy tylko do powtórzenia budowy; źródłem prawdy o podziale jest plik.
Gdyby split zależał od kodu, wystarczyłaby zmiana kolejności `glob` albo
`sorted`, żeby wszystko się przesunęło.

---

## Taksonomia `kind` — po co, skoro jest `label`

`label` mówi, czy to szkło. `kind` mówi, **jakim rodzajem dźwięku jest negatyw**,
a to jest inne pytanie i ważniejsze dla nas w praktyce.

Główną metryką bramy zawsze-czuwającej jest rozbicie fałszywych alarmów na
ciche tło, głośne zdarzenia i mowę. Dotąd powstawało ono heurystyką na nazwach
plików w `eval_stream.py`. Teraz jest polem w danych.

| `kind` | co to jest | rola w ocenie |
|---|---|---|
| `positive` | tłuczone szkło | recall |
| `stationary` | ciągłe tło bez ostrego ataku — deszcz, wiatr, turbina, pralka, silnik na biegu jałowym | fałszywe alarmy **w ciszy** |
| `loud_event` | głośne transienty — wystrzały, fajerwerki, młot, drzwi, syrena | fałszywe alarmy na **trudnych** negatywach |
| `speech` | mowa i odgłosy ludzkie | osobna kategoria, bo najczęstsza w instalacji |
| `animal` | psy, ptaki, koty | osobna kategoria FA |

### Skąd bierzemy tło stacjonarne

`notebooks/dataset` okazał się archiwalną pozostałością bez unikalnej zawartości,
więc tła stacjonarnego **nie bierzemy z osobnego zbioru** — bierzemy z ciągłych
klas, które już mamy:

- **DataSEC**: `Wind turbine`, `Vehicle idling`, `Vacuum cleaner fan and hairdryer`, `Cicadas and crickets`
- **ESC-50**: deszcz, fale, trzaskający ogień, świerszcze, wiatr, lanie wody, spłuczka, pralka, odkurzacz, tykanie zegara, owady

> **Sprawdzone i odrzucone:** przerwy między zdarzeniami w miksach VOICe wyglądały
> na naturalne źródło tła — 29% timeline'u, około 2,8 h materiału. Pomiar pokazał,
> że to **cisza cyfrowa**: mediana −240 dBFS, 93% przerw poniżej −80 dBFS.
> Wersja `clean` VOICe to dosłownie zdarzenia na ciszy. Realne sceny tła są
> w wariantach `snr_-3dB` i `snr_-9dB`, po 20,3 GB każdy.

**Ograniczenie, o którym trzeba pamiętać:** to jest zastępnik. Żadne z tych
nagrań nie jest tłem docelowego pomieszczenia, więc „fałszywe alarmy w ciszy"
mierzymy na przybliżeniu. Nagranie własnego tonu pomieszczenia zostaje
najlepszą inwestycją w jakość tej metryki.

---

## Wersjonowanie

Semantyczne, `MAJOR.MINOR.PATCH`:

| człon | kiedy | skutek |
|---|---|---|
| **MAJOR** | zmienia się skład zbioru albo podział | stare wyniki **nieporównywalne** |
| **MINOR** | dochodzą rekordy, podział istniejących bez zmian | wyniki porównywalne z zastrzeżeniem |
| **PATCH** | poprawki metadanych, żaden bajt audio się nie zmienia | wyniki porównywalne |

**Wersji nie wolno nadpisywać.** Builder odmawia zapisu do istniejącego katalogu.
Chodzi o to, żeby „wytrenowane na v1.0.0" zawsze znaczyło to samo.

Weryfikacja identyczności na innej maszynie: `dataset.json` niesie
`manifest_sha256`, a manifest niesie `sha256` każdego pliku. Walidator sprawdza
oba i failuje przy dowolnym rozjeździe.

---

## Audio: co standaryzujemy, a czego celowo nie

**Standaryzujemy kontener przez walidację, nie przez transkodowanie.** Manifest
zapisuje zmierzone `sample_rate`, `channels`, `subtype` i `duration_s`, a walidator
odrzuca to, co wypada poza zakres (sr ≥ 16 kHz, ≤ 2 kanały, 0,15–600 s).

**Nie normalizujemy amplitudy i nie transkodujemy.** Trzy powody:

1. Enkoder i tak resampluje do 19 231 Hz (tyle ma ADC Arduino) — drugi resampling
   tylko traciłby jakość.
2. `wav_to_adc_codes()` normalizuje szczytowo **każdy plik osobno**. Wypalenie
   normalizacji w zbiorze skasowałoby bezwzględną głośność na zawsze — a to jest
   cecha, którą realne urządzenie o stałym gainie widzi i która odróżnia cichy
   brzęk od głośnego wystrzału.
3. Kopia 11 GB audio w drugim formacie to koszt bez zysku.

Konwersja jest zadaniem czytającego, nie zbioru.

---

## Relacja: zbiór główny → artefakty spike'owe

```
dataset/versions/vX.Y.Z/manifest.csv        ZBIÓR GŁÓWNY (audio + etykiety + split)
            │
            │  enkoder (encoder_twin.py) + bank cech (build_ext_dataset.py)
            ▼
ga_neuron_search/spikes_ext/                ARTEFAKT SPIKE'OWY (14 kanałów)
architecture_14_neurons_patryk_09_07/
        spikes_v2/                          ARTEFAKT SPIKE'OWY (7 kanałów HW)
        spikes_manifest7/                   ↑ POPRZEDNIK, unieważniony (issue #36)
```

Artefakty spike'owe **nie są śledzone w gicie** — są w całości odtwarzalne
z audio, a `spikes_manifest/` i `spikes_manifest7/` (13.5k plików każdy)
odpowiadają za większość z 1,2 GB w `.git`. Odtworzenie:

```bash
python architecture_14_neurons_patryk_09_07/encoder_twin.py build-manifest \
    --manifest dataset/versions/v2.0.0/manifest.csv --root . \
    --out architecture_14_neurons_patryk_09_07/spikes_v2 --warmup-seconds 30
```

Artefakt spike'owy jest funkcją **trzech** rzeczy: audio, enkodera i banku cech.
Dwa artefakty są porównywalne tylko wtedy, gdy zgadzają się wszystkie trzy.

Dlatego każdy artefakt niesie w `channels.json` blok `provenance`:

```json
"provenance": {
  "dataset_version": "v1.0.0",
  "encoder_file": "encoder_twin.py",
  "encoder_sha256": "…",
  "feature_bank_file": "build_ext_dataset.py",
  "feature_bank_sha256": "…",
  "split": {"grouped_by": "miks VOICe (synthetic_NNN), wspólnie dla obu klas", "seed": 0},
  "stream_shuffled": true,
  "built_utc": "…"
}
```

Zamiast ręcznie wpisywanego numeru wersji enkodera bierzemy **sumę kontrolną
plików, które realnie decydują o wyniku**. Ręczny numer rozjeżdża się po dwóch
tygodniach; suma kontrolna nie.

### Obowiązki buildera artefaktu

Każdy builder artefaktu spike'owego musi:

1. **Przerwać budowę, jeśli jakiś `group_id` jest w więcej niż jednym splicie.**
   Nie ostrzec, przerwać. Brak tej asercji kosztował trzy kampanie treningowe:
   `spikes_manifest7` ma 194 z 194 miksów VOICe obecnych w teście również
   w treningu, więc każda metryka z niego jest liczona na przecieku.
2. **Przeplatać klasy w strumieniu kodowania.** Enkoder utrzymuje jeden ciągły
   stan (floor/MAD) przez cały zbiór; gdy negatywy idą przed pozytywami, stan
   koreluje z etykietą. Sprawdzian: średnia pozycja obu klas w strumieniu ~0.5.
3. **Rozgrzewać floor wyłącznie na tle stacjonarnym** (`kind == stationary`).
   Krótkie zdarzenia (wystrzały, dzwony) ustawiają floor na złym poziomie.
4. **Zapisać obok `channels.json` plik `files.csv`** z kolumnami
   `filepath,label,kind,source,group_id,csv`. Bez niego walidator nie ma jak
   sprawdzić, czy artefakt nie rozjechał się z manifestem (kontrola K9), i
   zgłasza ostrzeżenie O5.

Pliki `files.csv` i `channels.json` leżą w tych samych katalogach co dane, więc
konsumenci globujący `**/*.csv` muszą odfiltrować `files.csv` (robią to
`snn_hw_pipeline.SpikeClips` i `eval_stream.load_clips`).

Regresja na te cztery punkty: `snn_pipeline/tests/test_spike_artifact.py`.

### Status `spikes_ext`

`spikes_ext` jest dziś budowany z `voice_extracted/` (wycinki VOICe), a nie
z zatwierdzonej wersji zbioru głównego. Ma wąską klasę negatywną — wyłącznie
wystrzały i płacz dziecka, bez tła stacjonarnego, ESC-50 i DataSEC.

**To jest poligon do selekcji cech, nie zbiór produkcyjny.** Liczby z niego
nie są porównywalne z liczbami z `spikes_manifest7` i nie powinny trafiać
do `hw_config.json` bez potwierdzenia na zbiorze produkcyjnym.

---

## Próg akceptacji modelu (funkcja celu)

Dotąd repo nie miało ŻADNEJ liczby mówiącej, kiedy model jest gotowy. Wybór
`k=1` vs `k=2`, sweep `pos_weight` i selekcja seeda były rozstrzygane bez
zadeklarowanej funkcji celu (patrz `PRZEWODNIK_KOMPLETNY.md` §15 — kierunek bez
progu). Ta sekcja to ustala.

**Metryka:** recall przy ustalonym budżecie fałszywych alarmów na godzinę tła
(**FA/h**), liczona na poziomie klipu regułą dekodera „≥ k spików neuronu D w
oknie w ramek" (to samo, co zliczy Arduino na J4). Kod: [`snn_pipeline/stream_eval.py`](../snn_pipeline/stream_eval.py)
— jedno źródło dla `eval_stream`, treningu (selekcja checkpointu) i fitnessu GA.

**Zawsze raportujemy** (na splicie `test`, nietkniętym przez trening/selekcję):

- recall **@ 6 FA/h** (bramka always-on: analog zgrubnie, reaktor weryfikuje) —
  punkt główny,
- recall **@ 1 FA/h** (punkt ostry),
- **rozbicie po `kind`** (stationary / loud_event / speech / animal),
- **przedział ufności** bootstrapem po `group_id` (nie po klipie — grupy dzielą
  akustykę; patrz „Reguła grupy").

**Próg „gotowe" (wartość startowa, do rewizji z zespołem):**

> **recall ≥ 0.70 przy ≤ 6 FA/h na `test`, przy czym ŻADEN `kind` nie przekracza
> 6 FA/h.**

Zasady rozstrzygania:

- Punkt pracy (`k`, `w`), a więc i spór **k=1 vs k=2**, wynika z tego progu —
  nie z porównania F1.
- Różnica **mniejsza niż zmierzony szum** (±0.060 clip-F1 wg
  `ga_neuron_search/calibrate-results.txt`; dla recall@FA/h — szerokość CI
  bootstrap) **nie jest opisywana jako poprawa**.
- `spikes_ext` (poligon selekcji cech) nie służy do tej oceny — tylko zbiór
  produkcyjny z zatwierdzonej wersji.

> Wartości 0.70 / 6 FA/h są STARTOWE (uzgodnione na próbę, do przegadania).
> Zmiana progu = zmiana tej sekcji + zgoda zespołu.

---

## Kontrole walidatora

Krytyczne — kończą się kodem błędu:

| kod | co sprawdza |
|---|---|
| `K1` | schemat manifestu, dozwolone wartości, brak powtórzonych `id` |
| `K2` | czy wszystkie pliki istnieją |
| `K3` | **przeciek**: ta sama grupa w dwóch splitach |
| `K4` | rozjazd sum kontrolnych — plik zmieniony po zbudowaniu wersji |
| `K5` | identyczne pliki (ta sama `sha256`) rozrzucone po różnych splitach |
| `K6` | audio nieodczytywalne albo poza dozwolonym zakresem |
| `K7` | za mało **grup** pozytywnych w teście (domyślnie minimum 12) |
| `K8` | podklasa VOICe niezgodna z katalogiem ekstrakcji |
| `K9` | artefakt pochodny ma inną etykietę niż manifest, na który się powołuje |

Ostrzeżenia — nie blokują:

| kod | co sprawdza |
|---|---|
| `O1` | duplikaty treści wewnątrz jednego splitu |
| `O2` | udział pozytywów poza widełkami 1–60% |
| `O3` | brak któregoś `kind` w którymś splicie |
| `O4` | nagrania na licencji niekomercyjnej |
| `O5` | artefakt deklaruje tę wersję, ale nie ma `files.csv` — niesprawdzalny |

`K8` i `K9` istnieją, bo dokładnie te dwa błędy przeżyły w repo miesiące. `K8` łapie
sytuację, w której budowa zbioru wyprowadza etykietę od nowa zamiast wziąć ją
z katalogu ekstrakcji (issue #34: 1412 klipów szkła jako tło). `K9` łapie rozjazd
między manifestem a artefaktem, który się na niego powołuje (zmierzone na v1.0.0:
1653 pliki różniły się etykietą między manifestem a `spikes_ext`). `O5` mówi, że
artefakt w ogóle nie da się sprawdzić, bo nie zapisał ewidencji plików — wtedy `K9`
przechodzi pusto i cisza wygląda jak zgodność.

`K7` istnieje, bo zbiór może przejść kontrolę przecieku i mimo to mieć w teście
kilkanaście niezależnych nagrań szkła. Metryka policzona na takiej próbie ma
ogromny błąd — jedno nietypowe nagranie przesuwa wynik o kilka procent.
**Plików jest dużo, ale jednostką jest nagranie.**

---

## Licencje

| źródło | licencja | użycie komercyjne |
|---|---|---|
| ESC-50 | CC BY-NC 3.0 | **nie** |
| DataSEC | CC BY 4.0 | tak, z atrybucją |
| VOICe | Other (Attribution) | do sprawdzenia przed produktem |

ESC-50 jest niekomercyjne. Manifest niesie kolumnę `license`, więc podzbiór
nadający się do produktu buduje się jednym filtrem
(`license != 'CC BY-NC 3.0'`). Walidator zgłasza to jako ostrzeżenie `O4`,
żeby nikt się nie zdziwił na końcu.

---

## Znane ograniczenia wersji 1.0.0

- Klasa pozytywna jest zdominowana przez wycinki VOICe z 207 miksów — liczba
  **niezależnych** nagrań szkła jest o rząd wielkości mniejsza niż liczba plików.
- Tło stacjonarne jest zastępnikiem z ciągłych klas DataSEC i ESC-50, nie
  nagraniem docelowego pomieszczenia.
- Audio nie jest transkodowane; parametry są tylko mierzone i walidowane.
- ESC-50 wnosi 40 nagrań szkła na licencji niekomercyjnej.

---

## Wycofane: `dataset/combined/manifest.csv`

Plik przemianowany na `manifest.DEPRECATED.csv`. Nie używać.

- 5226 wierszy, **bez ani jednego pliku VOICe**; 149 pozytywów, z czego 16 w teście.
  Jeden klip to 6 punktów procentowych recall.
- Etykiety sprzed naprawy #34.
- Brak kolumny `kind`, więc nie da się rozbić fałszywych alarmów na tło stacjonarne
  / głośne zdarzenia / mowę / zwierzęta.
- Nie jest wersjonowany — brak sumy kontrolnej i commitu, więc „wytrenowane na
  combined/manifest.csv" nie znaczy nic konkretnego.

`DECYZJE_SESJI.md` §C1 opisuje **inny** plik pod tą samą ścieżką (13.5k plików).
Tamten nie istnieje — został zregenerowany do obecnych 5226 wierszy. Artefakt
`spikes_manifest7`, z którego pochodzi `hw7_config.json`, powstał z tej
nieistniejącej wersji i nie da się go odtworzyć (issue #36). Zastąpił go
`spikes_v2` zbudowany z `dataset/versions/v2.0.0`. Żadnej liczby ze starego
artefaktu nie wolno cytować bez przeliczenia na nowym.
