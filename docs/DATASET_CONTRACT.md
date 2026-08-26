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
        spikes_manifest7/                   ARTEFAKT SPIKE'OWY (7 kanałów HW)
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

### Status `spikes_ext`

`spikes_ext` jest dziś budowany z `voice_extracted/` (wycinki VOICe), a nie
z zatwierdzonej wersji zbioru głównego. Ma wąską klasę negatywną — wyłącznie
wystrzały i płacz dziecka, bez tła stacjonarnego, ESC-50 i DataSEC.

**To jest poligon do selekcji cech, nie zbiór produkcyjny.** Liczby z niego
nie są porównywalne z liczbami z `spikes_manifest7` i nie powinny trafiać
do `hw_config.json` bez potwierdzenia na zbiorze produkcyjnym.

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

Ostrzeżenia — nie blokują:

| kod | co sprawdza |
|---|---|
| `O1` | duplikaty treści wewnątrz jednego splitu |
| `O2` | udział pozytywów poza widełkami 1–60% |
| `O3` | brak któregoś `kind` w którymś splicie |
| `O4` | nagrania na licencji niekomercyjnej |

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
