# Jak to działa — materiał do nauki (Wake-Up AI / Delta Spike)

Ten dokument tłumaczy KONCEPCJE i INTUICJE stojące za tym, co zbudowaliśmy — żebyś umiał
to zrozumieć i wytłumaczyć (np. na obronie/publikacji). Idzie od ogółu do szczegółu. Do
konkretnych decyzji „co i dlaczego wybraliśmy" jest osobny `DECYZJE_SESJI.md`.

---

## 1. Wielki obraz — po co to wszystko

Cel: **detektor tłuczonego szkła always-on**, który zużywa grosze prądu i budzi dopiero
kosztowny system (LLM-reaktor), gdy usłyszy szkło. Klucz: detekcja ma być tak tania, żeby
mogła działać bez przerwy — stąd **analogowe neurony** zamiast procesora liczącego sieć.

Trzy warstwy systemu:
1. **Analog** — fizyczne płytki neuronów LIF „Lu.i". Każda to jeden neuron: 3 wejścia
   (J1/J2/J3, każde z trymerem = waga i przełącznikiem = znak ±), 1 wyjście (J4). Neuron
   „całkuje" impulsy wejściowe i strzela, gdy napięcie membrany przekroczy próg.
2. **Firmware/most** — Arduino zamienia audio z mikrofonu na impulsy TTL (spiki) dla płytek
   i odczytuje decyzję z ostatniego neuronu.
3. **Trening/symulacja (Python)** — bo NIE da się ręcznie nagrać setek tłuczeń szkła, cały
   tor uczenia jest programowy: gotowe nagrania → symulacja enkodera → symulacja sieci →
   trening → tabela nastaw trymerów, którą ręcznie wykręcasz na płytkach.

**Największa idea całości:** trenujemy w symulacji model, który jest WIERNY sprzętowi
(ten sam model neuronu, te same ograniczenia), więc nauczone liczby można przenieść na
pokrętła. Cały projekt to sztuka utrzymania tej wierności.

---

## 2. Model neuronu Lu.i — dlaczego nie „zwykły" LIF z biblioteki

Standardowy `snn.Leaky` z snnTorch ma tylko jedną stałą czasową (membrana). Płytka Lu.i ma
DWA obwody RC: synapsę (prąd) i membranę (napięcie). Model, który trenujemy:

```
I[t] = α·I[t−1] + Σ_j w_j · s_j[t]        α = exp(−dt/τ_syn)   (prąd synaptyczny)
V[t] = β·V[t−1] + (1−β)·V_leak + I[t]     β = exp(−dt/τ_mem)   (napięcie membrany)
spike gdy V[t] ≥ V_th ;  V[t] ← 0 po spiku (reset do zera)
```

- **τ_syn** (5–220 ms): jak długo „pamięta" pojedynczy impuls na synapsie.
- **τ_mem** (20–2200 ms): jak szybko membrana relaksuje do V_leak.
- **V_leak**: poziom spoczynkowy membrany (ustawiany paskiem LED na płytce).
- **V_th = VDD/2**: próg, sprzętowo NIERUCHOMY — nie możemy go uczyć.

**Pułapka, którą trzeba znać (do publikacji):** człon prądowy `I[t]` wchodzi BEZ czynnika
`(1−β)`. Fizycznie prąd synaptyczny nie skaluje się przewodnością upływu. Gdyby go tam
wstawić (łatwa pomyłka), to przy τ_mem ~1 s mamy `1−β ≈ 0.01` i gradient przez trzy warstwy
znika — sieć nigdy nie odpala. Sprawdzone empirycznie.

**Niezmienniczość skali (podstawa całej kalibracji):** pomnożenie wszystkich wag neuronu
przez `k` i zmiana `V_leak' = V_th − k·(V_th − V_leak)` NIE zmienia momentu pierwszego
spiku ze spoczynku — a to jest dokładnie zdarzenie, które wykrywamy. Dlatego można wykręcić
najsilniejszą wagę na pełną skalę trymera i odrobić to zapasem do progu (paskiem LED).

---

## 3. Enkoder — jak audio staje się spikami

Arduino próbkuje mikrofon ~19.2 kHz i co 192 próbek (=10 ms = „ramka" = krok czasowy sieci)
liczy zestaw **cech**, a potem decyduje, czy dana cecha „strzeliła" (impuls na odpowiednim
pinie). Maks 1 impuls na kanał na ramkę → `dt_symulacji = dt_sprzętu = 10 ms`.

### 3a. Cechy czasowe (liczone na całym paśmie)
- `peak` — maks |sygnału| w ramce (ostry transient).
- `peak_cnt` — ile próbek przekroczyło próg mikro-szpilki.
- `cv` — współczynnik zmienności obwiedni.
- `zcr` — zero-crossing rate (zgrubny charakter widma).
- `flux` — dodatni przyrost log-RMS (detektor ataku, nie wygaszania).
- (`crest` = peak/RMS — była, okazała się martwa, usunęliśmy.)

### 3b. Adaptacyjny próg z-score (dlaczego nie stałe progi)
Cechy nie mają stałych progów — każdy kanał utrzymuje **floor** (poziom tła) i **MAD**
(rozrzut), a strzela, gdy `(cecha − floor)/MAD > próg_z`. Floor rośnie WOLNO (A_UP) a spada
SZYBKO (A_DN) — śledzi poziom ciszy. Dzięki temu enkoder nie rozjeżdża się przy zmianie
mikrofonu/poziomu tła. Adaptacja jest ZAMRAŻANA w trakcie zdarzenia, żeby enkoder nie
„przyzwyczaił się" do szkła jako nowego tła.

**Intuicja:** to detektor ZMIANY względem tła. Świetny dla transientów. Ale — patrz niżej —
zły dla cech opisujących POZIOM widma.

### 3c. Cechy widmowe (dodane w v3) ★ sedno poprawy
Problem: wszystkie cechy 3a liczone na pełnym paśmie, więc głośny niskoczęstotliwościowy
łomot wygląda jak szkło. A szkło ma podpis WIDMOWY: dużo energii w 4–10 kHz.

Rozwiązanie: tani filtr 1-biegunowy (`lp += (x−lp)>>1`, cutoff ~2.2 kHz), pasmo górne
`hf = x − lp`, i cecha `hf_ratio = energia_pasma_górnego / energia_ramki`.

**Kluczowa lekcja (do publikacji):** `hf_ratio` MUSI być kodowana progiem BEZWZGLĘDNYM, nie
z-score. Bo hf_ratio to POZIOM (kształt widma), nie transient. Adaptacyjny floor odjąłby
trwale wysokie HF szkła — zmierzyliśmy, że z-score ODWRACA sygnał (kanał strzela rzadziej
dla szkła). Z progiem bezwzględnym (hf_lo>0.28, hf_hi>0.35, kod termometrowy) + bramką
zdarzenia (bo hf_ratio na ciszy = szum z dzielenia): szkło strzela 6–30× częściej niż
negatywy. To jest kanał, który przełamał sufit.

---

## 4. Sieć — topologia wpisana w sprzęt

```
7 kanałów enkodera → H0 H1 H2 H3 (4) → G0 G1 G2 (3) → D (1)     = 15 „neuronów"
```

**Zasada przewodnia:** NIE trenujemy gęstej sieci i nie przycinamy jej potem. Maska
łączności (fan-in ≤ 3, bo płytka ma 3 wejścia) jest w modelu OD PIERWSZEJ EPOKI. Sieć uczy
się w przestrzeni, którą sprzęt realnie potrafi zrealizować. Przedostatnia warstwa ma
DOKŁADNIE 3 neurony = 3 wejścia neuronu decyzyjnego D → D dostaje całą warstwę G, nic nie
jest odrzucane, sygnał nie zanika.

**Jak trenujemy mimo że neuron strzela nieciągle (0/1):** surrogate gradient. W przód
używamy twardej funkcji Heaviside'a (spike gdy V≥V_th), a w tył udajemy, że pochodna to
gładka funkcja (arctan) — dzięki temu gradient płynie przez spiki i można uczyć wag.

**Wagi ↔ sprzęt:** znak wagi = przełącznik ±, wartość = pozycja trymera (kwantyzujemy do
~20 działek, bo tyle realnie ustawi ręka). Znaki ZAMRAŻAMY pod koniec treningu, żeby ostatnie
epoki nie przerzucały polaryzacji.

**HAT + QAT (dlaczego dwie fazy):** każda liczba trafia na fizyczny element, więc:
- HAT (Hardware-Aware Training) — pełna precyzja, ale wstrzykujemy SZUM sprzętowy (rozrzut
  trymera ±½ działki, tolerancja τ ±10%, pasek V_leak ±2%). Rozwiązanie staje się odporne
  na to, że kalibracja ręką nie jest idealna.
- QAT (Quantization-Aware Training) — kwantyzujemy wagi do działek trymera i dostrajamy.
  Tylko skwantyzowany model da się wykręcić na płytce.

---

## 5. Metodyka „najpierw symulacja" — czemu to sedno, nie formalność

Nie mamy jak nagrać szkła i nie chcemy lutować w ciemno. Więc KAŻDĄ zmianę enkodera
walidujemy w cyfrowym bliźniaku (`encoder_twin.py`, wierny 1:1 firmware) na dużym zbiorze,
ZANIM tkniemy Arduino. To nie ostrożność „na wszelki wypadek" — to złapało realny błąd:

> Cecha `hf_ratio` na poziomie surowego audio ładnie rozdzielała klasy (AUC 0.73). Ale po
> zakodowaniu adaptacyjnym z-score sygnał się ODWRÓCIŁ. Gdybyśmy od razu wgrali to na
> płytki i wykręcali trymery, kręcilibyśmy godzinami detektor, który myli szkło z tłem.
> Symulacja pokazała to w minuty i wskazała naprawę (próg bezwzględny).

**Bramka decyzyjna:** zanim rozbudujemy model, sprawdzamy, czy nowa cecha W OGÓLE rozdziela
klasy. Jeśli nie — strojenie/stop, bez marnowania treningu.

---

## 6. Jak mierzymy sukces (i czemu nie „accuracy")

- **F1 na oknach 2 s** — zbalansowana miara przy 10% pozytywów (accuracy per ramka byłaby
  ~99% i nic nie znaczyła, bo tło dominuje).
- **Metryki klipowe** — % nagrań szkła, które budzą system, vs % klipów tła z fałszywym
  alarmem, przy regule dekodera „k spików neuronu D". To przewiduje demo.
- **Rozbicie fałszywych alarmów po źródle** — cisza vs głośne zdarzenia. To ono ujawniło,
  że sufit to enkoder (model radził sobie z ciszą, mylił się na głośnych nie-szkłach).
- **Odporność Monte Carlo** — F1 przy losowym rozrzucie trymerów/τ/V_leak. Przewiduje, czy
  ręczna kalibracja się nie rozsypie. To najważniejsza liczba dla praktyczności.

---

## 7. Co konkretnie osiągnęliśmy w tej sesji

Zamieniliśmy martwą cechę `crest` na dwie cechy WIDMOWE (`hf_lo`/`hf_hi`) i rozbudowaliśmy
sieć do 15 neuronów (7→4→3→1). Na zbiorze testowym (nigdy niewidzianym):

- F1 okienkowe **0.487 → 0.609**
- odporność na rozrzut trymerów (min F1) **0.355 → 0.550** — kalibracja dużo wybaczliwsza
- ten sam recall szkła (**72%**), ale **3× mniej fałszywych alarmów** (63% → 20%)
- fałszywe alarmy na głośnych zdarzeniach (strzały/dzwony/mowa) spadły **3–8×**

Mechanizm: sieć nauczyła się używać `hf_hi` HAMUJĄCO w części neuronów H („mocny HF wygasza
ten neuron") i excytująco w innych — czyli sama zbudowała detektor „głośne ORAZ o widmie
szkła". To była hipoteza, którą wpisaliśmy w architekturę (kanały widmowe zmieszane z
czasowymi w każdej płytce H), i sieć ją potwierdziła.

---

## 8. Słowniczek pojęć (do szybkiego przypomnienia)

- **LIF** — Leaky Integrate-and-Fire, model neuronu: całkuje wejścia, upływa, strzela przy
  progu, resetuje.
- **τ_syn / τ_mem** — stałe czasowe synapsy (prąd) i membrany (napięcie).
- **surrogate gradient** — zastępcza gładka pochodna nieciągłego spiku, żeby dało się uczyć.
- **fan-in** — liczba wejść neuronu; tu ≤ 3 (sprzętowo).
- **HAT / QAT** — trening świadomy sprzętu (z szumem) / świadomy kwantyzacji (działki trymera).
- **z-score adaptacyjny** — próg względem lokalnego tła (floor+MAD); detektor ZMIANY.
- **hf_ratio** — udział energii pasma górnego; deskryptor POZIOMU widma (próg bezwzględny!).
- **reguła k** — „alarm gdy neuron D strzeli ≥ k razy"; k=1 dla nas.
- **brama always-on** — analog wykrywa zgrubnie i tanio, reaktor (LLM) weryfikuje i odrzuca
  fałszywe wybudzenia. W tym układzie przeoczone szkło jest droższe niż fałszywy alarm.

---

## 9. Pliki (mapa projektu po tej sesji)

- `encoder_twin.py` — cyfrowy bliźniak enkodera (cechy, z-score, widmo, build-manifest).
- `encoder_v2.ino` — firmware Arduino (v3: filtr HF, progi bezwzględne, 7. kanał na D8).
- `snn_hw_pipeline.py` — model LuiNet, trening HAT/QAT, maski, eksport nastaw, metryki.
- `eval_stream.py` — metryki klipowe + rozbicie FA po źródle + reguły dekodera.
- `g_tap_eval.py` — przeszukiwanie reguł dekodera na warstwie G (odrzucone).
- `hw_config.json` / `hw7_config.json` — nastawy trymerów: wersja 14- i 15-neuronowa.
- `best.pt` / `best7.pt` — checkpointy zwycięskich modeli.
- `kalibracja_sciaga.md` / `kalibracja_sciaga_v3.md` — ściągi kalibracyjne (14 / 15 neur.).
- `architektura_i_kalibracja.md` — pełny opis architektury i faz kalibracji A–E.
- `DECYZJE_SESJI.md` — dziennik wszystkich decyzji tej sesji.
