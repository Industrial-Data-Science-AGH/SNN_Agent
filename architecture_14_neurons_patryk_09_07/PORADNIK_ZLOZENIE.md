# PORADNIK KROK PO KROKU — złożenie detektora szkła (dla laika)

Ten poradnik prowadzi Cię od pudełka z częściami do działającego detektora tłuczonego
szkła, **bez zakładania że cokolwiek wiesz o elektronice czy sieciach neuronowych**. Rób
kroki PO KOLEI i nie przeskakuj. Każdy krok mówi co zrobić i **co powinieneś zobaczyć**.

Budujemy wersję **15-neuronową** (najlepszą — patrz `DECYZJE_SESJI.md`). Jest też prostsza
14-neuronowa (bez zmian firmware) — o niej wzmianka na końcu.

> ⚠️ Najważniejsza zasada: **składamy i kalibrujemy OD DOŁU DO GÓRY** (najpierw H, potem G,
> potem D) i **jedną płytkę na raz**. Nigdy wszystko naraz. Błąd na jednej płytce jest
> wtedy oczywisty, a w całości ginie w szumie.

---

## CZĘŚĆ 0 — Sprawdź, czy masz wszystko

Odhacz każdą pozycję:

- [ ] **8 płytek Lu.i** — to Twoje neurony. (Masz 15; 8 używamy, 7 to zapas.)
      Na każdej: 3 wejścia synaptyczne (J1/J2/J3) z małym pokrętłem (trymer wagi) i
      przełącznikiem znaku (+/−), regulacja „leak" (V_leak) i dwóch stałych czasowych
      (τ_syn wspólne, τ_mem), pasek diod (pokazuje „napięcie membrany") + jedna dioda
      „spike", 3 terminale wyjściowe (ten sam sygnał, do 3 kabli), gniazdo baterii.
- [ ] **8 baterii CR2032** (po jednej na płytkę) + zapas.
- [ ] **Arduino Uno** (+ kabel USB do komputera).
- [ ] **Mikrofon** — MUSI dawać na wyjściu sygnał wzmocniony i wyśrodkowany na ~połowie
      napięcia zasilania (moduł typu MAX4466 / MAX9814 z kapsułą elektretową, albo Twój
      WM60 na małym przedwzmacniaczu). ⚠️ Goła kapsuła elektretowa NIE wystarczy — patrz
      Część 4.
- [ ] **Kable połączeniowe (jumpery)** — dużo, min. ~40 sztuk (żeński-żeński albo pod
      Twoje złącza). Każde połączenie to 1 kabel sygnału (masę robimy wspólną listwą).
- [ ] **Listwa/szyna masy (GND)** — cokolwiek, do czego zepniesz wszystkie masy
      (płytka stykowa / listwa zaciskowa / lutowana szyna).
- [ ] **Komputer** z zainstalowanym **Arduino IDE** (do wgrania firmware).
- [ ] (opcjonalnie) **telefon z trybem slow-motion** — do dokładnego strojenia τ (Część 7,
      sekcja zaawansowana).

Jeśli czegoś brakuje — zatrzymaj się i dokup, zanim zaczniesz.

---

## CZĘŚĆ 1 — 6 pojęć, które musisz znać (60 sekund)

1. **Neuron (płytka Lu.i)** — sumuje impulsy z wejść; gdy „napięcie" przekroczy próg,
   wypuszcza własny impuls (spike) na wyjściu. Pasek diod pokazuje to napięcie na żywo.
2. **Spike / impuls** — krótki skok napięcia. Tak neurony się „porozumiewają".
3. **Waga (trymer)** — pokrętło przy każdym wejściu: jak mocno to wejście popycha neuron.
4. **Znak (+/−)** — przełącznik: wejście POBUDZA (+) albo HAMUJE (−) neuron.
5. **V_leak / pasek LED** — poziom spoczynkowy „napięcia" neuronu. Ustawiasz go tak, by
   pasek diod stał na zadanej wysokości. Niżej = trudniej odpalić.
6. **Warstwy H → G → D** — H (4 płytki) patrzą na dźwięk, G (3 płytki) łączą ich wyniki,
   D (1 płytka) podejmuje finalną decyzję „szkło / nie szkło".

---

## CZĘŚĆ 2 — Skąd biorą się liczby (nic nie musisz liczyć)

Cała „inteligencja" jest już policzona i zapisana w pliku `hw7_config.json`. Ty tylko
**przepisujesz z tabeli** (Załącznik A) na pokrętła. **Nie musisz nic trenować ani
uruchamiać na komputerze**, żeby złożyć projekt — model jest gotowy.

(Gdybyś kiedyś chciał wytrenować od nowa — instrukcja jest w `kalibracja_sciaga_v3.md`,
sekcja „Reprodukcja". Ale do złożenia NIE jest potrzebna.)

---

## CZĘŚĆ 3 — Wgraj program na Arduino

1. Podłącz Arduino Uno kablem USB do komputera.
2. Otwórz **Arduino IDE**.
3. Otwórz plik `encoder_v2.ino` (z tego katalogu).
4. Menu **Narzędzia → Płytka** → wybierz **Arduino Uno**.
5. Menu **Narzędzia → Port** → wybierz port, który się pojawił po podłączeniu.
6. Kliknij **Upload** (strzałka w prawo).
7. **Co powinieneś zobaczyć:** na dole komunikat „Done uploading". Otwórz **Monitor
   portu szeregowego** (lupa/ikonka, prędkość **115200**) — powinny lecieć linijki typu
   `frame,s0,s1,s2,s3,s4,s5,s6` i dalej same zera (bo cicho).

Jeśli błąd kompilacji — upewnij się, że masz cały plik `.ino` i wybraną płytkę Uno.

---

## CZĘŚĆ 4 — Podłącz mikrofon do Arduino

Arduino czyta dźwięk z pinu **A0**. Sygnał z mikrofonu musi być:
- **wzmocniony** (goła kapsuła daje mikrowolty — za mało),
- **wyśrodkowany na ~2.5 V** (połowa zasilania), bo tak zakłada program.

Dlatego użyj **modułu mikrofonowego z wbudowanym wzmacniaczem** (np. MAX4466 — ma
pokrętło wzmocnienia, albo MAX9814). Podłącz:
- **VCC** modułu → **5V** Arduino,
- **GND** modułu → **GND** Arduino,
- **OUT** modułu → **A0** Arduino.

**Co powinieneś zobaczyć:** w monitorze portu, gdy klaśniesz/stukniesz blisko mikrofonu,
w liniach `frame,...` zaczną pojawiać się **jedynki** (spiki). W ciszy — same zera.
Jeśli w ciszy sypie jedynkami: przykręć wzmocnienie na module w dół. Jeśli nawet przy
głośnym dźwięku same zera: przykręć wzmocnienie w górę.

> To jest też Twój pierwszy sprawdzian, że enkoder żyje. Nie idź dalej, dopóki spiki nie
> reagują sensownie na dźwięk.

---

## CZĘŚĆ 5 — Wspólna masa (PIERWSZY krok sprzętowy, nie pomijaj)

Wszystkie płytki i Arduino muszą mieć **wspólny minus (GND)**, inaczej impulsy nie będą
poprawnie odczytywane.

1. Wyznacz jedną listwę/szynę jako „masę".
2. Połącz z nią: **GND Arduino** oraz **GND (minus) każdej z 8 płytek Lu.i**.
3. **Co powinieneś zobaczyć:** nic spektakularnego — ale to fundament. Zrób to starannie
   teraz, bo szukanie braku masy później to koszmar.

---

## CZĘŚĆ 6 — Włóż baterie

1. Włóż CR2032 do każdej z 8 płytek Lu.i.
2. **Co powinieneś zobaczyć:** pasek diod każdej płytki się zaświeca (pokazuje napięcie
   membrany na jakimś poziomie). Jeśli płytka nie świeci — sprawdź baterię/biegun.

---

## CZĘŚĆ 7 — Ustaw KAŻDĄ płytkę osobno (od dołu do góry: H → G → D)

Rób to dla płytek w kolejności: **H0, H1, H2, H3, potem G0, G1, G2, na końcu D.**
Dla każdej płytki — WSZYSTKIE poniższe kroki, **z ODŁĄCZONYMI wejściami** (kable między
płytkami podłączysz dopiero w Części 8).

Weź tabelę z **Załącznika A** dla danej płytki. Przykład dla H0:
`LED 10% | τ_syn 27ms τ_mem 113ms | J1 peak − 13% (n*=3) | J2 POMIŃ | J3 hf_lo + 57% (n*=1)`

### Krok 7a — stałe czasowe (τ)
Ustaw dwa pokrętła czasu: **τ_syn** i **τ_mem** na wartości z tabeli (w ms). Zidentyfikuj
je na płytce wg dokumentacji Lu.i (jedno reguluje „synaptic time constant", drugie
„membrane time constant"). Dokładność co do milisekundy nie jest krytyczna na tym etapie —
ustaw „na oko" wg skali; dokładne strojenie to sekcja zaawansowana niżej.

### Krok 7b — poziom spoczynkowy (V_leak / pasek LED)
Kręć pokrętłem **V_leak (leak)**, aż pasek diod osiądzie na wartości „LED %" z tabeli
(np. 10% = jedna dioca od dołu, ~26% = ok. 1.5–2 diody). 
⚠️ **Warunek nadrzędny:** dioda „spike" NIE MOŻE migać w spoczynku. Jeśli miga — zejdź
V_leak niżej, niezależnie od tabeli.

### Krok 7c — znaki wejść (+/−)
Ustaw przełącznik **+/−** przy każdym używanym wejściu wg kolumny „znak" z tabeli.
(J2 na H0 jest oznaczone POMIŃ — zostaw to wejście nieużywane, trymer na zero.)

### Krok 7d — wagi (trymery) + test binarny ★ najważniejsze
Dla każdego używanego wejścia:
1. Ustaw trymer mniej więcej na „%” z tabeli (0% = skręcony do zera, 100% = maks).
2. **Zweryfikuj testem impulsowym** (to zamienia „kręć na oko" w pewnik):
   - W monitorze portu szeregowego wpisz komendę: **`C <pin> <n> 100`** i Enter,
     gdzie `<pin>` = numer pinu Arduino tego wejścia (patrz Załącznik B — który kabel),
     `<n>` = liczba `n*` z tabeli. Program wyśle `n` impulsów po 100 Hz na ten pin.
   - **Co powinieneś zobaczyć:** neuron odpala (dioda „spike" mrugnie / wyjście strzeli)
     **dokładnie na n*-tym impulsie** — nie wcześniej, nie później.
   - Odpala za wcześnie (przy mniej niż n*) → trymer za wysoko (przykręć w dół) albo
     pasek LED za wysoko. Nie odpala przy n* wcale → trymer za nisko (podkręć).
   - Rozjazd o ±1 impuls jest OK. O ±3 znaczy, że coś jest nie tak (zły znak? zła płytka?).
3. Powtórz dla każdego używanego wejścia tej płytki.

> Uwaga do testu: żeby testować JEDNO wejście naraz, pozostałe wejścia tej płytki miej
> odłączone (albo ich trymery skręcone na zero). Test podaje impulsy tylko na jeden pin.

Gdy płytka przejdzie test na wszystkich wejściach — jest gotowa. Przejdź do następnej.

### (Zaawansowane, opcjonalne) dokładne strojenie τ — Faza A
Jeśli chcesz maksymalnej wierności: zmierz realne τ każdej płytki nagrywając pasek LED
telefonem w slow-motion (120–240 kl/s) po wywołaniu spiku i licząc czas zaniku do 37%/63%.
Szczegóły w `architektura_i_kalibracja.md`, Faza A. Do działającego prototypu NIE jest to
konieczne — test impulsowy z 7d kompensuje drobne rozjazdy τ przez ustawienie wag.

---

## CZĘŚĆ 8 — Połącz płytki kablami (tabela połączeń)

Teraz, gdy każda płytka jest ustawiona, łączysz je wg **Załącznika B**. Każda linia tabeli
to jeden kabel: od źródła (pin Arduino LUB terminal wyjściowy płytki) do wejścia (J1/J2/J3)
docelowej płytki. Masę już masz wspólną (Część 5), więc to tylko kable sygnału.

Rób to **warstwami**:
1. Najpierw kable **Arduino D2..D8 → wejścia płytek H** (7 sygnałów wejściowych).
2. Potem **wyjścia H → wejścia G**.
3. Potem **wyjścia G → wejścia D**.

⚠️ Dwie synapsy są „martwe" (w tabeli oznaczone POMIŃ): **H0.J2** i **G0.J3**. Tych NIE
podłączaj — zostają puste, trymer na zero. To normalne, sieć ich nie używa.

Fan-out: jedno wyjście płytki ma 3 terminale — jeśli sygnał idzie do 2–3 miejsc (patrz
Załącznik B), użyj kolejnych terminali wyjściowych tej samej płytki.

**Co powinieneś zobaczyć po podłączeniu:** przy dźwięku paski LED płytek H zaczynają
reagować, potem G, a przy szkle (lub nagraniu szkła z głośnika) — pasek/dioda spike
płytki **D**.

---

## CZĘŚĆ 9 — Odczyt decyzji

Wyjście płytki **D** (jej terminal wyjściowy) to sygnał alarmu „słyszę szkło".
- Podłącz go do wolnego pinu Arduino (np. **D9**) albo do wejścia reaktora (np. GPIO
  Raspberry Pi).
- **Reguła decyzji (zalecana): k = 1** — traktuj JAKIKOLWIEK spike z D jako alarm.
- Jeśli w Twoim pokoju tło zbytnio „gada" (za dużo fałszywych alarmów): albo zbieraj
  „2 spiki w oknie 2.5 s" jako warunek alarmu, albo obniż pasek LED płytki D o 1–2 diody
  (twardszy próg) — bez ruszania niczego innego.

---

## CZĘŚĆ 10 — Test końcowy

1. Puść z głośnika na przemian: ciszę, mowę, muzykę i **nagranie tłuczenia szkła**.
2. **Co powinieneś zobaczyć:** dioda spike płytki D odpala głównie na szkle. Trochę
   fałszywych alarmów na innych głośnych dźwiękach jest normalne (to zgrubna brama —
   docelowo reaktor/LLM je odfiltruje).
3. Orientacyjnie (z naszych testów): ~7 na 10 nagrań szkła budzi system; na cichym tle
   prawie nic; na głośnych nie-szkłach czasem strzeli.

Jeśli D nie odpala NIGDY albo odpala CIĄGLE — patrz Część 11.

---

## CZĘŚĆ 11 — Co jeśli nie działa (najczęstsze problemy)

| Objaw | Prawdopodobna przyczyna | Co zrobić |
|---|---|---|
| Płytka nie świeci | bateria / biegun | wymień/obróć CR2032 |
| Enkoder sypie spikami w ciszy | wzmocnienie mikrofonu za duże | przykręć pokrętło na module mic w dół |
| Enkoder milczy na dźwięk | wzmocnienie za małe / zły pin | podkręć wzmocnienie; sprawdź OUT→A0 |
| Dioda spike miga w spoczynku | V_leak za wysoko | zejdź paskiem LED niżej |
| Neuron odpala za wcześnie w teście | trymer wagi za wysoko | przykręć trymer w dół |
| Neuron nie odpala w teście | trymer za nisko / zły znak | podkręć trymer; sprawdź +/− |
| Cała warstwa G/D martwa | brak wspólnej masy | sprawdź, że wszystkie GND na jednej listwie |
| D odpala ciągle | za dużo wejść pobudzających / V_leak D za wysoko | obniż pasek LED D o 1–2 diody |
| Dziwne, niespójne zachowanie po wgraniu firmware | timing ISR (rzadkie) | patrz nota niżej |

**Nota o timingu (dla pewności):** firmware v3 dokłada trochę obliczeń w przerwaniu. Margines
jest duży (~6 µs z ~52 µs), więc na Uno powinno być OK, ale jeśli enkoder zachowuje się
losowo — zgłoś to; można zweryfikować oscyloskopem/analizatorem (miga wolnym pinem na
wejściu przerwania).

---

## ZAŁĄCZNIK A — Pełna tabela nastaw (8 płytek, model seed 2)

Format wejścia: `źródło  znak  trymer%  (n* = ile impulsów do odpalenia w teście)`.

| Płytka | Pasek LED | τ_syn | τ_mem | J1 | J2 | J3 |
|---|---|---|---|---|---|---|
| **H0** | 10% | 27 ms | 113 ms | peak, **−**, 13%, n*=3 | **POMIŃ** (peak_cnt) | hf_lo, **+**, 57%, n*=1 |
| **H1** | 10% | 78 ms | 779 ms | cv, **+**, 49%, n*=2 | zcr, **+**, 35%, n*=2 | hf_hi, **−**, 42%, n*=2 |
| **H2** | 15.5% | 34 ms | 270 ms | peak, **+**, 100%, n*=1 | flux, **+**, 12%, n*=3 | hf_lo, **+**, 12%, n*=3 |
| **H3** | 22.4% | 98 ms | 431 ms | peak_cnt, **−**, 40%, n*=1 | flux, **+**, 15%, n*=2 | hf_hi, **−**, 100%, n*=1 |
| **G0** | 17.1% | 19 ms | 418 ms | H0, **+**, 75%, n*=1 | H1, **−**, 100%, n*=1 | **POMIŃ** (H2) |
| **G1** | 10% | 51 ms | 158 ms | H1, **−**, 14%, n*=3 | H2, **−**, 9%, n*=4 | H3, **+**, 63%, n*=1 |
| **G2** | 10% | 79 ms | 667 ms | H0, **+**, 35%, n*=2 | H2, **−**, 40%, n*=2 | H3, **+**, 25%, n*=2 |
| **D**  | 26.4% | 20 ms | 328 ms | G0, **+**, 29%, n*=2 | G1, **−**, 100%, n*=1 | G2, **−**, 21%, n*=2 |

„Pasek LED %" = gdzie ma stać poziom spoczynkowy na pasku diod. „−" = przełącznik znaku na
minus (hamowanie), „+" = plus (pobudzanie). Źródło prawdy: `hw7_config.json`.

---

## ZAŁĄCZNIK B — Pełna tabela połączeń (kable sygnału)

Kanały enkodera na pinach Arduino: **D2**=peak, **D3**=peak_cnt, **D4**=cv, **D5**=zcr,
**D6**=flux, **D7**=hf_lo, **D8**=hf_hi.

### Arduino → warstwa H
| Kabel od | do | wejście |
|---|---|---|
| Arduino **D2** (peak) | H0 | J1 |
| Arduino **D7** (hf_lo) | H0 | J3 |
| Arduino **D4** (cv) | H1 | J1 |
| Arduino **D5** (zcr) | H1 | J2 |
| Arduino **D8** (hf_hi) | H1 | J3 |
| Arduino **D2** (peak) | H2 | J1 |
| Arduino **D6** (flux) | H2 | J2 |
| Arduino **D7** (hf_lo) | H2 | J3 |
| Arduino **D3** (peak_cnt) | H3 | J1 |
| Arduino **D6** (flux) | H3 | J2 |
| Arduino **D8** (hf_hi) | H3 | J3 |

(H0.J2 — nie podłączaj, POMIŃ.)

### Warstwa H → warstwa G
| Kabel od | do | wejście |
|---|---|---|
| H0 wyjście | G0 | J1 |
| H1 wyjście | G0 | J2 |
| H1 wyjście | G1 | J1 |
| H2 wyjście | G1 | J2 |
| H3 wyjście | G1 | J3 |
| H0 wyjście | G2 | J1 |
| H2 wyjście | G2 | J2 |
| H3 wyjście | G2 | J3 |

(G0.J3 — nie podłączaj, POMIŃ.)

### Warstwa G → neuron D
| Kabel od | do | wejście |
|---|---|---|
| G0 wyjście | D | J1 |
| G1 wyjście | D | J2 |
| G2 wyjście | D | J3 |

### Wyjście decyzji
| Kabel od | do |
|---|---|
| D wyjście | Arduino D9 (lub GPIO reaktora) |

Fan-out (ile kabli wychodzi z danego wyjścia — użyj kolejnych terminali wyjściowych):
H0→2, H1→2, H2→2, H3→2. Piny Arduino: D2→2, D6→2, D7→2, D8→2, D3→1.

---

## Wariant prostszy (14 neuronów) — gdyby coś

Jeśli nie chcesz ruszać firmware ani podłączać D8, jest starsza wersja 6-kanałowa
(`hw_config.json` + `kalibracja_sciaga.md`) — słabsza (3× więcej fałszywych alarmów), ale
bez zmian w firmware. Ta sama procedura składania, tylko inne liczby i 6 kanałów (D2..D7).
Rekomendacja: skoro firmware v3 już jest gotowy, złóż wersję 15-neuronową — jest wyraźnie
lepsza.
