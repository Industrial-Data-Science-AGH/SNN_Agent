# WAKE-UP AI — KOMPLETNY PRZEWODNIK: od zera do działającego detektora szkła

**To jest JEDYNY dokument, którego potrzebujesz do zbudowania projektu.** Zawiera wszystko:
części, teorię w pigułce, firmware, montaż, pełną kalibrację co do najmniejszego kroku,
aktualne wartości trymerów, know-how i rozwiązywanie problemów.

> **Status wersji (aktualne na dziś):** budujemy wersję **v3 — 8 płytek Lu.i, 7 kanałów**.
> Źródło prawdy dla liczb: **`hw7_config.json`** (model `best7.pt`).
> - `kalibracja_sciaga_v3.md` — **aktualna**, ale ten dokument ją zawiera i rozszerza.
> - `kalibracja_sciaga.md` + `hw_config.json` — **STARE** (wersja 6-kanałowa, 14-neuronowa). Ignoruj.
> - Wariant 15-płytkowy — **przetestowany i ODRZUCONY** (patrz Dodatek B). Nie buduj.

---

# CZĘŚĆ I — ZROZUM, CO BUDUJESZ

## 1. Idea w 60 sekund

Chcemy detektor tłuczonego szkła, który działa **bez przerwy** i zużywa grosze prądu, a budzi
kosztowny system (LLM) dopiero, gdy usłyszy szkło. Dlatego detekcję robią **analogowe neurony**
(płytki Lu.i), a nie procesor.

Łańcuch: **mikrofon → Arduino (enkoder) → 7 kanałów impulsów → 8 neuronów analogowych → decyzja**.

Arduino co 10 ms liczy 7 „cech" dźwięku i na każdą wysyła (lub nie) impuls na osobny pin.
Neurony sumują te impulsy i ostatni z nich (D) strzela, gdy „to brzmi jak szkło".

## 2. Sieć — kto komu podaje

```
7 kanałów (Arduino D2..D8)
     │
     ├──► H0  H1  H2  H3     warstwa ukryta (4 płytki) — patrzą na dźwięk
     │      └───┴───┴──► G0  G1  G2    warstwa łącząca (3 płytki)
     │                       └──┴──► D    neuron decyzyjny (1 płytka)
     │                                 └──► alarm
```

Dlaczego akurat tak: płytka ma **3 wejścia**, więc D (decyzyjny) może odczytać dokładnie 3
neurony — dlatego warstwa G ma **dokładnie 3** płytki. Cała topologia jest wpisana w
ograniczenia sprzętu od początku, nic nie jest „przycinane" po treningu.

**Razem 8 płytek.** Masz 15 → 7 zostaje jako zapas (i dobrze, patrz rozdział 14).

## 3. Kanały — co „słyszy" każdy pin

| # | Pin | Kanał | Co wykrywa | Jak progowany |
|---|---|---|---|---|
| 0 | D2 | `peak` | ostry transient (obwiednia) | adaptacyjny z-score |
| 1 | D3 | `peak_cnt` | liczba mikro-szpilek w ramce | adaptacyjny z-score |
| 2 | D4 | `cv` | zmienność obwiedni | adaptacyjny z-score |
| 3 | D5 | `zcr` | zero-crossing rate | adaptacyjny z-score |
| 4 | D6 | `flux` | narastanie głośności (atak) | adaptacyjny z-score |
| 5 | D7 | `hf_lo` | **udział energii >2.2 kHz, czuły** | **próg BEZWZGLĘDNY 0.28** |
| 6 | D8 | `hf_hi` | **udział energii >2.2 kHz, mocny** | **próg BEZWZGLĘDNY 0.35** |

**`hf_lo`/`hf_hi` to serce projektu.** Szkło ma dużo energii w wysokich częstotliwościach
(4–10 kHz), a głośne nie-szkła (łomot, trzaśnięcie drzwiami, strzał) są niskopasmowe. Te dwa
kanały ścięły fałszywe alarmy **3×**. Reszta kanałów mówi „coś głośnego się stało" — te dwa
mówią „i brzmi to jak szkło".

## 4. Model neuronu — co się dzieje na płytce

Neuron to „wiaderko z dziurą":
- Impuls na wejściu **dolewa** (waga = ile dolewa; znak `+` dolewa, `−` **wylewa**).
- Poziom sam **wycieka** z powrotem do poziomu spoczynkowego **V_leak**.
- Gdy poziom przekroczy **próg** → neuron **strzela** (dioda spike) i poziom **zeruje się**.

Pasek 6 diod **pokazuje ten poziom na żywo**. To Twój oscyloskop — patrz na niego przez całą
kalibrację.

**Próg jest sprzętowo nieruchomy** (VDD/2). Nie da się go ustawić. Jedyne, co ustawiasz, to:
- **V_leak** (jak wysoko poziom stoi w spoczynku = jak blisko progu startujesz),
- **wagi** (jak mocno każde wejście dolewa/wylewa).

---

# CZĘŚĆ II — CZĘŚCI I PŁYTKA

## 5. Lista części

- [ ] **8 płytek Lu.i** (+ zapas; masz 15)
- [ ] **8× bateria CR2032** + zapas
- [ ] **Arduino Uno** + kabel USB
- [ ] **Mikrofon ZE WZMACNIACZEM** — patrz rozdział 9. ⚠️ Goła kapsuła WM60 **nie wystarczy**.
- [ ] **Kable jumperowe** — min. 30 szt.
- [ ] **Listwa/szyna masy** (płytka stykowa lub listwa zaciskowa)
- [ ] **Mały, pasujący wkrętak precyzyjny** do trymerów ⚠️ **kluczowe** (patrz 11.0)
- [ ] Komputer z **Arduino IDE**

## 6. Mapa płytki Lu.i — co jest czym

**Trymery (6 sztuk):**

| Trymer | Funkcja | Wartość |
|---|---|---|
| **RV1** | waga wejścia **J1** | 22 kΩ |
| **RV2** | waga wejścia **J2** | 22 kΩ |
| **RV3** | waga wejścia **J3** | 22 kΩ |
| **RV4** | stała czasowa (τ) | 22 kΩ |
| **RV5** | **V_leak** (poziom spoczynkowy) | 100 kΩ |
| **RV6** | stała czasowa (τ) | 50 kΩ |

> **Uczciwa uwaga:** `RV5 = V_leak` jest **potwierdzone** (schemat: wpięty w VDD + Wasza
> dokumentacja kalibracji). `RV1–RV3 = wagi` — pewne (3 synapsy, 3 trymery). **RV4 vs RV6:
> to dwie stałe czasowe (τ_syn i τ_mem), ale nie dało się ze schematu jednoznacznie ustalić,
> który jest który.** Nie zgaduję. Dobra wiadomość: **do działającego prototypu NIE musisz
> ich ruszać** — zostaw je na środku zakresu (patrz 11.5).

**Złącza:**
- **J1, J2, J3** — wejścia (synapsy). Tu wpinasz sygnały.
- **J4 / terminale wyjściowe** — wyjście (3 terminale, ten sam sygnał → możesz nim zasilić do 3 płytek).
- **GND** — masa. **Musi** iść do wspólnej szyny.

**Diody:**
- **Pasek 6 diod** = poziom membrany na żywo.
- **Próg jest na 3. diodzie** (środek paska = 50%).
- **7. dioda (osobna)** = spike (neuron strzelił).

## 7. Czy muszę coś liczyć / trenować?

**Nie.** Model jest wytrenowany, wszystkie liczby są w `hw7_config.json` i przepisane do tabeli
w rozdziale 12. Ty tylko przepisujesz je na trymery. (Trening od zera — Dodatek A, opcjonalny.)

---

# CZĘŚĆ III — URUCHOMIENIE

## 8. Firmware na Arduino

1. Podłącz Arduino USB.
2. Arduino IDE → otwórz **`encoder_v2.ino`**.
3. **Narzędzia → Płytka → Arduino Uno**; **Narzędzia → Port →** (Twój port).
4. **Upload**.
5. Otwórz **Monitor portu szeregowego**, prędkość **115200**, zakończenie linii **Newline**.

**Co masz zobaczyć:** nagłówek `frame,s0,s1,s2,s3,s4,s5,s6` i lecące linijki z zerami (w ciszy).

## 9. Mikrofon — trzy drogi

Arduino czyta A0. Sygnał musi być **wzmocniony** i **wyśrodkowany na ~2.5 V**.

**Dlaczego goła kapsuła nie wystarczy:** ADC ma ~5 mV na schodek. Goła kapsuła daje przy głośnym
dźwięku ~10–50 mV = kilka schodków. Najbardziej cierpią wtedy `hf_lo`/`hf_hi` (liczą energię w
wysokim paśmie, gdzie sygnału jest mało) — czyli tracisz to, co daje projektowi przewagę.

> Ciekawostka i dobra wiadomość: enkoder jest **odporny na poziom sygnału** (cechy czasowe to
> z-score względem tła, a hf to **stosunek** energii). Więc nie musisz trafić w konkretną
> amplitudę — musisz tylko wyjść ponad szum kwantyzacji ADC.

**Opcja A (zalecana): moduł ze wzmacniaczem** — MAX4466 (z pokrętłem wzmocnienia) lub MAX9814.
```
moduł VCC → Arduino 5V
moduł GND → Arduino GND
moduł OUT → Arduino A0
```

**Opcja B: własny preamp** — LM386 albo LM358 (gain ~50–100×), albo pojedynczy tranzystor.

**Opcja C: masz „sound sensor" KY-037/KY-038?** Jego wyjście **AO** (analogowe) jest już wzmocnione → AO na A0.

**Opcja D (awaryjna, bez części):** kapsuła + rezystor polaryzujący + przełączenie ADC na
wewnętrzne **1.1 V** (`ADMUX` → `INTERNAL`, bias na ~0.55 V). Daje ~5× czułości. Zadziała na
głośne, bliskie szkło — ale cechy widmowe będą słabsze. Do testu toru, nie na demo.

**Test:** klaśnij przy mikrofonie → w monitorze pojawiają się **jedynki**. W ciszy — zera.
Sypie jedynkami w ciszy → zmniejsz wzmocnienie. Cisza przy hałasie → zwiększ.

**Nie idź dalej, dopóki to nie działa.**

## 10. Masa i zasilanie

**KROK 1 — wspólna masa (najważniejszy krok sprzętowy):**
Połącz **GND Arduino** i **GND każdej płytki** do jednej szyny.

> ⚠️ **Brak wspólnej masy = płytka nie widzi impulsów = „nic nie działa".** To przyczyna #1
> objawu „kręcę i nic". Impuls bez wspólnego odniesienia nie istnieje dla płytki.

**KROK 2 — baterie:** CR2032 do każdej płytki. Pasek LED powinien się zaświecić.
Nie świeci nic → bateria/biegun.

---

# CZĘŚĆ IV — KALIBRACJA (serce projektu)

## 11.0 ⚠️ ZANIM DOTKNIESZ TRYMERA — jak nie urwać (przeczytaj!)

Trymery to małe elementy 3/8″ (typ Bourns 3266). **Jednoobrotowe mają twarde ograniczniki
na ~270°.** Przekręcenie za ogranicznik **wyłamuje wiper** — i trymer jest do wymiany.

**Zasady:**
1. **Mały, pasujący wkrętak** (precyzyjny). Za duży ślizga się i wyłamuje.
2. **Zero docisku w dół.** Kręcisz lekko, samym obrotem.
3. **Wyczułeś opór → STOP.** To koniec zakresu. Kręć w drugą stronę. **Nigdy nie forsuj.**
4. Trymer ustawia się **lekko**. Jak musisz naciskać — coś robisz źle.

> Realny koszt złamania tej zasady w tym projekcie: **3 urwane trymery na jednej płytce.**
> Płytka poszła do wymiany. Masz zapas, ale nie marnuj go.

## 11.1 Know-how: jak w ogóle działa kalibracja

Trzy fakty, z których wynika CAŁA procedura:

**1. Sam trymer nic nie robi.** Waga to tylko „ile dolewa impuls". Bez impulsów nic się nie
stanie. **Musisz podać impulsy testowe komendą CALIB** (albo mieć realny dźwięk). Kręcenie
trymerem w ciszy = kręcenie w próżni.

**2. V_leak decyduje, jak daleko masz do progu.** Pasek stoi nisko (nasze cele to 10–26%) =
duży zapas = neuron potrzebuje realnego dowodu, żeby odpalić. Podniesienie V_leak = łatwiej
odpala. **V_leak za wysoko (nad progiem) = neuron strzela sam z siebie** — to nie awaria, to
sygnał, że przesadziłeś.

**3. Test binarny zamiast „na oko".** Dla każdej synapsy tabela podaje **n\*** = ile impulsów
100 Hz na TO JEDNO wejście ma wystarczyć do odpalenia. Ustawiasz wagę tak, żeby odpalało
**dokładnie na n\*-tym impulsie**. To jest mierzalne, w odróżnieniu od „kręć aż zamiga".

## 11.2 Układ do kalibracji JEDNEGO wejścia

```
Arduino D<pin> ────────────►  płytka, wejście J<n>   (terminal synapsy, NIE wyjście!)
Arduino GND    ────────────►  płytka, GND            ◄── bez tego NIC nie zadziała
płytka: bateria CR2032 w środku
pozostałe dwa wejścia: ODŁĄCZONE (albo ich trymery na 0)
```

**Komenda testowa** (w Monitorze portu, 115200, Newline):
```
C <pin> <n> 100
```
np. `C 7 20 100` = 20 impulsów po 100 Hz na pin D7.
Zakres pinów: **2–8**. Podczas testu enkoder jest wyłączony (nie przeszkadza).

## 11.3 Procedura dla JEDNEJ płytki (rób w kolejności!)

**Kolejność płytek: H0 → H1 → H2 → H3 → G0 → G1 → G2 → D. Nigdy odwrotnie, nigdy naraz.**
(Błąd na jednej płytce jest oczywisty w izolacji, a w całości ginie w szumie.)

Dla płytki weź jej wiersz z tabeli (rozdział 12). Przykład H0:
`LED 10% | J1: peak, −, 12.8%, n*=3 | J2: POMIŃ | J3: hf_lo, +, 57.4%, n*=1`

**Krok A — τ (opcjonalny, patrz 11.5).** Zostaw RV4/RV6 na środku zakresu.

**Krok B — znaki.** Ustaw przełączniki `+`/`−` przy J1/J2/J3 wg tabeli.
> ⚠️ Znak `−` = wejście **wylewa** (hamuje). Jeśli testujesz synapsę ze znakiem `−`, ona
> **nigdy nie odpali neuronu** — to normalne! Test n\* robisz przy `+`; dla synaps `−`
> ustaw trymer na % z tabeli i zweryfikuj później na poziomie warstwy (11.6).

**Krok C — V_leak (RV5).** Kręć **RV5**, aż pasek osiądzie na `LED %` z tabeli.
- **Warunek nadrzędny: 7. dioda NIE MOŻE migać w spoczynku.** Miga → zjedź niżej,
  niezależnie od tabeli.

**Krok D — wagi, po jednej.** Dla każdego używanego wejścia ze znakiem `+`:
1. Pozostałe wejścia odłączone / trymery na 0.
2. Ustaw trymer (RV1 dla J1, RV2 dla J2, RV3 dla J3) mniej więcej na % z tabeli.
3. Wyślij `C <pin> <n*> 100`.
4. **Ma odpalić dokładnie na n\*-tym impulsie.**
   - Odpala wcześniej → waga za wysoko, **przykręć w dół**.
   - Nie odpala → waga za nisko, **podkręć**.
   - Rozjazd ±1 impuls = OK. ±3 = coś jest nie tak (zły znak? zła płytka? zły pin?).
5. Powtórz dla kolejnego wejścia.

**Krok E — pomiń martwe.** `POMIŃ` w tabeli = synapsa nieużywana. Trymer na **0**, kabla
**nie podłączaj**. To normalne, sieć jej nie używa.

## 11.4 ★ Drabinka diagnostyczna — gdy „nic nie działa"

Rób PO KOLEI. Każdy krok odcina jedną możliwość.

**Krok 1 — czy płytka żyje?**
Bateria w środku → pasek świeci w spoczynku? **Nie** → bateria/biegun. **Tak** → dalej.

**Krok 2 — czy RV5 (V_leak) w ogóle działa?** (bez Arduino, bez impulsów!)
Kręć **RV5 powoli przez cały zakres** i patrz na pasek.
- **Pasek jedzie w górę, a przy górze 7. dioda zaczyna sama migać** → ✅ RV5 działa, płytka OK.
  (To samowzbudzenie jest **oczekiwane** przy V_leak nad progiem — nie awaria.)
- **Pasek stoi w miejscu przez cały obrót** → ❌ to nie RV5 (kręcisz wagę albo τ) **albo**
  płytka uszkodzona. Znajdź właściwy RV5.

**Krok 3 — czy impuls dochodzi?**
1. Ustaw RV5 tak, żeby pasek stał **tuż pod progiem** (2. dioda, 7. jeszcze nie miga).
2. Waga testowanego wejścia (`+`) na maks.
3. `C <pin> 5 100`.
- **Mrugnęło** → ✅ cała ścieżka OK (masa, kabel, pin, synapsa). Zjedź RV5 do docelowego %
  i szukaj wagi dla n\*.
- **Pasek nawet nie drgnął** → ❌ **problem to masa albo kabel**, nie waga:
  - GND Arduino ↔ GND płytki?
  - Kabel na **wejściu J**, nie na wyjściu?
  - Ten pin co trzeba? (tabela, rozdz. 13)
  - Znak na `+`? (przy `−` membrana idzie w dół!)

**Zasada:** najpierw udowodnij, że **cokolwiek** mrugnie (RV5 wysoko + waga maks), potem
dopiero celuj w docelowe wartości. Nie odwrotnie.

## 11.5 τ (stałe czasowe) — dlaczego możesz je pominąć

Tabela podaje docelowe τ_syn/τ_mem, ale:
- Nie potwierdziliśmy jednoznacznie, który trymer to który (RV4 vs RV6).
- **Test n\* z kroku D kompensuje drobne rozjazdy τ** przez ustawienie wagi.

**Zalecenie na prototyp: zostaw RV4/RV6 na środku i nie ruszaj.** Dokładne strojenie τ
(nagrywanie paska LED telefonem w slow-motion 120–240 kl/s, liczenie zaniku do 37%/63%) to
krok **zaawansowany i opcjonalny** — opisany w `architektura_i_kalibracja.md`, Faza A.

## 11.6 Weryfikacja warstwy (po skalibrowaniu wszystkich płytek warstwy)

Podłącz warstwę, puść dźwięk, patrz na paski. H reaguje na dźwięk → G reaguje → D strzela na
szkle. Rozjazd tylko na jednej płytce → wróć do **tej jednej** (krok D), nie do całości.

---

# CZĘŚĆ V — LICZBY

## 12. TABELA NASTAW — wszystkie 8 płytek (aktualne, z `hw7_config.json`)

Format: `źródło, znak, trymer%, (n* = impulsów do odpalenia)`.
Trymer: **RV1→J1, RV2→J2, RV3→J3**. Pasek LED ustawiasz **RV5**.

### Warstwa H (wejścia = kanały enkodera)

| Płytka | LED (RV5) | τ_syn | τ_mem | **J1 (RV1)** | **J2 (RV2)** | **J3 (RV3)** |
|---|---|---|---|---|---|---|
| **H0** | **10%** | 27 ms | 113 ms | peak, **−**, **12.8%**, n*=3 | **POMIŃ** (0%) | hf_lo, **+**, **57.4%**, n*=1 |
| **H1** | **10%** | 78 ms | 779 ms | cv, **+**, **49.1%**, n*=2 | zcr, **+**, **35.1%**, n*=2 | hf_hi, **−**, **42.1%**, n*=2 |
| **H2** | **15.5%** | 34 ms | 270 ms | peak, **+**, **100%**, n*=1 | flux, **+**, **12.5%**, n*=3 | hf_lo, **+**, **12.5%**, n*=3 |
| **H3** | **22.4%** | 98 ms | 431 ms | peak_cnt, **−**, **40%**, n*=1 | flux, **+**, **15%**, n*=2 | hf_hi, **−**, **100%**, n*=1 |

### Warstwa G (wejścia = wyjścia H)

| Płytka | LED (RV5) | τ_syn | τ_mem | **J1 (RV1)** | **J2 (RV2)** | **J3 (RV3)** |
|---|---|---|---|---|---|---|
| **G0** | **17.1%** | 19 ms | 418 ms | H0, **+**, **75%**, n*=1 | H1, **−**, **100%**, n*=1 | **POMIŃ** (0%) |
| **G1** | **10%** | 51 ms | 158 ms | H1, **−**, **13.5%**, n*=3 | H2, **−**, **9%**, n*=4 | H3, **+**, **63%**, n*=1 |
| **G2** | **10%** | 79 ms | 667 ms | H0, **+**, **34.8%**, n*=2 | H2, **−**, **39.8%**, n*=2 | H3, **+**, **24.9%**, n*=2 |

### Neuron decyzyjny D (wejścia = wyjścia G)

| Płytka | LED (RV5) | τ_syn | τ_mem | **J1 (RV1)** | **J2 (RV2)** | **J3 (RV3)** |
|---|---|---|---|---|---|---|
| **D** | **26.4%** | 20 ms | 328 ms | G0, **+**, **28.6%**, n*=2 | G1, **−**, **100%**, n*=1 | G2, **−**, **21.4%**, n*=2 |

**Martwe synapsy (nie podłączaj, trymer na 0): `H0.J2` i `G0.J3`.**

> Ciekawostka do zrozumienia sieci: `hf_hi` wchodzi **hamująco** (`−`) do H1 i H3, a `hf_lo`
> **pobudzająco** (`+`) do H0/H2. Sieć sama zbudowała detektor „głośne ORAZ o widmie szkła".

## 13. TABELA POŁĄCZEŃ (każdy kabel)

Piny: **D2**=peak, **D3**=peak_cnt, **D4**=cv, **D5**=zcr, **D6**=flux, **D7**=hf_lo, **D8**=hf_hi.

### Arduino → H
| Od | Do | Wejście |
|---|---|---|
| **D2** (peak) | H0 | J1 |
| **D7** (hf_lo) | H0 | J3 |
| **D4** (cv) | H1 | J1 |
| **D5** (zcr) | H1 | J2 |
| **D8** (hf_hi) | H1 | J3 |
| **D2** (peak) | H2 | J1 |
| **D6** (flux) | H2 | J2 |
| **D7** (hf_lo) | H2 | J3 |
| **D3** (peak_cnt) | H3 | J1 |
| **D6** (flux) | H3 | J2 |
| **D8** (hf_hi) | H3 | J3 |

### H → G
| Od | Do | Wejście |
|---|---|---|
| H0 wyjście | G0 | J1 |
| H1 wyjście | G0 | J2 |
| H1 wyjście | G1 | J1 |
| H2 wyjście | G1 | J2 |
| H3 wyjście | G1 | J3 |
| H0 wyjście | G2 | J1 |
| H2 wyjście | G2 | J2 |
| H3 wyjście | G2 | J3 |

### G → D → alarm
| Od | Do | Wejście |
|---|---|---|
| G0 wyjście | D | J1 |
| G1 wyjście | D | J2 |
| G2 wyjście | D | J3 |
| **D wyjście** | Arduino **D9** (lub GPIO reaktora) | — |

Fan-out (użyj kolejnych terminali wyjściowych): H0→2, H1→2, H2→2, H3→2, D2→2, D6→2, D7→2, D8→2.

## 14. Reguła decyzji (dekoder)

**Zalecane: k = 1** — alarm gdy D strzeli choć raz.
- Za dużo fałszywych alarmów? → wymagaj **2 spików w oknie 2.5 s**, albo **obniż RV5 na D**
  o 1–2 działki (twardszy próg). **Nie ruszaj wag.**
- D strzela na szkle ~15×/klip, na tle ~3.6× — reguła k rozróżnia po gęstości serii.

---

# CZĘŚĆ VI — TEST, PROBLEMY, NAPRAWA

## 15. Test końcowy — i czego realnie oczekiwać

Puść z głośnika: ciszę, mowę, muzykę, **nagranie tłuczenia szkła**.

**Realne liczby z naszych testów (zbiór testowy, reguła k=1):**
- **~72% nagrań szkła** budzi system,
- **~20% klipów tła** daje fałszywy alarm (głośne zdarzenia jak strzały/dzwony: ~19–23%; cisza: ~46%*),
- to jest **zgrubna brama** — reaktor (LLM) ma odrzucać fałszywki. **Przeoczone szkło jest
  droższe niż fałszywy alarm**, dlatego celujemy w recall.

\* „cisza" w naszym zbiorze to nagrania tła (nie absolutna cisza) — w realnym cichym pokoju
będzie znacznie lepiej, bo nie ma transientów do reagowania.

## 16. TROUBLESHOOTING — pełna tabela

| Objaw | Przyczyna | Co zrobić |
|---|---|---|
| Pasek nie świeci wcale | brak zasilania | CR2032, biegun |
| **Kręcę trymer i nic** | **nie wysyłasz impulsów** | wyślij `C <pin> 20 100` — sam trymer nic nie robi |
| **Nic na żadnej płytce** | **brak wspólnej masy** | GND Arduino ↔ GND wszystkich płytek na jedną szynę |
| **Strzela cały czas, nie da się zatrzymać** | **V_leak (RV5) za wysoko** — nad progiem | zjedź RV5 w dół, aż 7. dioda przestanie sama migać. To normalne zachowanie, nie awaria |
| RV5 na maks, a świeci 1 dioda | zły trymer albo zły kierunek | przejedź RV5 przez CAŁY zakres (drabinka 11.4 krok 2) |
| Pasek się rusza, ale nie dochodzi do progu | waga za mała / V_leak za nisko / za mało impulsów | podkręć wagę, zwiększ `n`, tymczasowo podnieś RV5 |
| Synapsa nigdy nie odpala mimo maks wagi | **znak na `−`** | `−` hamuje (wylewa) — sprawdź przełącznik |
| Neuron odpala za wcześnie | waga za wysoka | przykręć trymer w dół |
| Enkoder sypie spikami w ciszy | wzmocnienie mikrofonu za duże | przykręć pokrętło na module |
| Enkoder milczy na hałas | wzmocnienie za małe / zły pin | podkręć; sprawdź OUT→A0 |
| `hf_lo`/`hf_hi` nigdy nie strzelają | za słaby mikrofon (szum kwantyzacji zjada wysokie pasmo) | potrzebny wzmacniacz (rozdz. 9) |
| Cała warstwa G/D martwa | brak masy / H nie działa | sprawdź masę; kalibruj od dołu (H najpierw) |
| D strzela ciągle | V_leak D za wysoko | obniż RV5 na D o 1–2 działki |
| Trymer kręci się luźno / bez reakcji | **urwany wiper** | wymień płytkę na zapasową (rozdz. 17) |
| Dziwne, losowe zachowanie po wgraniu firmware | timing ISR (rzadkie) | margines jest duży (~6 µs z ~52 µs) — zgłoś, da się zweryfikować analizatorem |

## 17. Uszkodzenia i części zamienne

**Urwałeś trymer?** Nie walcz. **Weź zapasową płytkę** i skalibruj ją od zera. Masz 15 płytek,
potrzeba 8 — zapas jest po to. Uszkodzoną odłóż na kupkę „do naprawy po demie".

**Naprawa (później, jak będzie czas):** odlutuj urwany trymer, wlutuj nowy:

| Element | Wartość |
|---|---|
| RV5 (V_leak) | **100 kΩ** |
| RV1/RV2/RV3 (wagi) | **22 kΩ** |
| RV6 (τ) | ~50 kΩ |
| RV4 (τ) | 22 kΩ |

Obudowa: **trymer 3/8″ kwadratowy, typ Bourns 3266**. Dokładny footprint: `neuron.kicad_sch`
w `lu.i-neuron-pcb/` albo BOM Lu.i — sprawdź przed zamówieniem.

---

# DODATKI

## Dodatek A — Reprodukcja (trening od zera; NIE potrzebne do budowy)

> **UWAGA (issue #34/#36):** `spikes_manifest7` jest NIEAKTUALNY z trzech
> niezależnych powodów: powstał ze zbioru z odwróconymi etykietami VOICe
> (1412 klipów szkła oznaczonych jako tło), zawiera 40 nagrań ESC-50 klasy
> 38 = `clock_tick` w klasie pozytywnej, i ma przeciek między splitami
> (194 z 194 miksów VOICe obecnych w teście są też w treningu). Wszystkie liczby
> w §15 i w `hw7_config.json` z niego pochodzą i są nieważne. Poniższa procedura
> buduje `spikes_v2` z zatwierdzonej wersji zbioru; wyniki z niego NIE są
> porównywalne z §15 i trzeba je przemierzyć od zera.

```bash
cd "$(git rev-parse --show-toplevel)"
# 1. zakoduj audio na spiki (~15-20 min)
.venv/bin/python architecture_14_neurons_patryk_09_07/encoder_twin.py build-manifest \
    --manifest dataset/versions/v2.0.0/manifest.csv --root . \
    --out architecture_14_neurons_patryk_09_07/spikes_v2 --warmup-seconds 30

# 2. trening (zwycięska konfiguracja: seed 2, pos_weight 1.0)
cd architecture_14_neurons_patryk_09_07
../.venv/bin/python snn_hw_pipeline.py train \
    --data spikes_v2/train --val-data spikes_v2/val \
    --test-data spikes_v2/test --epochs 100 --patience 15 \
    --hat-frac 0.5 --seed 2 --pos-weight 1.0 --out hw7_config.json --ckpt best7.pt

# 3. metryki klipowe
../.venv/bin/python eval_stream.py --ckpt best7.pt --data spikes_v2/test
```
⚠️ Zmiana czegokolwiek w `encoder_twin.py` wymaga przekodowania zbioru **i** retreningu —
CSV-ki są „upieczone" z konkretną wersją enkodera.

## Dodatek B — Co sprawdziliśmy i ODRZUCILIŚMY (nie wracaj do tego)

| Pomysł | Wynik | Dlaczego odrzucony |
|---|---|---|
| **15 płytek zamiast 8** (szersza H) | test F1 0.576–0.640 vs **0.609** baseline | Nie bije wyraźnie. Najlepszy seed kupił F1 **regresem recall szkła 72%→54%** — dyskwalifikuje. Wąskie gardło G=3 ogranicza korzyść. |
| **Ensemble D (3 kopie, głos 2-z-3)** | odporność min 0.798 vs **0.805** pojedynczy | Nie pomaga. Rozrzut siedzi w H/G, wspólnych dla kopii D; własny szum D znikomy. |
| **`crest` jako cecha** | waga wyszła 0% | Martwa — zastąpiona przez `hf_lo`/`hf_hi`. |
| **`hf_flux`** | szkło 5.8% vs cisza 7.1% | Nie różnicuje. |
| **hf_ratio z progiem z-score** | sygnał **odwrócony** | z-score to detektor ZMIANY, a udział widma to POZIOM. Stąd próg bezwzględny. |

## Dodatek C — Mapa plików

| Plik | Co to |
|---|---|
| **`PRZEWODNIK_KOMPLETNY.md`** | **ten dokument — zacznij tutaj** |
| `hw7_config.json` | **źródło prawdy** dla nastaw (8 płytek, 7 kanałów) |
| `best7.pt` | wytrenowany model (seed 2) |
| `encoder_v2.ino` | firmware Arduino (v3: filtr HF, progi bezwzgl., 7 kanałów) |
| `encoder_twin.py` | cyfrowy bliźniak enkodera + budowanie datasetu |
| `snn_hw_pipeline.py` | model, trening HAT/QAT, eksport nastaw |
| `eval_stream.py` | metryki klipowe + rozbicie fałszywych alarmów |
| `DECYZJE_SESJI.md` | dziennik wszystkich decyzji (pod publikację) |
| `NAUKA_projekt.md` | teoria: jak to działa |
| `architektura_i_kalibracja.md` | architektura + zaawansowane fazy kalibracji A–E |
| `kalibracja_sciaga_v3.md` | skrócona ściąga (zawarta w tym dokumencie) |
| ~~`kalibracja_sciaga.md`, `hw_config.json`~~ | **STARE** (6-kanałowe) — ignoruj |
