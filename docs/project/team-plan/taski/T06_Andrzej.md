# T06 Andrzej

Fizyczne Lu.i kalibracja oraz wiarygodne pomiary energii

Branch: feat/andrzej-hardware-energy

Reviewer: Patryk dla Lu.i; Kacper dla Uno; Marcel dla metodologii

Zakres plików: hardware/calibration/**; hardware/measurements/**; hardware/runbooks/**; schematy bez sekretów; duże przebiegi poza Git

Punkt startowy: lu.i-neuron-pcb-master/doc; schematy i instrukcje elektroniki; posiadane płytki i sprzęt pomiarowy

## A1 Inwentaryzacja i bezpieczne stanowisko

Kolejność: D1 rano. Zależności: brak.

1. Spisać liczbę/sprawność Lu.i, model Pi 5 i opcjonalnego Zero, kamerę, kable USB/OTG/CSI, zasilacze, buzzer i dostępny miernik/oscyloskop.

2. Potwierdzić pinout i zakres MAX4466→Uno ADC, masę i napięcia; Pi komunikuje się z Uno po USB. GPIO Pi nie przyjmuje 5 V.

3. Ustalić sterowanie buzzerem i wymagany driver na podstawie rzeczywistego modułu. Zrobić zdjęcie realnego okablowania i listę ustawień.

Odbiór: Wiktor może uruchomić stanowisko z instrukcji; braki są zapisane pierwszego dnia, zanim zacznie się integracja.

Przekazanie: Kacper zna tor ADC; Wiktor zna adapter kamery i wyjść; zespół wie, czy Zero jest opcją.

## A2 Kalibracja pojedynczych neuronów

Kolejność: D1–D2. Zależności: A1, P1.

1. Nadać board_id, zapisać nastawy trymerów i napięcie zasilania. Podać impulsy o kontrolowanej amplitudzie, czasie i szerokości.

2. Zmierzyć odpowiedź pobudzającą/hamującą, tau membrany/synaps, próg, reset, refractory, zakres nasycenia i LED. Zachować surowe przebiegi.

3. Przekazać osobny zestaw do dopasowania i walidacji Patryka. Z góry wspólnie ustalić tolerancje i sposób porównania czasów spike.

Odbiór: Każda wartość ma jednostkę, warunki i board_id. Jeśli nie da się zmierzyć Vmem, raport nie deklaruje zweryfikowanego przebiegu potencjału.

Przekazanie: Patryk dostaje CSV i opis; Karolina zakres fizycznej skali LED.

## A3 Pomiar energii urządzeń i sieci

Kolejność: D2–D3. Zależności: A1, K2.

1. Zdefiniować granice pomiaru: Uno+mikrofon, sieć Lu.i, Pi idle/capture/transmit/alarm. Zmierzyć pobór z LED i jawnie opisać ich udział.

2. Użyć napięcia/prądu w czasie; podać częstotliwość próbkowania miernika i niepewność. Powtórzyć ten sam przebieg kilka razy.

3. Dla krótkich zdarzeń uwzględnić ograniczenie miernika USB; nie wyliczać impulsowej energii ze zbyt wolnego pojedynczego odczytu.

4. Chmurę mierzyć jako koszt/zasoby albo oddzielną estymację; nie utożsamiać liczby spike z energią fizycznej płytki.

Odbiór: Raport rozdziela measured/estimated i podaje W oraz J/Wh przy tej samej ekspozycji. Brak pomiaru oznacza unavailable.

Przekazanie: Marcel otrzymuje tabelę do artykułu; Karolina dane Energy z granicą systemu.

## A4 Sieć fizyczna lub dane partnerskie

Kolejność: D3. Zależności: A2, M5.

1. Odwzorować kompatybilną topologię championa i nastawy; weryfikować fan-in, fan-out, znak i timing. Nie każda sieć 0–50 z edytora pasuje do fizycznego zestawu.

2. Odtworzyć te same wejściowe spike co w symulacji, zapisać rzeczywiste output spike/decisions i źródło zegara.

3. Jeżeli używane są wyniki drugiego zespołu: zebrać board revision, nastawy, firmware, encoder/model hash, surowe dane, aparat pomiarowy i zgodę na wykorzystanie/atrybucję.

Odbiór: Identyczny typ Lu.i nie zastępuje zgodności konfiguracji. Dane nieporównywalne trafiają wyłącznie do wyników wstępnych z jasnym opisem.

Przekazanie: Marcel otrzymuje porównywalny raport albo jawny brak finalnego eksperymentu.

## A5 Instrukcja i finalny odbiór sprzętu

Kolejność: D3. Zależności: A3.

1. Opisać uruchomienie, kontrolę poziomów, bezpieczne wyłączenie i symptomy błędnego okablowania.

2. Z Wiktorem sprawdzić fizyczny LED/buzzer: brak alarmu po starcie, limit czasu, Stop i awaria połączenia.

3. Przekazać podpisane identyfikatorem sesji zdjęcia stanowiska i pliki kalibracji przez własny PR.

Odbiór: Druga osoba uruchamia stanowisko bez domysłów, a alarm nie pozostaje włączony po utracie chmury.

Przekazanie: Zespół ma materiał do demonstracji i metodologii artykułu.