# T03 Karolina

Angielski dashboard z płytkami Lu.i i pomiarami

Branch: feat/karolina-dashboard

Reviewer: Patryk dla sygnałów; Wiktor dla integracji

Zakres plików: rpi_agents/cloud/app/templates/**; static/**; routes_dashboard.py; tests/dashboard/**; UI fixtures

Punkt startowy: feat/dashboard: templates/static/routes_dashboard; referencja ciemna v2 i angielska v3; neuron.svg

## C1 Szkielet i ekran logowania

Kolejność: D1. Zależności: W0.

1. Przygotować ciemny shell: Network, Events, Experiments, Energy, Device. Wszystkie etykiety, błędy i tooltipy po angielsku.

2. Zbudować Username, Password, Sign in oraz Logout. Formularz korzysta z API Wiktora, nie przechowuje hasła w JS/localStorage i nie zawiera prawdziwych danych w mockach.

3. Wczytać fixture kontraktów i oznaczyć demo data. Oddzielić komponenty danych od renderowania, żeby realny backend zastąpił fixture bez przepisywania UI.

Odbiór: Widoki są dostępne bez sprzętu jako jawny tryb demo; formularz ma focus klawiatury, loading i Invalid credentials.

Przekazanie: Wiktor dostaje listę route/asset i kontrakt formularza; Patryk widzi format danych wymagany przez komponenty.

## C2 Płytki i edytor sieci

Kolejność: D1–D2. Zależności: C1.

1. Użyć neuron.svg jako źródła kształtu, osobne warstwy portów, LED i zaznaczenia. Nie rysować płytki jako zwykłego koła.

2. Dodać Board count 0–50, automatyczny layout, przeciąganie, zoom/pan/Fit, wybór neuronu i widok połączeń pobudzających/hamujących.

3. Rozdzielić edycję draft, załadowany model i uruchomioną sesję. Import/Export JSON nie może wykonywać kodu ani nadpisywać aktywnego championa.

Odbiór: Sprawdzone 0, 1, 8 i 50 płytek; puste płótno jest czytelne, etykiety nie nachodzą, połączenia zachowują ID po zmianie layoutu.

Przekazanie: Patryk dostaje payload draft topologii; zespół odnosi wygląd do referencji v3.

## C3 Potencjał LED raster i replay

Kolejność: D2. Zależności: C2, P3.

1. Implementować LED potencjału i osobny błysk spike na podstawie pól runtime, nie losowej animacji lub samego tau.

2. Dodać inspektor neuronu: Signals, Parameters, Connections, Notes; wykres Vmem z progiem, jednostki i calibration status.

3. Obsłużyć Live/Replay, Pause view/Resume, stale data oraz reconnect. Skok replay odtwarza spójny stan, nie tylko przewija ekran.

Odbiór: Z golden fixture LED i raster wskazują ten sam neuron/czas; utrata połączenia daje Stale data; Pause view nie wysyła Stop.

Przekazanie: Patryk zatwierdza mapowanie sygnałów; Wiktor sprawdza obciążenie i reconnect.

## C4 Zdarzenia metryki i energia

Kolejność: D2–D3. Zależności: C1, W2.

1. Event details: zdjęcie, SNN/capture/vision/alarm timeline, osobne glass/person/authorization, ACK i error state. Brak danych pokazuje Not available.

2. Experiments: split, godziny tła, seed, model_hash, encoder_hash, FA/h z przedziałem ufności i recall; oddzielić metryki SNN oraz całego systemu.

3. Energy: źródło measured/estimated, granica pomiaru, moc W i energia J/Wh; Device: heartbeat, bufor, gap, kamera, serial. Nie wpisywać przykładowych liczb w realny tryb.

Odbiór: Dane przykładowe są oznaczone; brak pomiaru nie staje się zerem. Filtr wyniku nie miesza różnych modeli, datasetów ani granic energii.

Przekazanie: Marcel i Andrzej sprawdzają znaczenie swoich metryk; Wiktor podłącza API.

## C5 Odbiór UI na realnych danych

Kolejność: D3. Zależności: C3, C4, W4.

1. Przejść login/logout, jedną sesję live i replay, błąd vision, utratę łącza, puste dane i próbę zmiany aktywnego modelu.

2. Przetestować desktop oraz węższy ekran, nawigację klawiaturą i 50 płytek. Renderowanie musi mieć ograniczony bufor i liczbę punktów wykresu.

3. Zrobić zrzuty referencyjne z jawnego demo i z jednej oznaczonej realnej sesji. Przekazać PR ze zdjęciami before/after i listą testów.

Odbiór: Brak polskich etykiet produktu, wiszących spinnerów i wymyślonych danych; serwer blokuje dane po logout niezależnie od ukrycia UI.

Przekazanie: Wiktor otrzymuje gotowe zasoby do tego samego kontenera; dokumentacja ma wersję UI i SHA.