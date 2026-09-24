# Aktualne założenia projektu i badania

Stan planu: 24 września 2026. Ten dokument ustala priorytet wykonania i raportowania. Opisuje decyzje projektowe, nie deklaruje gotowych wyników ani sprawności sprzętu.

## Pytanie badawcze i zakres obowiązkowy

Badamy, jak liczba symulowanych neuronów Lu.i wpływa na jakość ciągłej detekcji stłuczenia szkła, opóźnienie i koszt obliczeń, oraz jak detektor SNN wypada wobec detektora opartego na FFT uruchomionego na Arduino Uno. Nie zakładamy z góry przewagi SNN. Hipotezy o większej odporności na zakłócenia, lepszym recall, mniejszym FA/h, oszczędności energii i zgodności symulatora z elektroniką wymagają osobnych danych.

Minimalny wynik badawczy obejmuje: wersjonowany zbiór ciągły i jego generator; zamrożony champion SNN; działający baseline FFT na Uno; wspólny zbiór testowy i protokół zdarzeniowy; raport jakości, opóźnień oraz jawnie oznaczonych pomiarów i estymacji kosztu. Przy kilku liczbach neuronów pokazujemy krzywą kompromisu jakości i zasobów, a nie wybieramy „optymalnej” liczby z wczesnych prób. Architektura i dashboard tworzą demonstrator, lecz wyniki z obrazu, e-maila i alarmu nie wchodzą do bezpośredniego porównania dwóch detektorów dźwięku.

## Demonstrator i rozszerzenia

Demonstrator E2E: MAX4466 → wejście analogowe Uno → zakodowane ramki przez USB → stale działające Raspberry Pi 5 → symulacja SNN w Azure → trigger → zdjęcie i analiza w Azure AI Foundry → decyzja → LED/buzzer i e-mail zgodnie z regułami systemu. Agent obrazu działa dopiero po sygnale SNN. Dashboard w języku angielskim pokazuje 0–50 wizualizacji płytek Lu.i i telemetrię; widok nie jest samodzielnym pomiarem hardware. Wersja Pi Zero zależy od rozpoznania modelu, kamery, kabli i testu zgodności. Identyfikacja domowników pozostaje późniejszą fazą.

Pomiary na rzeczywistych płytkach Lu.i są rozszerzeniem, jeśli dostępna jest wystarczająca liczba sprawnych egzemplarzy i da się powtórzyć kalibrację oraz test na tym samym strumieniu. Inwentaryzacja, uszkodzenia i liczba działających neuronów wymagają potwierdzenia na stanowisku. Dane innego zespołu wolno wykorzystać jako zewnętrzne pomiary komponentów przy jawnie opisanej konfiguracji; nie wolno ich przedstawiać jako własnego pomiaru całego detektora ani dowodu zgodności 1:1. Jeśli fizyczny wariant nie powstanie, artykuł opisuje go jako przyszłą walidację, bez tabeli pozornych wyników.

## Protokół porównania

1. Zamrażamy wersję audio, grupy źródłowe, generator strumieni, seedy, podział train/validation/test, etykiety i manifest enkodera. Weryfikujemy, czy źródła testowe nie przeciekają do treningu ani strojenia. Syntetyczne złożenie strumienia musi pozostać rozpoznawalne jako taki typ testu, nie jako wielogodzinna rejestracja terenowa.
2. Oba detektory oceniamy na tych samych wejściach akustycznych, pozytywnych zdarzeniach i tle. Różnice mikrofonu, ADC, odtwarzania przez głośnik lub wejścia liniowego dokumentujemy; jeśli nie da się ich zrównać, osobno raportujemy efekt toru wejściowego. Baseline FFT musi mieć jawne okno, krok, pasma, próg i cooldown.
3. Parametry, próg decyzji i regułę cooldown ustalamy wyłącznie na walidacji, a test uruchamiamy raz po zamrożeniu konfiguracji. Dopasowanie triggerów do zdarzeń jest jeden do jednego. Raportujemy recall zdarzeniowy, FA/h czasu negatywnego, opóźnienie od początku zdarzenia do decyzji (p50/p95), duplikaty i liczbę aktywacji po cooldown. Dla FA/h mianownikiem jest ekspozycja negatywna po odjęciu przedziałów zdarzeń i tolerancji, nie całkowity czas testu.
4. Porównanie **detektorów** kończy się na decyzji „glass / no glass”. Osobno raportujemy wdrożony tor SNN (Uno, Pi, łącze, Azure) i tor FFT (Uno oraz wymagany hub, jeśli obecny). W torze wdrożonym mierzymy również opóźnienie sieci, cold start, błędy łączności i moc urządzeń. Wizja, kamera i e-mail mogą być mierzone jako koszt późniejszej reakcji, ale nie służą do przyznawania punktów tylko jednej stronie porównania detektorów.
5. Energia fizyczna wymaga podania punktu pomiaru, czasu, stanu spoczynku i aktywnego oraz powtórzeń. Koszt Azure i czas CPU nie są bezpośrednim pomiarem energii serwera. Estymacje skalowania Lu.i, np. `P_N ≈ P_base + N × P_increment`, muszą mieć podane założenia i niepewność. Oddzielamy koszt zakupu prototypu, hipotetyczny koszt produkcji seryjnej i bieżący rachunek chmurowy.

## Stan danych i twierdzeń

W repozytorium są wersjonowane dane, kod pipeline i eksperymenty na branchach. Integracja, końcowy champion i niezależny wynik continuous wymagają weryfikacji konkretnego commita oraz manifestu. Wczesne liczby recall, obserwacje o optymalnej liczbie neuronów, kosztach i poborze prądu nie są wynikami końcowymi. Trening projektujemy dla MacBooka M5 Max 128 GB, a rzeczywistą przepustowość i zużycie pamięci ustalamy benchmarkiem. Nie dopasowujemy tezy do pożądanego wyniku: jeśli FFT wygra, raportujemy to wprost i analizujemy ograniczenia oraz warunki porównania.

Priorytet pracy zespołu: Marcel i Kacper przygotowują odtwarzalny continuous dataset i manifest; Karolina rozwija dashboard; Patryk kończy symulator i eksport championa; Wiktor realizuje tor Pi/Azure i integrację; Andrzej weryfikuje Uno, stanowisko oraz opcjonalne płytki Lu.i. Wspólnymi bramkami są zgodny kontrakt impulsów, zamrożony test, baseline FFT oraz pomiary z opisanymi granicami. Rozwinięcie zadań pozostaje w [planie zespołu](team-plan/Plan_zespolu_SNN.md).
