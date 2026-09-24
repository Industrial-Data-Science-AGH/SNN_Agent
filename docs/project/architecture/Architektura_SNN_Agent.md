# Architektura systemu SNN Agent
## Detekcja stłuczenia szkła i cyfrowa reprezentacja sieci Lu.i
Wersja 1.1 • 24 września 2026 • dokument dla zespołu koła naukowego

Aktualny priorytet i granice eksperymentu: [CURRENT_ASSUMPTIONS.md](../CURRENT_ASSUMPTIONS.md). Ten dokument zachowuje pełną architekturę demonstratora; fizyczna sieć Lu.i jest rozszerzeniem badania, zależnym od walidacji płytek i czasu.

Budujemy kompletny demonstrator na Arduino Uno i Raspberry Pi 5. Arduino odczytuje analogowy mikrofon MAX4466 i koduje sygnał do impulsów. Raspberry Pi przesyła je do Azure, wykonuje zdjęcia na żądanie i steruje alarmem. Symulacja SNN, analiza obrazu w Foundry, reguły decyzji, historia i dashboard działają w chmurze.

Podział odpowiedzialności został uzgodniony z Wiktorem. Ewentualne przeniesienie na Pi Zero nastąpi po sprawdzeniu wariantu płytki, kamery i kabli. Celem jest zachowanie tego samego serwisu i interfejsu sieciowego. Ograniczeniem pozostaje zgodność systemu i adapterów sprzętowych.

Dokument określa docelowe zachowanie, kontrakty integracyjne i sposób sprawdzenia wyników. Istniejący kod jest punktem wyjścia, a nie dowodem ukończenia tej architektury. Szczegółowe wartości konfiguracyjne oznaczone jako propozycja wymagają testu na stanowisku przed zamrożeniem eksperymentu.

@fig assets/01-kontekst.png | System w otoczeniu urządzenia, chmury i operatora | 3.8

Wzorzec wyglądu dashboardu: zatwierdzona ciemna referencja UI v2; interfejs produktu w całości po angielsku. Dokument i objaśnienia techniczne pozostają po polsku. Diagramy przedstawiają architekturę, a obrazy generowane pokazują zamierzony wygląd i nie stanowią wyników pomiarów.
===
# 1 Zakres i sposób czytania dokumentu
## Rezultat pierwszego wdrożenia
Pierwsza wersja ma przeprowadzić cały łańcuch: dźwięk, impulsy, decyzja SNN, zdjęcie, analiza obrazu, reguła alarmu, LED i buzzer oraz e-mail. Operator widzi przebieg w dashboardzie i potrafi powiązać każdą reakcję z konkretnym zdarzeniem i wersją modelu. Raspberry Pi pozostaje włączone; „wybudzenie” oznacza uruchomienie ścieżki zdjęcia i analizy, nie start systemu po halt.

Drugim rezultatem jest stanowisko badawcze. Podstawowy protokół porównuje detektor symulowanej sieci SNN z detektorem FFT na Arduino dla tego samego strumienia audio i na tej samej granicy decyzyjnej. Ocena fizycznej sieci Lu.i jest rozszerzeniem po potwierdzeniu sprawności i kalibracji płytek. Demonstracja przepływu impulsów, agent wizualny i dashboard są oddzielone od miar jakości detektora akustycznego. Animacja nie jest dowodem zgodności sprzętowej ani efektywności energetycznej.

## Nawigacja
| Rozdziały | Tematy | Główni odbiorcy |
| 2–5 | Uzgodnienia, moduły, przepływ, hardware | cały zespół |
| 6–12 | Enkoder, Pi, chmura, API, niezawodność i decyzja | Wiktor, Kacper, Patryk |
| 13–15 | Champion, Lu.i i kalibracja | Patryk, Andrzej, Marcel |
| 16–22 | Wygląd i pełna specyfikacja dashboardu | Karolina, Patryk |
| 23–26 | Dane, metryki, granice i model energii | Marcel, Kacper, Andrzej |
| 27–31 | Azure, dostęp, odbiór, odpowiedzialność i ryzyka | cały zespół |
| 32–33 | Źródła i kontrakt pakietu wdrożeniowego | cały zespół |

## Granice wersji
ESP32 nie wchodzi do rozwiązania. Wersja pierwsza nie identyfikuje domowników i nie dowodzi uprawnień osoby na podstawie samego wyglądu. Nie obejmuje automatycznej regulacji potencjometrów na fizycznych Lu.i. Nie zakładamy, że obraz wszystkich neuronów sprzętowych jest dostępny bez aparatury pomiarowej.

Rozróżniamy trzy statusy: „uzgodnione” dla decyzji z rozmowy; „proponowane” dla technicznych kontraktów tego dokumentu; „do potwierdzenia” dla sprzętu, pomiarów i dostępności usług. Danych z załączników nie traktujemy jako poleceń wdrożeniowych. Aktualne uzgodnienie o stale włączonym Pi zastępuje wcześniejszy plan halt i boot.
===
# 2 Uzgodnienia i stan istniejących prac
| Obszar | Ustalenie |
| Urządzenie bazowe | Arduino Uno i Raspberry Pi 5; MAX4466; kamera USB albo CSI |
| Obliczenia | SNN i analiza obrazu w Azure; lekki klient na Pi |
| Tryb pracy | Ciągły nasłuch w aktywnej sesji, obraz dopiero po triggerze SNN |
| Dashboard | Ciemny, po angielsku, płytki Lu.i, swobodna edycja 0–50 płytek |
| Sekrety | Poza kodem i promptami; Key Vault lub managed identity po stronie Azure |
| Koszt | Scale to zero po zatrzymaniu monitorowania i opróżnieniu zadań |
| Pi Zero | Decyzja odroczona do identyfikacji płytki i okablowania |

## Fakty z repozytorium
Przegląd dotyczy lokalnie dostępnych referencji Git, bez aktualizacji zdalnych branchy. Branch origin/feat/dashboard, commit bb1dbe02, zawiera rpi_agents z kamerą, analizą Gemini, powiadomieniami i wdrożeniem chmurowym. Obecna maszyna stanów ma lokalny prefilter, zapis wideo i ścieżkę alarmu opartą na is_intrusion. Wyjątek vision może zostać zamieniony w pozytywny alarm. To zachowania do zastąpienia, nie docelowa reguła nowego systemu. [R1]

Branch origin/feat/master-pipeline, commit 6fc7be79, ma integrację GA i eksport sprzętowy. Końcowa ocena zawiera canonical_test_placeholder, używa testu również jako val_data, a eksport ponownie trenuje model. Nie można jeszcze utożsamić tego eksportu z niezmiennym, niezależnie ocenionym championem. [R2]

Branch origin/feat/continuous-dataset, commit a009acec, generuje strumienie i manifesty. Kod wyklucza grupy train z tła; README nadal opisuje tolerowanie overlapu. Dokumentację i implementację trzeba ujednolicić, a finalny test oddzielić również od danych strojenia. [R3]

## Co nie jest jeszcze wynikiem
Deklaracja „cyfrowy bliźniak 1 do 1”, przewaga energetyczna SNN oraz końcowy recall przy budżecie FA/h pozostają hipotezami do sprawdzenia. Starsze wyniki okienkowe i wyniki sprzed naprawy etykiet nie mogą być przedstawiane jako finalna ewaluacja. Uruchomiony trening nie oznacza automatycznie gotowego artefaktu produkcyjnego.
===
# 3 Moduły i własność stanu
@fig assets/02-moduly.png | Podział logiczny modułów urządzenia i chmury | 5.0

Serwis edge jest jedynym procesem aplikacyjnym wymaganym na Raspberry Pi. Adapter serial zamienia ramki Uno na wspólny kontrakt impulsów. Klient komunikacyjny utrzymuje kolejność danych, a adaptery kamery i GPIO realizują mały, jawny zestaw poleceń. Żaden moduł na Pi nie wybiera modelu SNN ani nie interpretuje obrazu.

API w Azure uwierzytelnia klienta, sprawdza wersje protokołu i prowadzi sesję symulacji. Silnik SNN posiada stan neuronów i dekodera pomiędzy porcjami danych. Stan wizualizacji jest pochodną telemetrii silnika, a nie niezależną symulacją w przeglądarce.

Worker przetwarza trwałe zadania zdjęć i powiadomień. Reguła alarmowania interpretuje zwalidowany wynik modelu. Baza zdarzeń przechowuje decyzje i status wykonania, a magazyn obiektów zdjęcia, konfiguracje oraz przebiegi. Oddzielny proces worker zapobiega blokowaniu akwizycji przez długie zapytanie vision.

Źródłem prawdy o eksperymencie jest jego niezmienny manifest. Źródłem prawdy o wykonaniu alarmu jest potwierdzenie z Pi; samo utworzenie komendy w Azure oznacza tylko żądanie. W przypadku poczty odróżniamy przyjęcie przez dostawcę od dostarczenia odbiorcy, o ile dostawca udostępnia taki status.
===
# 4 Przebieg end to end
@fig assets/12-sekwencja.png | Kolejność komunikatów dla pojedynczego zdarzenia | 4.7

1. Operator rozpoczyna sesję. API zatwierdza zgodność urządzenia, enkodera i modelu. Kamera oraz wyjścia przechodzą kontrolę dostępności; alarm pozostaje wyłączony.
2. Uno wysyła zdarzenia impulsowe z czasem źródłowym. Pi zbiera je w krótkie porcje i wysyła do API. Również porcja bez impulsów niesie informację o upływie czasu.
3. SNN przetwarza dane kolejno. Dekoder wyjścia, próg i cooldown pochodzą z wersji modelu. Po spełnieniu warunku powstaje event_id oraz komenda capture z terminem ważności.
4. Pi pobiera komendę, wykonuje JPEG i wysyła go z event_id. Potwierdza wykonanie dopiero po rzeczywistym wykonaniu czynności. Zdjęcie musi mieć czas przechwycenia, a nie tylko czas wysyłki.
5. Worker analizuje obraz w Foundry. Wynik przechodzi walidację struktury; kod reguły tworzy decyzję i ewentualne żądania alarmu oraz e-maila.
6. Pi wykonuje ważną komendę alarmu z ograniczeniem czasu. API pokazuje osobno status lokalnego alarmu i powiadomienia. Nasłuch audio trwa podczas analizy obrazu.

Przeglądarka jest obserwatorem i panelem sterowania. Jej rozłączenie nie może przerwać aktywnej sesji. Odtwarzanie historycznego eksperymentu domyślnie nigdy nie uruchamia fizycznego alarmu ani poczty.
===
# 5 Stanowisko i połączenia sprzętowe
@fig assets/03-hardware.png | Schemat funkcjonalny połączeń wymagających weryfikacji przez Andrzeja | 4.7

MAX4466 jest mikrofonem z analogowym wzmacniaczem. Przetwarzanie cyfrowe wymaga ADC; w przyjętym wariancie używamy przetwornika w Uno. Raspberry Pi nie odczytuje tego sygnału analogowego przez GPIO. PWM nie zastępuje wejścia ADC. [S3, S4]

Połączenie Uno z Pi odbywa się przez USB, bez bezpośredniego łączenia linii 5 V Uno z GPIO 3,3 V Pi. Numer wejścia analogowego, poziom odniesienia ADC, napięcie zasilania mikrofonu, gain, masa oraz zakres napięcia OUT muszą być sprawdzone przed uruchomieniem. A0 jest propozycją zgodną z jednym z istniejących szkiców, a nie zatwierdzonym pinoutem wszystkich wersji.

Kamera USB jest wariantem bazowym, jeśli posiadany model działa jako obsługiwane urządzenie wideo. CSI jest alternatywą po identyfikacji sensora i taśmy. Nie kupujemy urządzeń na podstawie tego dokumentu. LED potrzebuje rezystora; stopień sterujący buzzerem zależy od jego napięcia i poboru prądu. Nie zakładamy zasilania buzzera bezpośrednio z GPIO.

Andrzej zapisuje faktyczny model Uno, napięcia i połączenia. Jeżeli to UNO R3, ograniczeniem jest ATmega328P 16 MHz i 2 kB SRAM. Raport obliczeń dla Cortex-M4F nie opisuje takiego układu. Wiktor zapisuje model Pi, system, kamerę i rozpoznane interfejsy. [S3, R4]
===
# 6 Enkoder i protokół z Arduino
## Zgodność treningu z urządzeniem
Champion musi wskazywać zestaw cech, ich kolejność, częstotliwość próbkowania, rozmiar i krok okna, filtrację, normalizację, metodę kodowania oraz szerokość impulsu. Nie wolno podłączyć trzykanałowego firmware do modelu wymagającego siedmiu innych kanałów i uznać, że interfejs jest zgodny. Nazwy i hash konfiguracji enkodera są sprawdzane przed Start.

Normalizacja w trybie live musi być przyczynowa: stałe wyznaczone na train albo jawny stan adaptacji oparty wyłącznie na przeszłości. Normalizacja klipu jego przyszłym maksimum nie jest równoważna urządzeniu nasłuchującemu stale. Przy zmianie gain mikrofonu lub częstotliwości próbkowania trzeba powtórzyć test zgodności cech.

## Proponowana ramka binarna USB
| Pole | Znaczenie |
| magic, version, type | Synchronizacja ramki, wersja i typ: hello, spikes, status |
| sequence, payload_length | Wykrywanie braków i kontrola długości przed alokacją |
| boot_id lub handshake sesji | Oddzielenie restartu Uno od zawinięcia licznika |
| source_time_us | Monotoniczny czas źródłowy; mapowanie i rozszerzenie licznika na Pi |
| channel_id, flags | Kanał impulsu i bity jakości; szerokość pochodzi z manifestu |
| CRC | Wykrywanie uszkodzonych ramek; nie służy do uwierzytelnienia |

Punktem startowym jest 115200 baud, 8N1 i wysyłanie ramek co około 20 ms. To propozycja do testu. Zdarzenie 6 B przy 1400 impulsach/s wymaga 8400 B/s; narzut ramek zwiększa ten wynik. Łącze 115200 8N1 przenosi maksymalnie około 11520 B/s, więc nie wolno równolegle drukować obszernych logów tekstowych. Próbki audio nie są standardowo przesyłane tym kanałem.

Firmware zgłasza przepełnienie bufora i utracone zdarzenia. Serializacja nie może blokować ISR próbkowania. Test obejmuje najszybszy spodziewany strumień, restart podczas odbioru, błędną długość, CRC i zawinięcie czasu. Szerokości impulsów w istniejących wersjach różnią się; przykład 500 µs w jednym szkicu nie jest zgodny z referencyjnymi około 15 ms Lu.i bez dodatkowej kalibracji. [R4, S1]
===
# 7 Lekki serwis Raspberry Pi i migracja
@fig assets/11-przenosnosc.png | Migracja urządzenia zachowuje logikę i kontrakt chmurowy | 3.7

Proponowany runtime to Python w środowisku wirtualnym i usługa systemd. Serwis ma oddzielne kolejki akwizycji, transportu i zadań kamery. Zablokowane zdjęcie lub HTTP nie może zatrzymać odbioru serial. Minimalne zależności obejmują bibliotekę portu szeregowego, klienta HTTPS i adapter GPIO. Zdjęcia pozyskujemy narzędziem systemowym lub lekkim adapterem, bez wczytywania klatek do dużych tablic NumPy.

Konfiguracja zawiera device_id, adres API, ścieżkę poświadczenia, stabilny identyfikator portu USB, typ kamery, rozdzielczość, limit JPEG, numery BCM GPIO, polaryzację wyjść i maksymalny czas alarmu. Preferujemy /dev/serial/by-id, jeśli jest dostępne; /dev/ttyACM0 może zmienić się po podłączeniu innych urządzeń.

Proponowane budżety do walidacji: proces serwisu poniżej 100 MB RSS poza chwilowym procesem kamery; bufor impulsów maksymalnie 60 s i 8 MiB; zdjęcia oczekujące maksymalnie 20 MiB; brak swapowania i utraty próbek w próbie godzinnej. To kryteria projektowe, nie zmierzone możliwości każdej wersji Zero. Bufory mają limity czasu i bajtów jednocześnie.

Na Zero instalujemy właściwy system i zależności od nowa. Nie kopiujemy venv ani obrazu systemu Pi 5. Starsze Zero/W wymaga sprawdzenia pakietów ARMv6 i 32-bit; Zero 2 W jest inną platformą. Dwa urządzenia USB mogą wymagać huba i OTG, a CSI właściwej taśmy. Migrację zatwierdzamy dopiero po tym samym teście kamery, serial, reconnect i alarmu, który przeszedł Pi 5. [S5]
===
# 8 Sesja symulacji i czas
@fig assets/04-sesja.png | Stan sesji jest niezależny od stanu pojedynczego alarmu | 3.9

Sesja identyfikuje urządzenie, tryb, model, enkoder i epokę stanu. Jedna sesja live ma jednego właściciela zapisu oraz kolejno przetwarzane porcje. Minimalna implementacja nie skaluje jednego SNN poziomo. Blokada lub lease sesji i token epoki chronią przed równoległą obsługą podczas restartu lub zmiany rewizji kontenera; samo maxReplicas=1 nie jest gwarancją braku chwilowego nakładania instancji. [S6]

SNN zachowuje Vmem, stany synaps, refractory oraz stan dekodera między porcjami. HTTP nie wyznacza kroku symulacji. Puste porcje przesuwają czas i pozwalają wygasać potencjałom. Brak porcji oznacza brak obserwacji, nie ciszę. Wersja integratora i dt należą do manifestu modelu.

Propozycja transportu: batch co 250 ms, krótkie potwierdzenie oraz dołączone komendy, gdy są dostępne. Pi odpytuje komendy w aktywnej sesji najrzadziej co 1 s, jeśli brak danych do wysłania. Dopasowanie interwałów do opóźnienia i kosztu jest pomiarem, nie zmianą dt neuronu.

Czas akustyczny opiera się na liczniku urządzenia. UTC służy do korelacji logów, a monotoniczny zegar Pi do lokalnych czasów trwania. Latencję obliczeń chmury mierzymy lokalnie w procesie chmurowym. Dokładnego opóźnienia w jedną stronę między niezestrojonymi zegarami nie raportujemy jako pomiaru bez informacji o błędzie synchronizacji.
===
# 9 Interfejs HTTP i telemetria
Wszystkie ścieżki poniżej są propozycją API v1, a nie deklaracją istniejących endpointów. Urządzenie inicjuje połączenia wychodzące HTTPS; nie otwieramy portu kamery ani serwisu Pi na internet. Komendy wracają w odpowiedzi API lub przez pobieranie oczekujących komend.

| Metoda i ścieżka | Zastosowanie | Wynik |
| POST /v1/sessions | Start po preflight i sprawdzeniu zgodności | session_id, epoch, limity |
| POST /v1/sessions/{id}/batches | Impulsy i zakres czasu porcji | ACK, stan przetwarzania, komendy |
| GET /v1/devices/{id}/commands | Pobranie poleceń należących do urządzenia | lista ważnych poleceń |
| POST /v1/commands/{id}/ack | Wynik wykonania i czas lokalny | potwierdzenie zapisu |
| POST /v1/events/{id}/image | JPEG z metadanymi i hashem | image_id, status zadania |
| POST /v1/sessions/{id}/stop | Zatrzymanie akwizycji i opróżnienie | DRAINING, potem STOPPED |
| GET /v1/sessions/{id}/telemetry | SSE dla przeglądarki | snapshot, delty, luki |
| GET /v1/events oraz /{id} | Historia i szczegół zdarzenia | stronicowane dane |
| GET /v1/experiments/{id} | Manifest, metryki, pochodzenie | niezmienny raport |

Każda mutacja ma request_id i klucz idempotencji związany z urządzeniem lub zdarzeniem. Duplikat tej samej treści zwraca ten sam rezultat; ten sam klucz z inną treścią oznacza konflikt 409. Błąd 401/403 nie uruchamia bezterminowych retry. 429 respektuje Retry-After; 5xx ma ograniczony backoff z jitterem. Przekroczenie limitu to 413, błąd schematu 422, niezgodna wersja kontraktu 409.

Proponowane limity: batch do 64 KiB, JPEG do 1 MiB, maksymalnie trzy zdjęcia na zdarzenie. Nazwa typu MIME nie wystarcza do uznania pliku za poprawny obraz. Backend sprawdza format i wymiary. Metadane odpowiedzi nie zawierają sekretów ani podpisanych adresów w logach.

SSE przesyła stan wybranej sieci około 10 razy/s, a nie każdą iterację symulatora. Przeglądarka renderuje niezależnie. Po reconnect pobiera pełny snapshot i ostatni numer telemetrii; utraconych próbek Vmem nie interpoluje jako zmierzonych danych.
===
# 10 Kontrakty danych i identyfikatory
## Przykład porcji impulsów
Poniższe wartości są ilustracją formatu. source_start_us i source_end_us określają półotwarty przedział [start, end). Impulsy są uporządkowane według dt_us w tym przedziale. Porty i kanały muszą należeć do manifestu enkodera.
```json
{
  "schema_version": "1.0",
  "device_id": "lab-pi-01",
  "session_id": "session-example",
  "epoch": 1,
  "batch_seq": 42,
  "encoder_hash": "sha256:<digest>",
  "source_start_us": 10000000,
  "source_end_us": 10250000,
  "spikes": [{"dt_us": 1200, "channel": "zcr"}],
  "quality": {"dropped_events": 0, "adc_clipped": false}
}
```
## Przykład komendy urządzenia
```json
{
  "schema_version": "1.0",
  "command_id": "command-example",
  "device_id": "lab-pi-01",
  "session_id": "session-example",
  "epoch": 1,
  "event_id": "event-example",
  "type": "capture",
  "issued_at": "2026-09-23T12:00:00Z",
  "expires_at": "2026-09-23T12:00:10Z",
  "parameters": {"frames": 1, "max_bytes": 1048576}
}
```
ACK zawiera command_id, status accepted/completed/failed/expired, czas wykonania, kod błędu oraz opcjonalny image_id. accepted nie oznacza completed. Komenda alarm wymaga duration_ms ograniczonego lokalnie na Pi; chmura nie może wymusić nieograniczonego czasu działania.

Identyfikatory session_id, event_id i command_id są stabilne przy retry. run_id dotyczy eksperymentu, model_hash artefaktu, a calibration_id profilu konkretnej płytki. Domyślna szerokość impulsu i jednostki wag pochodzą z konfiguracji; nie są zgadywane z liczby spike’ów.
===
# 11 Niezawodność i odzyskiwanie
| Sytuacja | Wymagane zachowanie |
| Ponowiona porcja | W tej samej epoce przetwarzana raz; sprawdzenie sekwencji i hasha |
| Luka lub przepełnienie | Jawne gap z przedziałem czasu; brak udawanej ciszy |
| Restart Uno | Nowy boot_id; nowa epoka i ponowny warmup |
| Restart backendu | Utrata stanu live oznacza przerwę; odtworzenie tylko ze spójnego zapisu |
| Opóźniona komenda | Odrzucona po TTL albo zmianie epoki |
| Brak kamery | CAPTURE_FAILED; nasłuch trwa, błąd widoczny operatorowi |
| Timeout Foundry | VISION_ERROR lub RETRY_PENDING; brak automatycznego uznania intruza |
| Błąd e-maila | Oddzielny status powiadomienia; nie cofa wykonanego alarmu |
| Brak internetu | DEGRADED, ograniczony bufor i wyłączona deklaracja ochrony online |

Propozycja pierwszej wersji: nie obiecujemy ciągłości detekcji podczas restartu chmury. Po utracie stanu tworzymy nową epokę, zapisujemy lukę i wykonujemy warmup. Porcje historyczne można zachować do replay; nie uruchamiają zdjęć po czasie. Jeśli później wdrożymy checkpoint i pełny dziennik wejść, test odtwarzania musi wykazać identyczny stan dekodera i brak powtórzenia efektów ubocznych.

Odpowiedź API rozróżnia received_seq, processed_seq i durable_seq. Potwierdzenie w RAM nie jest obietnicą trwałego archiwum. Proponowany zapis telemetryczny partiami co 5 s może pozostawić ostatni fragment niezapisany przy awarii; eksperyment ma wtedy oznaczenie incomplete. Dane wymagane do nieodwracalnej akcji, w tym decyzja i komenda, zapisujemy trwale przed jej wydaniem.

Queue Storage dostarcza zadania co najmniej raz. Worker ma deduplikację event_id i image_hash, odnawia czas niewidoczności długiego zadania i przenosi wyczerpane retry do kolejki błędów. Brak wspólnej transakcji Table i Queue rozwiązujemy rekordem outbox oraz okresowym dosyłaniem nieopublikowanych zadań. E-mail po niejednoznacznym timeout może mieć status delivery_unknown; bez wsparcia dostawcy nie gwarantujemy exactly once.
===
# 12 Analiza obrazu i reguła alarmu
@fig assets/05-zdarzenie.png | Zdarzenie przechodzi przez trwałe stany i niezależne efekty uboczne | 4.1

Foundry zwraca dane, a nie instrukcje dla urządzenia. Proponowany wynik ma glass_visible i person_visible z wartościami true, false lub unknown, image_quality, krótki opis obserwacji, model_deployment oraz prompt_version. Pole authorization pozostaje unknown, dopóki system nie ma niezależnego źródła uprawnień. Ocena pewności generowana przez model nie jest skalibrowanym prawdopodobieństwem.

Reguła uzbrojenia i reakcji na widoczną osobę bez rozpoznawania domowników wymaga odpowiedzi Wiktora. Architektura obsługuje wariant uzbrojony glass_visible=true AND person_visible=true oraz wariant ręcznego zatwierdzania. Dopóki reguła nie zostanie zaakceptowana, automatyczny buzzer pozostaje wyłączony, a zdarzenie trafia do Review required. To nierozstrzygnięta decyzja wdrożeniowa, nie wynik analizy modelu.

Wyjątek, odmowa modelu, brak obrazu, błędny JSON i unknown nie są równoważne potwierdzeniu zagrożenia. Trafiają do stanu wymagającego weryfikacji lub błędu. Odbiorca e-maila i treść szablonu są skonfigurowani po stronie backendu; model nie może podmienić adresu, dołączyć innych plików ani wywołać dowolnego narzędzia.

Pierwsza próba obejmuje cztery kombinacje obecności szkła i osoby, scenę nieczytelną oraz brak odpowiedzi API. Zdjęcie ekranu laptopa jest testem integracji i odporności na artefakty obrazu. Nie jest dowodem skuteczności rozpoznawania prawdziwego włamania. Materiały syntetyczne mają osobne oznaczenie i nie mieszają się z ewaluacją na rzeczywistych scenach.
===
# 13 Champion i ciągły pipeline badawczy
@fig assets/06-model.png | Wybór i test modelu bez ponownego uczenia na zbiorze testowym | 4.3

Trening docelowo działa lokalnie na MacBooku M5 Max 128 GB w natywnym środowisku ARM64. Marcel dobiera CPU/MPS i równoległość na rzeczywistym benchmarku; 128 GB jest pamięcią współdzieloną z systemem. Profil oraz wersje zapisujemy w manifeście. Azure realizuje inferencję. Marcel i Kacper dostarczają odtwarzalny pipeline, a Patryk sprawdza zgodność pakietu z runtime. Trening i wybór topologii używają train oraz validation. Końcowy continuous test uruchamia zamrożony model bez aktualizacji wag, wyboru progu i ponownego treningu. Liczba ziaren i sposób agregacji są zapisane przed oceną.

Eksport musi pochodzić z dokładnie tego samego checkpointu, który oceniono. Pakiet zawiera model, topologię, encoder_config, decoder_config, sim_config, manifest danych, raport ewaluacji i hashe. Obecna ścieżka retreningu podczas eksportu wymaga zmiany, bo tworzy inny model niż wcześniej oceniony. [R2]

Progi i cooldown wybieramy na walidacji. Dla budżetów FA/h prezentujemy wynik walidacyjnego doboru oraz osiągnięty punkt na teście, nawet jeżeli test przekroczy budżet. Nie wybieramy najlepszego punktu z testu i nie nazywamy go niezależnym wynikiem.

Dashboard importuje champion jako wersję immutable. Swobodna edycja tworzy kopię sandbox z nowym identyfikatorem. Taka kopia może działać w symulacji, lecz traci etykietę evaluated champion. Powrót do poprzedniego championa oznacza uruchomienie nowej sesji z poprzednim pakietem, nie mieszanie wag w bieżącym stanie neuronów.
===
# 14 Model Lu.i i znaczenie cyfrowego bliźniaka
@fig assets/lui-fotografia-zrodlowa.jpg | Rzeczywisty projekt Lu.i z repozytorium giantaxon jako odniesienie kształtu i elementów | 3.6

Lu.i realizuje analogowy neuron leaky integrate-and-fire. Dokumentacja wskazuje trzy synapsy z regulowaną wagą i znakiem oraz wspólną stałą synaptyczną, regulację potencjału spoczynkowego i stałej błonowej oraz trzy terminale wyjścia. Pasek LED przedstawia potencjał, a dioda spike zdarzenie wyjściowe. [S1, S2]

Model matematyczny obejmuje zanik prądu synaptycznego i integrację potencjału. W zapisie referencyjnym: tau_mem · dV/dt = −(V − V_leak) + I_syn/g_leak. Dla każdego wejścia określamy znak, siłę i kształt pobudzenia. Po przekroczeniu progu model generuje spike, przechodzi reset i uwzględnia zmierzoną refrakcję. Wagi bez jednostek nie stają się automatycznie prądem w amperach.

Określenie „cyfrowy bliźniak” ma trzy poziomy: wizualny, funkcjonalny i skalibrowany. Poziom wizualny odwzorowuje elementy płytki. Funkcjonalny odtwarza przyjęty model dynamiki. Dopiero porównanie z konkretną płytką, z zakresem i tolerancją błędu, uzasadnia opis „skalibrowany”. Nie wymagamy matematycznie identycznych przebiegów rzeczywistego układu analogowego.

Patryk definiuje reset, refractory, próg, integrator i dt. Andrzej mierzy szerokość impulsów oraz charakterystyki płytek. Interfejs źródłowy podaje około 15 ms i 2,5 V na wyjściu, lecz te wartości trzeba sprawdzić na posiadanych egzemplarzach. Na dashboardzie jednostki fizyczne pojawiają się dopiero po kalibracji; wcześniej używamy a.u. [S1]
===
# 15 Kalibracja i ograniczenia sprzętowe sieci
@fig assets/07-kalibracja.png | Kalibracja i niezależny test zgodności symulacji z Lu.i | 3.8

Dla każdej płytki zapisujemy board_id, rewizję PCB, napięcie, temperaturę, ustawienia potencjometrów i identyfikator pomiaru. Minimum pomiarowe obejmuje odpowiedź na pojedynczy impuls, serię impulsów, hamowanie, zanik potencjału, reset i maksymalną częstość spike’ów. Profile kalibracyjne nie są wspólne domyślnie dla wszystkich egzemplarzy.

Zestaw bodźców dzielimy na dopasowanie i test. Wynikiem są odchylenia przebiegu Vmem, czasów spike’ów, liczby spike’ów i decyzji sieci. Proponowane kryteria robocze do uzgodnienia przez Patryka i Andrzeja: NRMSE potencjału względem ustalonego zakresu, p95 błędu czasu i udział zgodnych decyzji. Nie wpisujemy arbitralnej tolerancji jako już osiągniętej jakości.

Tryb Hardware compatible ogranicza fan-in do trzech fizycznych synaps, wymaga współdzielonego tau_syn na płytce i dopuszczalnego zakresu ustawień. Dostępne terminale wyjścia nie dowodzą dowolnego fan-out: obciążenie wielu wejść i przewody wymagają pomiaru. Eksport nie może cicho obcinać czwartego wejścia ani pomijać małych wag.

Suwak potencjometru w UI nie jest zdalnym sterowaniem płytką. Pokazuje ustawienie zadane lub zmierzone i instrukcję ręcznego dostrojenia. Fizyczny Vmem każdego neuronu wymaga odpowiedniego toru akwizycji; jeśli dostępne są tylko wyjściowe spike’y, UI oznacza potencjał jako modelowany. Ta sama płytka drugiego zespołu ułatwia porównanie, ale nie zastępuje profilu jej konfiguracji i źródła danych.
===
# 16 Zatwierdzona referencja dashboardu
@fig assets/ui-reference-en.png | Referencja UI w języku angielskim oparta na zatwierdzonym ciemnym projekcie | 4.55

Wiktor zatwierdził ciemny wzorzec v2. Wersja v3 lokalizuje go na angielski zgodnie z późniejszym wymaganiem; potwierdzenie tej konkretnej ilustracji jest oczekujące. Wymagania funkcjonalne z kolejnych rozdziałów rozstrzygają szczegóły, których makieta nie pokazuje.

Dominantą ekranu Network jest ciemny stół z rozłożonymi płytkami Lu.i. Pasek boczny zawiera Network, Events, Experiments, Energy i Device. Górny pasek identyfikuje sesję, tryb i źródło danych. Karty metryk zajmują mało miejsca, a po prawej znajduje się inspektor zaznaczonego neuronu. Dół zawiera raster i listę zdarzeń.

Ilustracja określa hierarchię, kolor i charakter ekranu. Źródłowa geometria płytki, liczba elementów, porty, wykresy i wartości wynikają z specyfikacji i danych. Kształt końcowego komponentu powinien bazować na dostarczonym neuron.svg oraz fotografii Lu.i, nie na niedokładnym odwzorowaniu wygenerowanego obrazu.

Motyw: tło #10151D, panele #19212B, tekst #E8EDF2, akcent fioletowy, LED bursztynowe. Pobudzenie oznaczamy kolorem turkusowym i symbolem +, hamowanie magentą i symbolem −. Kolor nie jest jedynym nośnikiem znaczenia. Zdarzenia błędów mają tekstowy status. Animacje nie mogą zastępować danych liczbowych ani legendy.
===
# 17 Ekran Network i edycja topologii
## Układ i zachowanie obszaru sieci
Dla szerokości od 1440 px proponujemy sidebar 200 px, inspektor 320 px i resztę dla canvas. Przy 1024–1439 px sidebar zwija się do ikon z etykietą po fokusie, a inspektor może być wysuwany. Na mniejszych ekranach zachowujemy podgląd i historię; edycja gęstej sieci wymaga powiększenia obszaru. Nie zmniejszamy tekstu do nieczytelnego rozmiaru.

Board count przyjmuje liczby całkowite 0–50 w trybie Edit. Dla zera pokazujemy pusty stan „Add a board or import a model”; Start simulation jest nieaktywny. Dla jednego neuronu płytka jest wyśrodkowana. Dla wielu dostępne są Layered i Grid oraz Fit view. Układ warstwowy wynika z połączeń; dla sieci rekurencyjnej lub ręcznej używamy układu siatkowego albo zapisanych współrzędnych.

Dla 50 płytek overview upraszcza drobne napisy, ale nie usuwa neuronów. Zoom przywraca szczegóły. Sieć można przesuwać, zaznaczać prostokątem i wyszukiwać po ID. Obszar zachowuje odstępy i unika nakładania płytek; przewody mogą się krzyżować, a zaznaczenie neuronu podświetla wyłącznie jego połączenia.

## Reguły edycji
Dodanie płytki tworzy stabilny neuron_id. Usunięcie wymaga potwierdzenia liczby usuwanych połączeń i można je cofnąć. Przeciąganie zmienia tylko layout. Utworzenie przewodu wymaga portu wyjściowego i wejściowego; zmiana znaku i wagi jest osobną operacją. Self-loop i cykle są dopuszczalne wyłącznie, jeśli wspiera je runtime wybranego trybu.

Save draft zapisuje wersję sandbox. Validate sprawdza porty, jednostki, zakresy, liczbę wejść oraz zgodność z trybem hardware. Apply to new session nigdy nie modyfikuje bieżącego championa. Wyniki starej konfiguracji nie pozostają widoczne jako wynik nowej sieci. Tryb Live blokuje edycję parametrów obliczeniowych, ale pozwala przestawiać płytki.

Import JSON jest walidowany i nie wykonuje kodu. Export zawiera topologię, parametry, layout i wersję schematu. Oddzielamy liczbę fizycznych neuronów od liczby kanałów enkodera: wejście cechowe nie jest automatycznie kolejną płytką Lu.i.
===
# 18 Komponent płytki i inspektor neuronu
| Element interfejsu | Źródło i zachowanie |
| Obrys PCB | Dostarczone SVG; proporcje zachowane przy skalowaniu |
| Trzy wejścia | Stabilne identyfikatory portów, znak + lub − i przypięty sygnał |
| Wyjście spike | Wspólny logiczny sygnał; terminale odpowiadają dokumentacji płytki |
| Pasek LED | Sześć segmentów potencjału według profilu wizualizacji lub kalibracji |
| Spike LED | Osobny impuls wizualny wywołany zdarzeniem spike, nie timerem losowym |
| Etykieta neuronu | ID, rola, status aktywności i źródło danych |
| Błąd konfiguracji | Ikona i opis; płytka nie znika z obszaru sieci |

Stała tau wpływa na dynamikę potencjału, a nie bezpośrednio na częstotliwość animacji diody. Pasek LED jest funkcją Vmem. Przed kalibracją można użyć jawnie oznaczonego mapowania liniowego do sześciu poziomów; nie nazywamy go pomiarem progów sprzętowych. Po kalibracji każdy segment ma próg z profilu.

Diodę spike można wizualnie podtrzymać np. 80 ms, aby człowiek dostrzegł krótszy impuls. To propozycja czasu animacji; surowy timestamp i szerokość impulsu pozostają bez zmian. Przy dużej częstości pokazujemy częstotliwość oraz zagęszczenie rastra, zamiast obiecywać rozróżnienie każdego błysku. Opcja Reduced motion wyłącza ruch kropek bez utraty informacji.

Inspektor ma zakładki Signals, Parameters, Connections i Calibration. Signals pokazuje Vmem, próg, reset oraz raster w wspólnym oknie czasu. Parameters zawiera tau_mem, tau_syn, V_leak, V_threshold, V_reset, refractory i wersję integratora; pola nieobsługiwane przez fizyczną płytkę nie udają regulowanych potencjometrów.

Connections pokazuje port, source_id, znak, wagę, jednostkę i szerokość bodźca. Calibration pokazuje board_id, źródło pomiaru, datę, napięcie i zakres zgodności. Dane mają jawne pochodzenie: Simulated, Measured, Estimated lub Unavailable. Nie wstawiamy wartości zero w miejsce brakującego pomiaru.
===
# 19 Live Replay i synchronizacja obrazu
## Live
Start monitoring rozpoczyna sesję detekcji po preflight. Pause view zatrzymuje wyłącznie aktualizację obrazu w przeglądarce i pokazuje komunikat „Monitoring continues”. Stop monitoring zatrzymuje sesję systemu i wymaga świadomej akcji operatora. Te trzy operacje muszą być osobnymi kontrolkami, aby pauza animacji nie wyglądała jak wyłączenie ochrony.

Górny status pokazuje Running, Warming up, Degraded, Stopped lub Reconnecting, numer epoki i opóźnienie telemetrii. Gdy dane są stare, obszar sieci otrzymuje etykietę Stale data oraz czas ostatniej próbki; diody nie kontynuują fikcyjnej animacji. Nie zerujemy potencjału tylko dlatego, że odłączono SSE.

## Replay
Operator wybiera run_id, nagranie i model. Oś czasu pokazuje zdarzenia ground truth, trigger SNN, capture, decyzję i komendy, jeżeli należą do tego samego eksperymentu. Prędkości 0.25×, 1× i 4× dotyczą odtwarzania, nie biologicznego tau. Seek używa zapisanego checkpointu albo deterministycznego odtworzenia od wcześniejszego punktu; nie ustawia neuronów arbitralnie na zero w środku nagrania.

Replay ma widoczne „Actuators disabled”. Historycznych impulsów nie wolno kierować do aktywnego urządzenia bez oddzielnego trybu testu sprzętowego. Tryb ten wymaga odrębnej sesji i operatora, a historia nie zmienia się w live przez zmianę zakładki.

## Telemetria a raster
Snapshot zawiera source_time, model_hash, topology_version, epoch, wektor Vmem i status danych. Delta zawiera zmiany oraz zdarzenia spike z timestampami. Frontend może redukować liczbę punktów wykresu, zachowując ekstremalne wartości i progi; nie zmienia archiwum potrzebnego do badań. Surowe impulsy są oddzielone od próbkowanego Vmem.

Na osi czasu zakresy warmup, gap i invalid mają wyróżnione tło i opis. Hover pokazuje dokładny czas źródłowy, neuron i pochodzenie. Kliknięcie triggera przechodzi do Events z tym samym event_id. Dane syntetyczne mają trwały znacznik Demo i nie mogą być eksportowane jako pomiar sprzętowy.
===
# 20 Ekran Events i analiza zdarzenia
@fig assets/ui-event-en.png | Uzupełniający widok szczegółu zdarzenia w tym samym stylu | 4.35

Lista Events zawiera czas, event_id, session_id, status SNN, jakość obrazu, decyzję, alarm i powiadomienie. Filtry obejmują zakres czasu, model, urządzenie, tryb oraz błędy. Statusy Photo requested, Analyzing, Alarm confirmed, No alarm, Review required i Failed są odróżnione tekstem. Widok nie sprowadza wszystkich etapów do jednej zielonej lub czerwonej kropki.

Szczegół zdarzenia pokazuje zdjęcia, obserwacje modelu i wynik reguły osobno. Pod obrazem znajdują się czas wykonania zdjęcia, opóźnienie względem triggera oraz etykieta Synthetic, Screen capture test lub Real scene. Nierozpoznane uprawnienia pozostają Unknown niezależnie od widoczności osoby.

Decision trace obejmuje wersje SNN, enkodera, promptu, wdrożenia Foundry i reguły. Actions zawiera osobne czasy command issued, acknowledged i completed oraz status e-maila. Operator może nadać ground truth do późniejszej oceny; ta adnotacja ma autora i czas i nie nadpisuje pierwotnej decyzji.

Eksport zdarzenia zawiera rekord JSON i opcjonalnie zdjęcie dla uprawnionego operatora. Domyślny eksport metryk nie zawiera obrazów ani danych dostępu. Uzupełniająca makieta pokazuje przykład Review required; nie zastępuje uzgodnionej tabeli reguł alarmowych.
===
# 21 Ekran Experiments i interpretacja wyników
Ekran Experiments odpowiada na pytanie, co oceniono i z jaką wiarygodnością. Operator wybiera run_id i widzi hash modelu, seed treningu, wersję danych, seed strumienia, encoder_config, czas oceniany, liczbę niezależnych grup i status Complete albo Incomplete. Wynik niekompletny nie jest domyślnie kandydatem na champion.

| Sekcja | Zawartość i zachowanie |
| Quality | Event recall, FA/h, event precision i liczności TP FP FN |
| Operating point | Próg wybrany na validation, budżet FA/h i faktyczny wynik testu |
| Curves | Recall względem FA/h, porównanie modeli i przedziały ufności |
| Background | FA/h i czas ekspozycji osobno dla speech, stationary, loud_event, animal |
| Latency | p50, p95 i liczba zdarzeń; oddzielnie SNN, capture, vision i end to end |
| Provenance | Zbiory źródłowe, hashe, reguła dopasowania, cooldown, warmup |
| Export | CSV metryk i manifest JSON z jednostkami oraz wersją schematu |

Wykresy nie pokazują linii „przewagi SNN”, dopóki nie ma porównywalnych wyników. Brak ekspozycji tła oznacza FA/h = Not available, a nie zero. Brak pozytywnych zdarzeń oznacza nieokreślony recall. Gdy FP=0, pokazujemy zarówno zero zaobserwowanych alarmów, jak i górną granicę niepewności.

Wyniki okienkowe F1 mają osobny panel Clip metrics. Nie przeliczamy odsetka błędnych klipów na FA/h bez czasowego strumienia i reguły grupowania triggerów. Budżety 0.01, 0.1, 1 i 5 FA/h są punktami odniesienia badań, a nie obietnicą osiągnięcia wszystkich punktów przez aktualny model.

Eksperyment i dashboard mają wspólny backend obliczania metryk. Frontend formatuje wartości, ale nie tworzy alternatywnej definicji FA/h. Zmiana reguły dopasowania lub wyłączeń tworzy nową wersję raportu. Wersja wcześniej wykorzystana w artykule pozostaje odtwarzalna.
===
# 22 Ekrany Energy i Device oraz odbiór UI
## Energy
Widok Energy ma przełącznik granicy Edge only albo Extended estimate. Dla każdego wariantu pokazuje P_idle, P_active, energię przyrostową zdarzenia, czas pomiaru, napięcie i źródło. Każda liczba ma status Measured, Estimated lub Missing. Pole Power nie może pokazywać „zużycia symulowanych neuronów” w watach bez określonego modelu estymacji.

Wykres break-even ma oś FA/h i energię na godzinę w Wh. Operator widzi założenia: częstość zdarzeń prawdziwych, recall, koszt aktywacji i stan LED. Edycja założeń tworzy scenariusz What-if, nie modyfikuje pomiaru. Wynik pochodzi z modelu w rozdziale 26 i jest oznaczony jako estymacja.

## Device
Widok Device pokazuje model Pi, system i architekturę CPU, wersję serwisu, boot_id Uno, encoder_hash, stan serial, camera ready, GPIO ready, ostatni heartbeat i liczbę utraconych zdarzeń. Pokazuje status sieci, kolejki i czas pozostały do wygaśnięcia ostatniej komendy. Przycisk Test capture zapisuje zdarzenie testowe bez alarmu; Test alarm jest osobną, limitowaną akcją operatora.

## Kryteria odbioru interfejsu
1. Cały tekst produktu, w tym błędy, etykiety, tooltipy i eksportowane nagłówki, jest po angielsku. Opisy w dokumencie mogą być po polsku.
2. Liczby płytek 0, 1, 8, 25 i 50 działają bez utraty elementów; zoom i Fit zachowują czytelność. Usuwanie połączeń jest jawne i odwracalne w edytorze.
3. Ten sam timestamp daje ten sam stan w replay. Symulacja nie zależy od FPS przeglądarki.
4. Missing, stale, error i zero są odróżnialne. Źródło Measured albo Simulated jest widoczne przy przebiegu.
5. Obsługa klawiatury obejmuje nawigację, wybór neuronu i formularze. Fokus jest widoczny; kluczowe statusy nie zależą wyłącznie od koloru.
6. Pause view nie zatrzymuje monitorowania, Stop monitoring nie jest ukrytym efektem zamknięcia zakładki, a Replay nie uruchamia buzzera.
===
# 23 Dataset i generowanie strumieni ciągłych
@fig assets/08-dane.png | Podział źródeł poprzedza cięcie i mieszanie strumieni | 3.35

Raport v2.0.0 w repozytorium podaje 10 853 klipy, 4878 grup źródłowych i około 20,8 h audio. To liczności raportu źródłowego, nie nowo wykonany audyt. Dane są mono PCM16 44,1 kHz; enkoder odpowiada za kontrolowaną konwersję do konfiguracji urządzenia. [R5]

| Split | Klipy | Grupy | Czas h | Grupy pozytywne |
| Train | 7120 | 3337 | 14,17 | 156 |
| Validation | 1866 | 770 | 3,22 | 100 |
| Test | 1867 | 771 | 3,41 | 96 |

Źródła obejmują DataSEC, ESC-50 i VOICe. W raporcie udokumentowano korektę 1412 fałszywie ujemnych i 241 fałszywie dodatnich etykiet VOICe. Wycięte fragmenty wspólnego miksu nie są niezależnymi nagraniami; także źródła składowe miksów wymagają audytu pochodzenia, jeśli dostępna jest taka informacja. Licencje i przypisanie źródeł należy zachować w manifeście. [R5]

Generator continuous tworzy tło z rozmieszczonymi zdarzeniami oraz manifestem czasowym. Obecna wersja zakłada pięć zdarzeń i domyślny warmup 30 s, a tryby clean i background różnią się dopuszczeniem nakładających klas. To dobre narzędzie integracyjne, lecz pięć zdarzeń nie wystarcza do precyzyjnej oceny recall. Powtarzanie tych samych źródeł pod nowymi seedami nie zwiększa liczby niezależnych przykładów.

Do finalnego testu wymagamy rozłączności grup szkła i tła względem train oraz validation, audytu etykiet i zapisania wszystkich transformacji. README generatora i jego kod mają rozbieżność dotyczącą tła; rozstrzyga test manifestów i rzeczywistych plików, nie nazwa katalogu. [R3]
===
# 24 Metryki ciągłe i reguła dopasowania
## Definicje obowiązujące dla raportu
Event recall = TP / (TP + FN). FA/h = FP / T_negative_h, gdzie T_negative_h jest sumą ocenianych przedziałów bez zdarzenia docelowego, po wyłączeniu warmup, luk i ustalonych marginesów dopasowania. Zawsze zapisujemy licznik FP i mianownik, a nie wyłącznie iloraz. Alarm końcowy i trigger SNN mają osobne FP oraz osobne raporty.

Proponowany protokół dopasowuje trigger do adnotacji od onset do offset + tolerance_after. Tolerancję ustalamy na walidacji i zapisujemy przed testem. Każde zdarzenie może otrzymać jeden TP, a każdy trigger pasować do jednego zdarzenia. Duplikaty w tym samym zdarzeniu raportujemy oddzielnie; wpływają na liczbę aktywacji i energię, nawet jeżeli cooldown usuwa część z nich z reguły alarmowej.

Czas negatywny liczymy po odjęciu sumy przedziałów pozytywnych wraz z marginesami, bez podwójnego odejmowania nakładających się odcinków. FA/h per kind używa czasu danego rodzaju tła; segmenty o wielu klasach muszą mieć deterministyczną regułę przypisania albo osobną kategorię mixed. Nie sumujemy podgrup z nakładającymi się mianownikami jako niezależnych godzin.

## Niepewność i ekspozycja
Raport zawiera przedziały ufności recall i FA/h, z uwzględnieniem grup źródłowych. Bootstrap losuje grupy lub sesje, nie pojedyncze skorelowane okna. Przy zerze FP jednostronna górna granica 95% w prostym modelu Poissona wynosi około 3/T_negative_h. Przykładowo brak FP przez 3 h nie dowodzi poziomu 0,01 FA/h; do granicy około 0,01 potrzeba około 300 h ekspozycji przy założeniach tego modelu.

Pokazujemy p50 i p95 opóźnienia detekcji względem onset oraz opóźnienia całego alarmu. Czas trwania etapu treningowego nie jest latency detekcji. Recall oceniamy również przed końcową wizją: brak osoby w kadrze nie powinien zmieniać etykiety poprawnej detekcji akustycznej szkła.

Raportuje się surowe triggery, aktywacje po cooldown i końcowe alarmy. To pozwala powiązać FA/h z rzeczywistą liczbą uruchomień kamery i wywołań Foundry, zamiast utożsamiać wszystkie trzy wielkości.
===
# 25 Granice pomiaru energii
@fig assets/09-energia.png | Oddzielne granice pomiaru lokalnego i rozszerzonej estymacji | 4.0

Porównanie obowiązkowe: symulowany detektor SNN i klasyczny detektor FFT na Uno. Fizyczne Lu.i oraz FFT na komputerze są dodatkowymi punktami odniesienia, wyłącznie jeśli zostaną uruchomione i ocenione według tego samego protokołu. Raportujemy osobno (a) jakość i czas samego detektora przy wspólnej granicy oraz (b) właściwości całego wdrożonego toru Uno–Pi–sieć–Azure. W wariancie (b) uwzględniamy transmisję i pobór energii urządzeń; nie można pominąć ich z powodu przyszłego wariantu analogowego. Dla każdej konfiguracji zapisujemy miejsce enkodera, model, tor transmisji, identyczny zbiór testowy i stan hosta. Inna granica pomiaru oznacza inny wynik, nawet przy tej samej nazwie „SNN”.

Pomiar edge obejmuje mikrofon, Uno, Pi, kamerę, alarm i ewentualne Lu.i, jeśli należą do danego wariantu. Przy zasilaniu Uno z Pi nie dodajemy ponownie jego energii do pomiaru na wspólnym wejściu USB. Pomiar przy zasilaczu zawiera jego straty, a na szynie DC może ich nie zawierać; punkt pomiarowy musi być opisany.

Andrzej mierzy spoczynek i aktywność Lu.i, osobno ze standardową sygnalizacją LED oraz wariantem LED off, jeśli technicznie dopuszczalny. Nie porównujemy Lu.i bez LED z demonstratorem z LED bez wskazania różnicy. Potrzebne są napięcie, średnia moc, energia, czas i rozrzut co najmniej trzech powtórzeń.

W chmurze czas CPU, liczba operacji i rachunek Azure nie są bezpośrednim pomiarem energii fizycznego serwera. Jeśli nie ma odpowiedniej telemetrii, podajemy koszt i czas obliczeń oraz osobno scenariusz estymacji energii. Nie przeliczamy spike’ów na pJ/SOP zaczerpnięte z innego chipu i nie przypisujemy tej liczby Lu.i ani CPU Azure.
===
# 26 Model energetyczny przy stale włączonym Pi
## Bilans z poprawnymi jednostkami
Dla okresu T sekund proponujemy model: E_total[J] = P_base[W] · T[s] + N_activation · E_activation_incremental[J] + E_other[J]. P_base obejmuje stale aktywne składniki w przyjętej granicy. Energia przyrostowa jednej aktywacji to całka z P(t) − P_base w czasie obsługi zdarzenia. Dzieląc dżule przez 3600, otrzymujemy Wh.

To zastępuje wcześniejszy model kosztu bootowania Pi. W nowej architekturze nie ma oszczędności wynikającej z halt pomiędzy zdarzeniami; jest oszczędność z niewykonywania obrazu i analizy bez potrzeby. Jeżeli kamera pozostaje zasilana, jej pobór w oczekiwaniu jest częścią P_base.

Liczba aktywacji uwzględnia prawdziwe wykrycia, fałszywe triggery, duplikaty i ograniczenia cooldown. Nie wystarcza samo FA/h końcowego alarmu, ponieważ nawet odrzucona przez vision sytuacja zużyła energię przechwycenia i analizy.

## Warunek porównania
Przy tym samym czasie obserwacji, porównywalnym recall i takim samym koszcie pojedynczej aktywacji SNN jest korzystniejsze od FFT, gdy:

3600 · (P_base_SNN − P_base_FFT) + (N_SNN − N_FFT) · E_activation_incremental < 0

N dotyczy jednej godziny. Jeśli koszty aktywacji są różne, stosujemy oddzielne E_activation dla wariantów. Jeśli zdarzenia zachodzą na siebie, mierzymy wspólny przedział pracy albo stosujemy model stanów; prosta suma pełnych kosztów może zawyżać zużycie.

Przykład wyłącznie ilustracyjny: dodatkowe 0,1 W stałego poboru kosztuje 360 J na godzinę. Przy 10 J kosztu aktywacji potrzeba uniknąć ponad 36 aktywacji/h, aby pokryć tę różnicę. To pokazuje sens progu, ale nie opisuje pomiarów zespołu.

Do wykresu zapisujemy niepewność P_base i E_activation, liczbę aktywacji oraz wariant źródła danych. Lu.i drugiego zespołu może dostarczyć pomiary komponentów przy tej samej konfiguracji; nie dowodzi jakości klasyfikatora, jeśli topologia, enkoder lub zbiór są inne.
===
# 27 Wdrożenie Azure i koszt pracy
@fig assets/10-azure.png | Proponowane usługi i rozdzielenie pracy online od zadań vision | 4.3

Proponujemy jeden kontener API z dashboardem i runtime SNN oraz drugi worker z kodem vision i powiadomień. Oba mogą powstać z jednego obrazu z różnymi poleceniami startu. Blob Storage przechowuje artefakty, Table Storage rekordy, a Queue Storage zadania. Nie wymagamy klastra Kubernetes, Redis ani stale aktywnej bazy SQL dla demonstratora.

API ma docelowo minReplicas=0 i maxReplicas=1. Podczas kontrolowanej sesji pomiarowej proponujemy profil minReplicas=1 ustawiany przed Start, aby wyłączyć cold start i przypadkowe skalowanie do zera. Po Stop, zapisie wyników i zakończeniu zadań profil wraca do zera. Zwykły ruch sesji również podtrzymuje aktywność, ale nie zastępuje jawnego profilu pomiarowego. Worker skaluje się z kolejki; zimny start może wydłużyć pierwszy alarm. [S6]

Wiktor sprawdza przez Azure CLI dozwolone regiony, dostępność Container Apps, Storage, Key Vault i wybranego vision deploymentu. Model Foundry wybieramy po sprawdzeniu obsługi obrazów, sposobu uwierzytelnienia, limitów, opóźnienia i ceny na konkretnej subskrypcji. Dostępność portalu nie gwarantuje dostępnego quota dla dowolnego modelu.

Budżet obejmuje vCPU i RAM aktywnych kontenerów, żądania HTTP, operacje i pojemność Storage, logi, rejestr obrazów, Key Vault i wywołania Foundry. Zero replik nie oznacza zerowego rachunku wszystkich usług. 250 ms batch oznacza około 14 400 żądań/h na urządzenie przed doliczeniem zdjęć i UI; ten koszt porównujemy z opóźnieniem dla batch 500 ms lub 1 s. [S6, S7]
===
# 28 Dostęp i przechowywanie danych
## Tożsamości
Portal używa jednego wspólnego loginu i hasła dla zespołu, zgodnie z decyzją Wiktora z 24 września 2026. Każda sesja ma te same uprawnienia operatora; nie budujemy kont imiennych ani ról viewer/operator. Formularz Username/Password i Logout jest po angielsku. Dane logowania lub hash przechowujemy w Key Vault, bez domyślnych wartości. Serwer egzekwuje sesję, jej TTL, bezpieczne cookie, CSRF i limit prób. Konto wspólne nie daje audytu osobowego. Urządzenie używa osobnego poświadczenia i ma dostęp tylko do własnej sesji, zdjęć i ACK. Szczegóły wdrożenia zawiera Plan zespołu, rozdział 5.

Backend i worker używają managed identity do obsługujących ją usług. Pozostałe sekrety są pobierane z Key Vault przez kod infrastrukturalny i nie trafiają do promptu ani telemetrii. Pi nie ma automatycznie managed identity Azure: wymaga provisioningu odrębnego poświadczenia urządzenia, np. certyfikatu lub rotowanego tokena ograniczonego do API. To poświadczenie musi istnieć lokalnie w chronionym pliku; sam Key Vault nie rozwiązuje pierwszego uwierzytelnienia Pi. [S8]

Model vision otrzymuje obraz i instrukcję obserwacji. Tekst na fotografii jest danymi, nie poleceniem dla agenta. Dostęp do e-maila, storage i GPIO pozostaje w kodzie, poza swobodnym wykonaniem narzędzi przez model. Logi redagują nagłówki Authorization, podpisane URL i treść sekretów.

## Proponowana retencja do zatwierdzenia przed wdrożeniem
Zdjęcia demonstracyjne: 7 dni; zdarzenia i techniczne metryki: 30 dni; zamrożone raporty badań i nieidentyfikujące manifesty: do zakończenia pracy naukowej według ustaleń zespołu. To parametry projektowe, nie istniejąca polityka organizacji. Dane archiwalne do publikacji wybieramy oddzielnie i opisujemy pochodzenie.

Blob nie jest publiczny. Dashboard pobiera zdjęcia przez autoryzowany backend lub krótko ważny dostęp ograniczony do obiektu. E-mail ze zdjęciem tworzy kopię poza retencją Blob; odbiorcy muszą być jawnie skonfigurowani. Usunięcie lokalnego obiektu nie usuwa automatycznie załącznika ze skrzynki odbiorcy.

Rozpoznawanie domowników pozostaje poza pierwszą wersją. Przy jego dodaniu potrzebny będzie odrębny projekt danych referencyjnych, oceny pomyłek i zasad dostępu; nie jest to samo dodanie kolejnego pytania do promptu vision.
===
# 29 Testy odbioru systemu
| Próba | Dowód zaliczenia |
| Kontrakt encoder–SNN | Te same kanały, jednostki i konfiguracja; niezgodność blokuje Start |
| Streaming | Godzina pracy przy docelowym ruchu; brak niewyjaśnionych luk i rosnącej kolejki |
| Stan SNN | Podział tego samego wejścia na różne batch daje zgodne wyjście |
| Kamera | Poprawny JPEG po triggerze, timestamp i event_id; brak zdjęć w ciszy bez triggera |
| Reguła vision | Wszystkie kombinacje szkła i osoby, unknown oraz timeout zgodne z tabelą reguł |
| Retry komendy | Powtórne dostarczenie nie powtarza zakończonego wykonania |
| Awaria chmury | Jawna luka, nowa epoka i warmup zamiast cichego resetu |
| Brak internetu | Ograniczony bufor, widoczny Degraded, brak alarmu z przeterminowanej komendy |
| UI | 0–50 płytek, stale data, angielskie etykiety, rozdzielenie Pause view i Stop |
| Trening i test | Hash eksportu zgodny z ocenionym checkpointem; test bez strojenia |
| Energia | Znana granica, czas, powtórzenia, jednostki i źródło każdej wartości |
| Pi Zero | Ten sam scenariusz E2E i budżety zasobów po migracji, jeśli sprzęt dostępny |

Próba batch-invariance jest szczególnie ważna: jeśli wynik zmienia się przy podziale strumienia na porcje 100, 250 i 1000 ms, transport wpływa na model albo resetuje jego stan. Test ma użyć identycznych timestampów wejścia i porównać decyzje oraz czasy spike’ów w określonej tolerancji integratora.

Wydajność mierzymy oddzielnie dla akwizycji, transportu, SNN, kamery, Foundry i reakcji. Nie wpisujemy bez pomiaru gwarancji „alarm poniżej jednej sekundy”. Sieć, cold start i wybrany model mogą dominować wynik. Celem odbioru jest podanie rozkładu opóźnień oraz odsetka błędów, z rozdzieleniem prób ciepłych i zimnych.

Minimalny raport E2E zawiera konfigurację hardware, wersje kodu, model i prompt, przebieg jednego poprawnego alarmu, poprawnego odrzucenia, niepewnej analizy i przerwy łączności. Wynik demonstracji na obrazku z ekranu jest oznaczony jako test integracyjny.
===
# 30 Współpraca i kolejność integracji
| Osoba | Odpowiedzialność | Artefakt przekazania |
| Andrzej | Hardware, Lu.i i pomiary energii | pinout stanowiska, profile kalibracji, surowe pomiary |
| Patryk | Symulator i zgodność Lu.i | runtime streaming, model schema, telemetria neuronów |
| Wiktor | Pi, kamera, alarm, integracja i Azure CLI | serwis edge, konfiguracja środowiska, test E2E |
| Karolina | Dashboard i wizualizacja | komponent płytki, ekrany, obsługa telemetrii i edytora |
| Marcel | Master pipeline i końcowa ewaluacja | niezmienny champion, raport continuous, manifest |
| Kacper | Enkoder i wsparcie continuous | specyfikacja kanałów, firmware, test zgodności danych |

Pierwszym punktem integracji jest wspólny manifest enkodera i protokół impulsów. Karolina może równolegle pracować na jawnie syntetycznej telemetrii, ale mock musi mieć dokładnie ten sam kontrakt co Patryk. Wiktor może sprawdzić aparat i wyjścia bez czekania na trening, korzystając z kontrolowanej komendy testowej.

Następnie spinamy Uno → Pi → API → SNN i sprawdzamy ciągłość czasu. Dopiero potem dodajemy zdjęcie, kolejkę vision i efekty uboczne. W ten sposób błąd kamery nie maskuje błędu enkodera, a brak championa nie blokuje technicznego testu przesyłania danych. Wyniki mocka nie są wynikiem naukowym.

Po stabilnym E2E zamrażamy wersje urządzenia, protokołu i modelu na pomiar. Trening może nadal trwać osobno, lecz nie zmienia działającej sesji. Nowy champion otrzymuje oddzielne wdrożenie i raport, zamiast podmieniać plik w tle.

Priorytet przy trzydniowym terminie: poprawny przepływ, odtwarzalny continuous test, pomiar energii i czytelny dashboard. Rozpoznawanie domowników, rozbudowane role, zaawansowany edytor rekurencyjny i automatyczny tuning hardware nie mogą blokować tych rezultatów. Zakres podstawowej wizualizacji obejmuje jednak 0–50 płytek i rzeczywistą telemetrię, zgodnie z wymaganiem.
===
# 31 Otwarte punkty i granice twierdzeń badawczych
| Punkt do rozstrzygnięcia | Właściciel | Co zależy od odpowiedzi |
| Wersja Pi Zero, kamera i kable | Wiktor | możliwość migracji i adapter kamery |
| Dokładny model Uno i działający firmware | Kacper i Andrzej | próbkowanie, liczba cech, transfer USB |
| Szerokość impulsu i parametry Lu.i | Andrzej i Patryk | zgodność modelu i eksportu |
| Finalny champion i podział continuous | Marcel, Kacper, Patryk | rzetelność końcowej oceny |
| Foundry model, quota i regiony | Wiktor | koszt, latency i sposób autoryzacji |
| Reguła alarmu i uzbrojenie | Wiktor | dopuszczenie automatycznego buzzera |
| Retencja i odbiorcy e-maila | Wiktor i zespół | konfiguracja przed uruchomieniem |
| Zakres danych drugiego zespołu | Andrzej | czy to pomiar komponentów czy całego detektora |

## Pytanie badawcze
Przy jakiej jakości detekcji, liczbie aktywacji i koszcie stałym bramka SNN zmniejsza energię całego badanego układu względem FFT? Odpowiedź może być negatywna. Wartością pracy jest uczciwy bilans i zakres opłacalności, a nie z góry założona przewaga neuromorficzności. Metryka recall przy FA/h nie jest nowym wynalazkiem zespołu.

Symulacja w Azure jest demonstratorem i narzędziem oceny, a nie dowodem bardzo niskiego poboru sprzętu neuromorficznego. Porównanie energii musi obejmować jednakowe granice albo jawnie opisywać wyłączone składniki. Dane z drugiego zespołu mają autora, konfigurację i metodę pomiaru; nie przedstawiamy ich jako własnej ewaluacji całego systemu.

## Konferencja
Strona ICTMOD podaje przedłużony termin zgłoszenia do 30 września 2026. Pozycjonowanie wokół kosztu, energii, decyzji i wdrożenia technologii jest bardziej zgodne z charakterem konferencji niż sam atrakcyjny dashboard. Nie ma podstaw do wiarygodnego procentowego oszacowania akceptacji konkretnego tekstu przed uzyskaniem wyników i recenzją pracy. [S9]

Przed wysłaniem trzeba ponownie sprawdzić aktualne instructions for authors i wybraną ścieżkę. Ten dokument jest specyfikacją projektu, nie gotowym artykułem ani zapewnieniem publikacji. Instrukcje i terminy ze starszych załączników nie zastępują aktualnej strony konferencji.
===
# 32 Źródła i pochodzenie ilustracji
## Dokumenty dostarczone przez zespół
[D1] propozycje_projektu_SNN_Lui.md — trzy kierunki prac; wybrane połączenie demonstratora i bilansu energetycznego.
[D2] 01_Propozycja_kiedy_SNN_sie_oplaca.pdf — 12 stron, pytania badawcze, eksperymenty i model energii; model halt zaktualizowano do stale aktywnego Pi.
[D3] Mindmapa_continuous_dataset_FAh.pdf oraz 05_Mindmapa_jedna_strona.pdf — kontekst metryk i tematu artykułu. Wcześniejsze liczby traktujemy jako wyniki historyczne o określonym pochodzeniu.
[D4] neuron.svg, lu.i-neuron-pcb-master oraz obrazy referencyjne; IMG_5774.PNG identyfikuje MAX4466. ESP32 z IMG_5776.PNG nie należy do przyjętej architektury.

## Referencje repozytorium
[R1] origin/feat/dashboard @ bb1dbe02; rpi_agents/agent/machine.py, camera.py, prefilter.py, requirements.txt i cloud/infra.
[R2] origin/feat/master-pipeline @ 6fc7be79; master_pipeline/ga_runner.py — final evaluation i hardware export.
[R3] origin/feat/continuous-dataset @ a009acec; dataset/continuous/README.md oraz eval/stream_builder.py i annotations.py.
[R4] origin/feat/encoder; encoder/fixed_encoder_08_06_26/fixed_encoder_08_06_26.ino; origin/feat/encoder-features, encoder/feature_bank/README.md — różne warianty i różne platformy.
[R5] origin/feat/testing-ideas; dataset/versions/v2.0.0/stats.md — liczności i opis audytu.

## Dokumentacja zewnętrzna sprawdzona 23 września 2026
[S1] Lu.i README i electronic interface: https://github.com/giantaxon/lu.i-neuron-pcb oraz https://github.com/giantaxon/lu.i-neuron-pcb/blob/master/doc/electronic-interface.md
[S2] Stradmann i in., Lu.i — A low-cost electronic neuron for education and outreach: https://arxiv.org/abs/2404.16664 ; publikacja wskazana w README: https://doi.org/10.1016/j.tine.2025.100248
[S3] Arduino UNO R3: https://docs.arduino.cc/hardware/uno-rev3/ ; datasheet: https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf
[S4] MAX4466: https://www.analog.com/en/products/max4466.html
[S5] Raspberry Pi: https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/ ; kamery: https://www.raspberrypi.com/documentation/accessories/camera.html
[S6] Container Apps scaling: https://learn.microsoft.com/en-us/azure/container-apps/scale-app
[S7] Container Apps billing: https://learn.microsoft.com/en-us/azure/container-apps/billing
[S8] Foundry Entra ID: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/configure-entra-id
[S9] ICTMOD 2026: https://ictmod-conference.com/

Referencje UI i wizualizacja stanowiska zostały wygenerowane na potrzeby projektu. Fotografia Lu.i pochodzi z dostarczonego repozytorium giantaxon, doc/figures/photograph-coin.jpg; zachowujemy przypisanie autorstwa i dokument LICENSE repozytorium. Ilustracje generowane nie są zdjęciami stanowiska zespołu ani schematem montażowym. Diagramy architektury są opracowaniem projektu i mają edytowalne źródła w pakiecie.
===
# 33 Pakiet wdrożeniowy i wizja stanowiska
@fig assets/stanowisko-koncepcja.png | Wizualizacja koncepcyjna stanowiska i oddzielnej sieci porównawczej Lu.i | 3.6

Ilustracja służy rozmowie o układzie demonstratora. Detale płytek i kabli są poglądowe; montaż wynika z zatwierdzonego pinoutu i dokumentacji posiadanego sprzętu. Ekran w tle nie zastępuje zatwierdzonej referencji dashboardu.

## Minimalny manifest championa
| Pole | Wymaganie |
| model_id, model_hash, schema_version | Jednoznaczna wersja i integralność pakietu |
| source_commit, training_seed | Pochodzenie kodu i przebiegu uczenia |
| dataset_manifest_hash, split_policy | Wersja danych i reguła niezależności |
| encoder_hash, channels, sample_rate | Zgodność wejścia z firmware i urządzeniem |
| dt_s, integrator, reset, refractory | Kompletna dynamika runtime |
| topology, weights, decoder_config | Sieć, parametry oraz reguła wyjścia |
| calibration_id, hardware_compatible | Status przenoszalności na Lu.i |
| evaluation_report_hash | Powiązanie z zamrożonym raportem |

Walidator odrzuca brakujące jednostki, odwołania do nieistniejących neuronów i niespójne kanały. Nie zakłada domyślnych wartości, które zmieniają znaczenie modelu. Pakiet zawiera także krótki wektor testowy: zapis impulsów oraz oczekiwane spike’y i decyzję. Wiktor uruchamia ten sam test po wdrożeniu w Azure i po ewentualnej zmianie Pi.
