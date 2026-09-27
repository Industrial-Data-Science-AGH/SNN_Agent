# Plan wykonania SNN Agent
## Zadania zespołu i integracja przez pull request
Wersja 1 • 24 września 2026 • plan na trzy dni pracy zespołu

Budujemy działający łańcuch Arduino Uno → Raspberry Pi 5 → symulacja Lu.i w Azure → zdjęcie → model vision w Foundry → decyzja i alarm. Raspberry Pi pozostaje lekkim mostem sprzętowym. Dashboard jest po angielsku, a dokumentacja zespołu po polsku.

Każda z sześciu wymienionych osób ma jeden główny pakiet odpowiedzialności. W jego obrębie znajdują się uporządkowane, odbieralne kroki. Osobny krok nie zawsze oznacza osobny PR: mały, spójny i zielony przyrost powinien trafiać do master możliwie szybko. Kacper kończy dwa już istniejące zakresy w dwóch istniejących PR; tworzenie ich kopii utrudniłoby review.

## Wykonane ustawienia repozytorium
| Ustawienie | Stan po tej sesji |
| master | Utworzony na c8c6eab18b5a93a4b9a8b7656ebf391cfac1ace2 z feat/testing-ideas |
| Domyślna gałąź | master; main pozostaje zachowany |
| Włączenie zmian | Wyłącznie PR; dopuszczony squash |
| Review | Minimum jedna aprobata, akceptacja ostatniego push przez inną osobę, wyczyszczone dyskusje |
| Ochrona historii | Force-push i usunięcie master zablokowane |
| Wyjątki administratorów | Brak bypass actors w aktywnym ruleset |
| Lokalny klon Wiktora | pre-commit, pre-merge-commit i pre-push blokują master |
| Kod źródłowy | Bez zmian; brak wykonanych merge i nowych commitów |

Reguła GitHub ma ID 23904463 i nazwę master PR only. Odczyt API potwierdził enforcement=active i protected=true. Utworzenie master wskazało istniejący commit; nie przeniosło do niego niezrecenzowanych zmian z innych branchy.

Plan implementacji, nowe branche osób i testy opisane dalej są pracą do wykonania. Nie zostały przedstawione jako gotowy system. Otwarte PR #47–#50 nie zostały w tej sesji scalone ani retargetowane.

===
# 1 Dlaczego master z testing ideas
Wybrana baza feat/testing-ideas @ c8c6eab1 zawiera już PR #44 z poprawkami rozdzielenia walidacji i testu, wyboru checkpointu i raportowania. To najlepszy punkt wspólny nowych prac, ale nie deklaracja, że całe repo jest produkcyjnie gotowe albo wszystkie testy zostały teraz uruchomione.

main @ 00c9d582 jest znacznie starszy. W porównaniu historii z wybraną bazą ma dwa własne commity, a brakuje mu 80 commitów z drugiej strony. Jego własna zmiana od wspólnego przodka dotyczy build_combined_dataset.py; plik istnieje także w rozwiniętej bazie. Nie nadpisujemy nowszego pipeline’u starym plikiem. dev ma 106 własnych commitów i brakuje mu 85 commitów bazy, więc nie jest czystą integracją aktualnych kierunków.

Liczby poniżej oznaczają: „brak” to commity obecne w bazie, których branch nie zawiera; „własne” to commity brancha nieobecne w bazie. Nie mierzą jakości ani liczby konfliktów. Dla continuous-dataset oznacza to konieczność aktualizacji, nie odrzucenie jego ośmiu commitów.

| Branch | SHA | Brak / własne | Decyzja |
| feat/testing-ideas | c8c6eab1 | 0 / 0 | Baza master, dalsza integracja przez master |
| feat/master-pipeline | 6fc7be79 | 0 / 24 | Marcel kontynuuje, PR #47 po poprawkach |
| feat/continuous-dataset | a009acec | 30 / 8 | Kacper kontynuuje, aktualizacja bazy i PR #48 |
| feat/encoder-features | c9ff2346 | 0 / 3 | Kacper kontynuuje, PR #50 |
| feat/dashboard | bb1dbe02 | 80 / 22 | Źródło wybranych modułów; nowy branch dla nowego UI |
| feat/rpi | 9c329a9f | 80 / 14 | Starszy agent; preferować nowszy snapshot dashboard |
| feat/encoder | 3bf5b300 | 85 / 105 | Źródło porównania firmware, bez pełnego merge |
| fah-metric-kn | e09456a8 | 0 / 3 | PR #49: tylko po audycie wyników i zakresu |
| feat/training_N8 | e36bc467 | 64 / 3 | Historyczne wyniki; nie scalać PR #23 automatycznie |

Istotne różnice: master-pipeline dodaje 8 plików i 983 linie, ale final evaluation używa testu jako walidacji, a eksport potrafi trenować ponownie. Continuous dodaje 11 plików i 1745 linii; filtr tła wymaga niezależności także od val. Encoder-features zmienia 21 plików, w tym poboczny train_log.csv. Te problemy mają konkretne zadania M2, K3 i K5.

===
# 2 Pozostałe branche i zakres analizy
Sprawdzono listę branchy i SHA bezpośrednio na GitHub, a lokalnie ich graf, rozbieżności oraz zakresy plików. Aktualne zdalne SHA pokrywały się z lokalnymi referencjami. Nie wykonywano treningu ani pełnego audytu każdego historycznego eksperymentu.

| Branch | Brak / własne | Postępowanie |
| dataset/proposed-dataset | 85 / 35 | Historyczny builder; nie importować całego brancha. |
| dev | 85 / 106 | Stara integracja; nie używać jako nowej bazy. |
| docs | 86 / 1 | Starszy opis; źródło kontekstu. |
| feat/ci-cd-pipeline | 85 / 65 | Stary pipeline; przegląd pomysłów, bez pełnego merge. |
| feat/dataset-contract-v1 | 62 / 0 | Przodek bazy; brak własnych commitów do integracji. |
| feat/dataset-remove-atrifacts | 31 / 0 | Przodek bazy; nie scalać ponownie. |
| feat/decoder | 85 / 5 | Stary prototyp; źródło porównania dekodera. |
| feat/rebuild-clean-benchmark | 56 / 0 | Przodek bazy; zachować pochodzenie. |
| feat/voice-label-fix-dataset-contract-v2 | 61 / 0 | Poprawka etykiet już w historii bazy. |
| hat-metrics-work | 81 / 2 | Historyczna gałąź metryk; bez automatycznego importu. |
| integration/ci-cd-test | 80 / 31 | Eksperyment integracyjny, nie baza release. |
| main | 80 / 2 | Poprzednia gałąź domyślna; pozostaje zachowana. |
| snn/ready-for-test-pipeline | 85 / 54 | Starszy zrzut pipeline’u; nie baza nowych prac. |
| snn_marcel_meeting | 85 / 45 | Starszy eksperyment; tylko udokumentowane referencje. |
| worktree-dataset-expansion | 80 / 1 | Historyczny builder danych, zachować jako referencję. |

Nie usuwamy historycznych branchy ani datasetów. Brak konfliktów według GitHub nie dowodzi poprawności naukowej. Przy późniejszym sprzątaniu najpierw zachowujemy mapę SHA i pochodzenie wyników. Pełny snapshot liczb znajduje się w analiza_branchy.json.

Nazwa rpi-agents z rozmowy opisuje obszar kodu. Na zdalnym repo nie ma obecnie brancha o tej nazwie; katalog rpi_agents jest na feat/rpi i rozwiniętym feat/dashboard.

===
# 3 Branche zespołu i kolejność integracji
| Osoba | Główny zakres | Branch |
| Wiktor | Edge, API, Foundry, Azure, auth, integracja | nowy feat/wiktor-edge-cloud |
| Patryk | Lu.i runtime, stan, telemetria | nowy feat/patryk-lui-runtime |
| Karolina | Angielski dashboard i formularz loginu | nowy feat/karolina-dashboard |
| Marcel | Trening, ocena, niezmienny champion | istniejący feat/master-pipeline |
| Kacper | Enkoder oraz builder continuous | istniejące feat/encoder-features i feat/continuous-dataset |
| Andrzej | Pomiary, kalibracja i instrukcje hardware | nowy feat/andrzej-hardware-energy |

Nowe branche są zaplanowane, nie zostały jeszcze utworzone. Powstają z aktualnego origin/master po pobraniu zmian. Najpierw Wiktor dostarcza mały PR kontraktów i szkieletu; reszta może projektować na jego fixture, ale przed integracją pobiera już zatwierdzony master. Nie tworzymy sześciu kopii całego starego dashboardu.

Kacper przygotowuje #48 i #50, Marcel #47. Ich base należy zmienić z feat/testing-ideas na master po sprawdzeniu diffu i zależności. #49 pozostaje osobnym przeglądem historycznych metryk; #23 nie jest automatycznym kandydatem do integracji, ponieważ poprawki treningu trafiły już innymi commitami do bazy. Nie zamykamy cudzych PR bez sprawdzenia ich rzeczywiście unikalnego wkładu.

## Kolejność pierwszych PR
1. W0: kontrakty, fixture i root workflow pr-gate. Jeden reviewer, lokalne dowody testów; po pierwszym udanym checku dodać go jako required status check.
2. K1/K2: zgodny enkoder i format serial; K3: niezależny continuous dataset, w osobnym PR. Dane nie blokują budowania UI na fixture.
3. P1/P2: runtime; W1/W2: bridge i backend; C1/C2: UI. Te przyrosty mogą wejść w różnej kolejności, jeśli kontrakty i testy są spełnione.
4. M2–M5: poprawny eksport oraz champion; P3–P5 i C3/C4: telemetria, replay i metryki.
5. W3/W4 i C5: prawdziwe vision, dostęp i deployment; W5/A5: finalny odbiór i manifest release.

Po squash PR branch zadania jest zakończony. Następny przyrost danej osoby powstaje z aktualnego master, np. feat/wiktor-edge-cloud-02. Nie kontynuujemy serii commitów na branchu, którego wcześniejsza historia została squashed — to powoduje powracające diffy i utrudnia review.

===
# 4 Proces pull request i egzekwowanie reguł
## Codzienny przebieg pracy
1. Pobrać origin/master i utworzyć branch zadania. Nigdy nie implementować na lokalnym master. W commicie opisać zachowanie, nie ogólne „update”.
2. Przed otwarciem PR uruchomić testy swojego zakresu i wspólne testy kontraktów. Testy naukowe z brakującym datasetem mają być jawnie skipped/not run, nie przedstawiane jako pełny PASS.
3. Otworzyć Draft PR do master z zakresem, właścicielem, zależnościami, dowodami testów i wpływem na konfigurację. Bez sekretów, pełnych logów API i prywatnych zdjęć.
4. Zaktualizować feature o master, rozwiązać konflikty na feature, ponownie przetestować. Na branchach współdzielonych dozwolony jest merge master → feature. Zakaz dotyczy bezpośredniego feature → master oraz push do master.
5. Zmienić PR na Ready i poprosić wskazanego reviewera. Autor nie zatwierdza własnego PR. Osoba, która zrobiła ostatni push, nie jest jedyną akceptującą tę zmianę.
6. Po akceptacji i zamknięciu dyskusji wybrać Squash and merge w GitHub. Każdy nowy push wymagający review unieważnia stare akceptacje. Nie używać --admin do obejścia reguł.
7. Pobrać master, wykonać smoke test integracji, opisać release SHA. Kolejny przyrost zaczynać z nowego brancha. Błąd na master naprawiać przez fix PR lub revert PR, nigdy reset/force-push.

## Co jest rzeczywiście włączone
Ruleset master PR only działa na refs/heads/master, ma brak bypassów, blokadę usunięcia i non-fast-forward oraz wymaganie PR z jedną akceptacją, akceptacją ostatniego push i rozwiązanymi rozmowami. Dopuszcza wyłącznie squash. Reguła obejmuje też administratorów, o ile pozostaje aktywna. Administrator uprawniony do edycji reguł nadal może je zmienić — repozytorium nie potrafi odebrać właścicielowi organizacji tej kontroli.

Nie dodano fikcyjnych required checks: w bazie nie ma działającego nowego pr-gate. To jawny kolejny krok W0. Do jego wdrożenia ochrona wymusza PR i review, lecz nie egzekwuje automatycznie przejścia testów. Nie używamy reguły Restrict updates z pustą listą wyjątków, bo zablokowałaby także dozwolone scalanie PR.

Lokalne hooki zostały zainstalowane i sprawdzone na pomocniczym repo. Commit i merge na master oraz push do refs/heads/master kończą się kodem 1; feature przechodzi. Hooki nie kopiują się przy clone i można je obejść przez --no-verify, dlatego nie zastępują GitHub. Pakiet zawiera instalator dla pozostałych osób, który nie nadpisuje istniejących hooków.

===
# 5 Jeden login i hasło do portalu
Decyzja użytkownika zastępuje organizacyjne logowanie i role viewer/operator z rozdziału 28 pierwotnej architektury. Portal ma jedną wspólną parę danych i jednakowe uprawnienia operatora dla zespołu. Nie budujemy rejestracji, zaproszeń, zarządzania kontami ani odzyskiwania hasła. Wiktor zmienia wspólne hasło poza UI.

Karolina przygotowuje angielski formularz Username, Password, Sign in, komunikat Invalid credentials i Logout. Wiktor implementuje POST /auth/login, POST /auth/logout oraz GET /auth/session. Login weryfikowany na serwerze zakłada sesję; na kliencie nie przechowujemy hasła. Konto współdzielone nie daje audytu osobowego: zapis actor=shared_operator oznacza konto, nie wskazanego członka zespołu.

| Element | Kontrakt wdrożenia |
| Dane logowania | Wiktor ustawia sam; sekret lub hash w Key Vault, brak domyślnych danych |
| Hasło weryfikowane lokalnie | Gotowa biblioteka hashująca hasła, np. Argon2id; nie własny algorytm |
| Sesja | Losowy nieprzewidywalny identyfikator; po stronie serwera hash ID, czas ważności i wersja credentials |
| Cookie | __Host-snn_session; Secure, HttpOnly, SameSite=Strict, Path=/, bez Domain |
| Limit czasu | Proponowane 8 h bez przedłużania w nieskończoność; logout unieważnia rekord |
| Zmiana hasła | Zwiększenie credentials_version i unieważnienie wszystkich starych sesji |
| CSRF | Sprawdzany token dla mutacji i zgodny Origin; samo SameSite nie wystarcza |
| Brute force | Proponowane 5 nieudanych prób/min/IP i ograniczenie łączne; 429 + Retry-After |
| UI i fotografie | Backend sprawdza sesję dla danych, zdjęć i streamu, nie tylko strony HTML |
| Raspberry Pi | Oddzielne poświadczenie urządzenia i zakres tylko jego sesji |

Sesje można utrzymać w Table Storage z warunkową aktualizacją. Nie dodajemy nowego stale działającego serwera tylko na logowanie. Rate limiter pamięciowy jest uproszczeniem wyłącznie na demonstrator z jedną repliką; restart i nakładanie rewizji osłabiają taki limit, co trzeba opisać i sprawdzić. Docelowo użyć współdzielonego licznika albo warstwy ingress.

Starszy auth.py na feat/dashboard ma HTTP Basic i domyślne ids/ids. Nie wolno bez zmian publikować tego ustawienia. Test brakującego sekretu ma blokować dostęp. Model vision nigdy nie otrzymuje danych logowania, a Wiktor nie wkleja ich do promptu ani dokumentu. Zalecenia cookies i sesji oparto na OWASP, źródła na końcu.

===
# 6 Kolejność trzech dni
@fig assets/kolejnosc.png | Zależności pracy i punkt integracji | 5.25

D1 oznacza pierwszy wspólny dzień pracy po przyjęciu planu, nie sztywną datę kalendarzową. Termin trzech dni jest bardzo napięty; plan organizuje równoległą pracę, nie gwarantuje uzyskania docelowego recall/FA/h ani pełnej zgodności fizycznego twin.

Koniec D1: kontrakty, wybrany enkoder, inwentaryzacja, mock UI, testowy runtime i eksport bez ponownego treningu. Koniec D2: pierwszy prawdziwy E2E na Pi 5, UI live, ciągły dataset i pierwsze pomiary. D3: poprawki integracji, zamrożona ocena, porównanie hardware, demonstracja oraz manifest release.

Nie odkładamy pierwszego E2E do dnia trzeciego. Tymczasowy model testowy może służyć integracji wyłącznie z etykietą demo; nie staje się wynikiem eksperymentu ani championem przez sam fakt wdrożenia.

===
# 7 Własność plików i przekazania
Wspólne kontrakty są pierwszym punktem integracji. Wiktor utrzymuje API, storage, auth i deployment; Patryk dostarcza moduł runtime wywoływany przez to API. Karolina utrzymuje szablony i zasoby UI. Dzięki temu trzy osoby nie budują równolegle różnych serwerów ani różnych formatów zdarzeń.

| Właściciel | Główny zakres plików |
| Wiktor | rpi_agents/agent, cloud/app backend, cloud/infra, deploy, root .github/workflows, contracts |
| Patryk | nowy snn_runtime, tests/runtime; model i neuron schema wspólnie z Wiktorem |
| Karolina | rpi_agents/cloud/app/templates, static, routes_dashboard; tests/dashboard |
| Marcel | master_pipeline; niezbędne poprawki ga_neuron_search; tests/pipeline |
| Kacper | encoder/features-improvement; dataset/continuous; testy enkodera i Uno baseline |
| Andrzej | nowe hardware/calibration, measurements, runbooks |

Wspólne pliki konfiguracyjne, zależności i root CI zmienia jeden właściciel — Wiktor — po uzgodnieniu potrzeb. Patryk zgłasza zależności runtime, Karolina zasoby frontendu. Podział jest proponowaną organizacją nowej implementacji; nowe katalogi nie są przedstawiane jako istniejące już moduły.

## Warunek przekazania kroku
Przekazanie zawiera: commit lub PR, kontrakt/schema_version, uruchomiony test z wynikiem, fixture lub raport i znane ograniczenia. „Działa u mnie” nie jest odbiorem. Zadanie blokowane przez brak urządzenia raportuje dokładnie brakujący pomiar; niezależna praca na fixture może trwać dalej.

## Reviewerzy
Wiktor ↔ Patryk: API, edge i runtime. Karolina ↔ Wiktor: UI i integracja. Marcel ↔ Patryk: model i eksport. Kacper ↔ Marcel: continuous i metryki; Kacper ↔ Wiktor/Andrzej: firmware. Andrzej ↔ Patryk: kalibracja. Wymagane review GitHub to minimum jedna osoba; druga konsultacja domenowa jest potrzebna tam, gdzie jedna osoba nie potrafi ocenić pomiaru lub modelu.

Nowe hooki są lokalnym zabezpieczeniem, nie modyfikacją kodu projektu. Kod aplikacji, wspólny login i CI pozostają zadaniami do wdrożenia. Dokładne checklisty poniżej oraz osobne pliki taski/T01–T06 stanowią te same pakiety pracy.

===
# T01 Wiktor krok 1 do 2
Lekki edge, chmura, wspólny dostęp i integracja end to end. Branch: feat/wiktor-edge-cloud.

## W0 Kontrakty i podstawa integracji
D1 rano. Zależności: brak — można rozpocząć od razu.

1. Z Patrykiem i Kacprem zamrozić schema_version, channel_map, jednostki, zegar i encoder_hash. Zapisać kontrakty SpikeBatch, SNNDecision, NeuronFrame, CaptureCommand, VisionResult, AlarmCommand oraz manifest modelu.

2. W osobnym krótkim PR utworzyć contracts/ z JSON Schema i minimalnymi fixture: cisza, spike, duplikat, luka, trigger, vision unavailable. Dodać mock API do pracy bez urządzeń.

3. Przenieść wybrane rpi_agents z feat/dashboard do nowego brancha. Nie kopiować .env, wyników, starych ADR jako obowiązującego projektu ani całej historii dev. Zapisać źródłowy SHA.

4. Dodać root .github/workflows/pr.yml: pull_request na master, read-only token, testy kontraktów i lekkie testy modułów, bez sekretów i bez kosztownych treningów. Dopiero po udanym uruchomieniu ustawić wymagany check pr-gate.

**Odbiór:** Pierwszy PR pokazuje walidację poprawnych fixture i odrzucenie błędnego kanału/wersji. Workflow rzeczywiście uruchamia się z głównego .github/workflows, a nie z rpi_agents/.github/workflows.

**Przekazanie:** Patryk i Karolina dostają fixtures oraz stabilne nazwy endpointów; wszyscy pobierają master po squash PR.

## W1 Bridge Uno i Raspberry Pi 5
D1. Zależności: W0, K1.

1. Z Andrzejem sprawdzić urządzenia USB, mikrofon i wyjścia; zapisać adapter kamery i serial device w konfiguracji.

2. Z Kacprem odebrać ramki Uno. Walidować seq, boot_id, zakresy i długość; bufor ograniczony, jawny gap zamiast cichej utraty. Batching z czasem źródłowym, nie czasem nadejścia HTTP.

3. Zaimplementować lekki proces systemd: reconnect USB/HTTPS, trwały mały outbox, heartbeat i obsługę SIGTERM. Nie instalować torch ani lokalnego vision.

4. Oddzielić CameraAdapter, SerialAdapter i AlarmAdapter od logiki. Capture wykonać dopiero po ważnej komendzie SNN; czas TTL i deduplikacja po command_id.

**Odbiór:** Na Pi 5 działa serial replay i fizyczne Uno; ponowne podłączenie nie gubi tożsamości sesji. Powtórzona komenda nie robi kolejnych zdjęć.

**Przekazanie:** Patryk otrzymuje zapis wejścia; Karolina otrzymuje DeviceStatus; ten sam program ma działać na przyszłym Pi Zero po zmianie konfiguracji.


===
# T01 Wiktor krok 3 do 4
## W2 Trwałe zdarzenia i polecenia
D1–D2. Zależności: W0.

1. Przygotować API urządzenia, session/event_id, własność obiektów i idempotentne zapisy do Table/Blob. Zdefiniować limity JPEG i batchy.

2. Zapis zdarzenia i outbox powiązać jednym planem odzyskiwania; publikować zadania do Queue. Worker deduplikuje, odnawia visibility timeout i kończy retry w kolejce błędów.

3. Outbox skanować przy uruchomieniu i okresowo podczas pracy; przed scale-to-zero potwierdzić opróżnienie. Przy awarii pomiędzy zapisem a enqueue potrzebny jest niezależny okresowy reconciler albo jawne utrzymywanie minReplicas=1 aż do odrobienia zaległości.

4. Komenda alarmu zawiera TTL, event_id i command_id. Pi potwierdza applied/expired/failed, wyjścia wyłączają się po lokalnym limicie niezależnie od chmury.

**Odbiór:** Test crash-before-enqueue kończy się odtworzeniem zadania; duplikat nie tworzy dwóch zdarzeń ani ponownego buzzera; obce device_id jest odrzucone.

**Przekazanie:** Karolina otrzymuje listę zdarzeń i statusy ACK; W3 dostaje kolejkę vision.

## W3 Foundry i reguła alarmu
D2. Zależności: W1, W2, P2.

1. Wybrać dostępny na subskrypcji model vision, zapisać model/deployment/version i sprawdzić poprawny JPEG oraz timeout. Dostęp z managed identity, jeśli wdrożenie go wspiera.

2. Model zwraca wyłącznie obserwacje zgodne ze schematem: glass_visible, person_visible, uncertainty. Nie nadaje osobie statusu unauthorized bez osobnej informacji.

3. Politykę alarmu trzymać w kodzie. Do czasu rozstrzygnięcia uzbrojenia/nieuprawnionej osoby: Review required. SNN=true samo nie uruchamia alarmu.

4. Przenieść e-mail do workera; adresaci z konfiguracji, załącznik wskazuje event_id. Retry nie może obiecywać exactly-once e-mail przy niejednoznacznym timeout.

**Odbiór:** Cztery scenariusze glass/person oraz błąd vision dają zapisane statusy. Błąd API nigdy nie jest zamieniany w pewne włamanie. Alarm spełnia zatwierdzoną regułę i gaśnie po limicie.

**Przekazanie:** Karolina otrzymuje realne wyniki i obraz do Event details; Andrzej rejestruje pobór w fazie capture/vision/alarm.


===
# T01 Wiktor krok 5 do 6
## W4 Azure i wspólny login
D2. Zależności: W0, C1.

1. Sprawdzić dostępne regiony az CLI, modele Foundry i quota. Zbudować kontenery API/SNN/UI oraz workera; utworzyć Blob/Table/Queue i Key Vault.

2. Dodać formularz Username/Password dla jednej wspólnej pary, sesję serwerową i Logout. Brak rejestracji, ról i kont per osoba. Hasło lub jego weryfikator tylko w Key Vault; brak domyślnych danych logowania.

3. Włączyć HTTPS, Secure/HttpOnly/SameSite cookie, ochronę CSRF, TTL sesji, ograniczenie prób logowania oraz redakcję logów. Pi używa osobnego poświadczenia urządzenia.

4. Na test aktywować minReplicas=1 dla runtime SNN. Po Stop i drain wrócić do min0. Jedna aktywna epoka i lease również podczas zmiany rewizji; bez obietnicy zerowego kosztu całej subskrypcji.

**Odbiór:** Login działa z dwóch przeglądarek tą samą parą, bezpieczny logout unieważnia daną sesję, nieautoryzowany odczyt zdjęć/telemetrii jest blokowany. Po restarcie sekret nie jest drukowany ani dodany do obrazu.

**Przekazanie:** Zespół dostaje URL; hasło Wiktor przekazuje własnym kanałem, poza rozmową z LLM. Karolina podłącza ekran loginu.

## W5 Odbiór i wersja demonstracyjna
D3. Zależności: W3, W4, P5, C5, M5, K2, A3.

1. Uruchomić skryptowany test E2E: Uno → Pi → SNN → capture → Foundry → policy → email/LED/buzzer. Zachować event_id oraz czasy wszystkich faz.

2. Sprawdzić przerwanie Wi-Fi, restart kontenera, odłączenie USB, timeout vision, duplikat i TTL. Awaria ma widoczny stan, bez fikcyjnej ciągłości.

3. Zamrozić release po zatwierdzonych PR, zapisać SHA obrazu/modelu/enkodera i parametry sesji. Nie poprawiać kodu bezpośrednio na urządzeniu bez przeniesienia przez PR.

4. Przygotować checklistę migracji: model Zero, architektura CPU, OS, USB/OTG, zasilanie, kamera i instalacja zależności. Decyzja o migracji dopiero po identyfikacji sprzętu.

**Odbiór:** Demonstracja powtarza się z czystego wdrożenia; awarie mają ślad i dają się odtworzyć. Raport rozdziela wyniki rzeczywiste od replay.

**Przekazanie:** Cały zespół: instrukcja uruchomienia, rollback obrazu/modelu, manifest release i lista ograniczeń.


===
# T02 Patryk krok 1 do 2
Stanowy symulator Lu.i i kontrakt cyfrowego bliźniaka. Branch: feat/patryk-lui-runtime.

## P1 Zdefiniować model i wejście runtime
D1 rano. Zależności: W0, K1.

1. Wybrać jeden adapter pakietu modelu; wymagać topology, weights, dt, reset, refractory, tau i decoder, bez ukrytych domyślnych wartości.

2. Zmapować kanały wejściowe na porty neuronów, odróżnić trzy fizyczne wejścia Lu.i od liczby cech enkodera.

3. Opisać jednostki napięcia/czasu i przejście z parametrów treningu na model referencyjny. Pozostawić oznaczenie uncalibrated do pomiarów Andrzeja.

**Odbiór:** Niekompletny model, niezgodny encoder_hash i błędny port kończą się kontrolowanym błędem przed Start.

**Przekazanie:** Marcel zna format eksportu; Karolina zna ID neuronów i pól.

## P2 Stanowy streaming i dekoder
D1–D2. Zależności: P1.

1. Wydzielić Runtime.load/reset/step/checkpoint bez zależności od UI. Utrzymywać Vmem, prądy synaps i historię dekodera pomiędzy batchami.

2. Zaimplementować kolejność po seq i source time, duplikaty, jawne luki oraz session_epoch. Nie resetować stanu na każde HTTP.

3. Dostarczyć Wiktorowi adapter route SNN i wynik zawierający decision_id, source_time, model_hash. Nie tworzyć równoległego drugiego API.

**Odbiór:** Ten sam strumień podzielony na różne batch sizes daje zgodne spike’y i decyzje w ustalonej tolerancji; luka generuje gap/warmup.

**Przekazanie:** Wiktor może podłączyć prawdziwy bridge; Karolina może odtworzyć krótką sesję.


===
# T02 Patryk krok 3 do 5
## P3 Telemetria i edytowalna topologia
D2. Zależności: P2.

1. Emitować NeuronFrame z Vmem, progami, spike timestamp, stanem synaps, mode i calibration_id. Określić downsampling wykresu niezależnie od czasu symulacji.

2. Zapewnić snapshot i zdarzenia z rosnącym event sequence do live/replay. Telemetria UI może pomijać klatki, lecz nie ukrywać luk w danych wejściowych.

3. Walidować draft topologii 0–50: 0 oznacza pusty edytor, nie działający klasyfikator. Zmiana liczby neuronów nie przebudowuje po cichu championa; nowa sesja wymaga nowego ważnego modelu.

**Odbiór:** Karolina widzi wierny raster i potencjał na znanym wektorze testowym; Pause view nie zatrzymuje backendu.

**Przekazanie:** Udokumentowane mapowanie LED oraz przykład 8 i 50 neuronów.

## P4 Porównać z fizycznym Lu.i
D2–D3. Zależności: P2, A2.

1. Odtworzyć wejściowe impulsy pomiarów Andrzeja z amplitudą i szerokością; uwzględnić ich znaczenie dla wagi.

2. Dopasować parametry na zbiorze kalibracyjnym i porównać na oddzielnym przebiegu: czasy spike, przebieg Vmem, nasycenie i reset.

3. Raportować błąd oraz zakres ważności; bez pomiarów oznaczyć model functional simulation, nie zweryfikowany twin 1:1.

**Odbiór:** Raport zawiera identyfikatory płytek, nastawy, warunki i tolerancje ustalone przed oceną; błąd nie jest ukryty średnią.

**Przekazanie:** Karolina wyświetla calibration status; Andrzej potwierdza zgodność warunków.

## P5 Zamrożony runtime i golden replay
D3. Zależności: P3, M5.

1. Wczytać dokładny checkpoint Marcela bez dodatkowego treningu i porównać decoder output z eksportem.

2. Zapisać krótki golden replay z przewidywanym rastrem, decyzją i tolerancją. Uruchomić go lokalnie i w kontenerze Azure.

3. Przekazać lock zależności i wynik testu reset/restart/duplicate; trening nie może działać w runtime request path.

**Odbiór:** Hash modelu pozostaje taki sam od oceny do Azure. Golden replay przechodzi w obu środowiskach.

**Przekazanie:** Wiktor dostaje kontrakt uruchomienia; zespół jeden identyfikator championa.


===
# T03 Karolina krok 1 do 2
Angielski dashboard z płytkami Lu.i i pomiarami. Branch: feat/karolina-dashboard.

## C1 Szkielet i ekran logowania
D1. Zależności: W0.

1. Przygotować ciemny shell: Network, Events, Experiments, Energy, Device. Wszystkie etykiety, błędy i tooltipy po angielsku.

2. Zbudować Username, Password, Sign in oraz Logout. Formularz korzysta z API Wiktora, nie przechowuje hasła w JS/localStorage i nie zawiera prawdziwych danych w mockach.

3. Wczytać fixture kontraktów i oznaczyć demo data. Oddzielić komponenty danych od renderowania, żeby realny backend zastąpił fixture bez przepisywania UI.

**Odbiór:** Widoki są dostępne bez sprzętu jako jawny tryb demo; formularz ma focus klawiatury, loading i Invalid credentials.

**Przekazanie:** Wiktor dostaje listę route/asset i kontrakt formularza; Patryk widzi format danych wymagany przez komponenty.

## C2 Płytki i edytor sieci
D1–D2. Zależności: C1.

1. Użyć neuron.svg jako źródła kształtu, osobne warstwy portów, LED i zaznaczenia. Nie rysować płytki jako zwykłego koła.

2. Dodać Board count 0–50, automatyczny layout, przeciąganie, zoom/pan/Fit, wybór neuronu i widok połączeń pobudzających/hamujących.

3. Rozdzielić edycję draft, załadowany model i uruchomioną sesję. Import/Export JSON nie może wykonywać kodu ani nadpisywać aktywnego championa.

**Odbiór:** Sprawdzone 0, 1, 8 i 50 płytek; puste płótno jest czytelne, etykiety nie nachodzą, połączenia zachowują ID po zmianie layoutu.

**Przekazanie:** Patryk dostaje payload draft topologii; zespół odnosi wygląd do referencji v3.


===
# T03 Karolina krok 3 do 5
## C3 Potencjał LED raster i replay
D2. Zależności: C2, P3.

1. Implementować LED potencjału i osobny błysk spike na podstawie pól runtime, nie losowej animacji lub samego tau.

2. Dodać inspektor neuronu: Signals, Parameters, Connections, Notes; wykres Vmem z progiem, jednostki i calibration status.

3. Obsłużyć Live/Replay, Pause view/Resume, stale data oraz reconnect. Skok replay odtwarza spójny stan, nie tylko przewija ekran.

**Odbiór:** Z golden fixture LED i raster wskazują ten sam neuron/czas; utrata połączenia daje Stale data; Pause view nie wysyła Stop.

**Przekazanie:** Patryk zatwierdza mapowanie sygnałów; Wiktor sprawdza obciążenie i reconnect.

## C4 Zdarzenia metryki i energia
D2–D3. Zależności: C1, W2.

1. Event details: zdjęcie, SNN/capture/vision/alarm timeline, osobne glass/person/authorization, ACK i error state. Brak danych pokazuje Not available.

2. Experiments: split, godziny tła, seed, model_hash, encoder_hash, FA/h z przedziałem ufności i recall; oddzielić metryki SNN oraz całego systemu.

3. Energy: źródło measured/estimated, granica pomiaru, moc W i energia J/Wh; Device: heartbeat, bufor, gap, kamera, serial. Nie wpisywać przykładowych liczb w realny tryb.

**Odbiór:** Dane przykładowe są oznaczone; brak pomiaru nie staje się zerem. Filtr wyniku nie miesza różnych modeli, datasetów ani granic energii.

**Przekazanie:** Marcel i Andrzej sprawdzają znaczenie swoich metryk; Wiktor podłącza API.

## C5 Odbiór UI na realnych danych
D3. Zależności: C3, C4, W4.

1. Przejść login/logout, jedną sesję live i replay, błąd vision, utratę łącza, puste dane i próbę zmiany aktywnego modelu.

2. Przetestować desktop oraz węższy ekran, nawigację klawiaturą i 50 płytek. Renderowanie musi mieć ograniczony bufor i liczbę punktów wykresu.

3. Zrobić zrzuty referencyjne z jawnego demo i z jednej oznaczonej realnej sesji. Przekazać PR ze zdjęciami before/after i listą testów.

**Odbiór:** Brak polskich etykiet produktu, wiszących spinnerów i wymyślonych danych; serwer blokuje dane po logout niezależnie od ukrycia UI.

**Przekazanie:** Wiktor otrzymuje gotowe zasoby do tego samego kontenera; dokumentacja ma wersję UI i SHA.


===
# T04 Marcel krok 1 do 2
Pipeline badań i niezmienny champion na MacBooku M5 Max 128 GB. Branch: feat/master-pipeline — kontynuacja PR #47.

## M0 Profil treningu M5 Max 128 GB
D1 rano. Zależności: brak — można rozpocząć od razu.

1. Potwierdzić ARM64 bez Rosetty, model CPU/GPU, macOS, Python i torch; sprawdzić is_built/is_available MPS. Nie przebudowywać środowiska trwającego treningu. Utworzyć odrębny lock i profil macbook_m5_max_128gb.

2. Naprawić benchmark_workers: każda konfiguracja ocenia ten sam zestaw rzeczywistych genomów, epok, próbek i seedów. Obecne tasks_per_worker mnoży ilość pracy przez liczbę workerów, a dummy matmul nie reprezentuje SNN.

3. Zmierzyć CPU GA z 1/2/4/8/12/16 workerami w granicach wykrytej maszyny i jednym wątkiem torch na worker. MPS przetestować osobno z jednym procesem treningowym, dla batch 128/256/512. Wybrać per etap na podstawie kandydatów/min, czasu epoki, jakości i pamięci.

4. Raportować RSS, memory pressure, swap, pamięć MPS i czas po synchronizacji GPU; zacząć od budżetu procesu około 80 GB jako ustawienia ostrożnego, nie fizycznego limitu. Bez wyłączania limitu MPS. Zapisać checkpoint/resume i porównanie CPU/MPS na identycznym wektorze.

**Odbiór:** Benchmark na rzeczywistym M5 Max 128 GB wybiera profil; manifest podaje sprzęt, wersje i parametry. Samo auto=cpu lub wymuszenie mps nie jest odbiorem optymalizacji.

**Przekazanie:** Patryk dostaje wersję torch i test zgodności; Wiktor uruchamia właściwy profil lokalnie. Azure wykonuje inferencję i obsługę aplikacji, a nie ten trening.

## M1 Ustalić protokół eksperymentu
D1 rano. Zależności: K1.

1. Zamrozić train/val/test, wersję enkodera i cel wyboru modelu na walidacji. Zachować lineage source/group_id.

2. Sprawdzić że żaden checkpoint, seed ani próg nie jest wybrany po końcowym teście. Wyniki historyczne oznaczyć osobno.

3. Zapisać macierz porównań SNN symulowane, Lu.i fizyczne i Uno FFT z identycznymi strumieniami oraz oddzielnymi granicami energii.

**Odbiór:** Konfiguracja i manifest pozwalają odtworzyć selekcję; test nie trafia do funkcji fitness ani strojenia progu.

**Przekazanie:** Kacper otrzymuje protokół budowy continuous; Andrzej listę przebiegów pomiarowych.


===
# T04 Marcel krok 3 do 4
## M2 Naprawić final evaluation i eksport
D1. Zależności: M1.

1. W ga_runner.py usunąć podstawianie testu do val_data w run_final_evaluation_stage i etykietę canonical_test_placeholder.

2. Rozdzielić trening, wybór, eksport i eval. Eksport ma serializować wybrany checkpoint, nie uruchamiać train_winner jeszcze raz.

3. Zachować jednocześnie source_commit, seed, dataset_manifest_hash, encoder_hash, checkpoint_hash, decoder i jednostki. Test sprawdza niezmienność wag po eksporcie.

4. Sprawdzić przekazywanie config.train.fitness_seeds: obecny run_ga_stage ma literal fitness_seeds=3. Test ma dowodzić, że profil sprzętowy i konfiguracja eksperymentu nie są ignorowane.

**Odbiór:** Wywołanie eval/export nie zmienia parametrów ani czasu uczenia; test wykrywa próbę użycia testu do wyboru modelu.

**Przekazanie:** Patryk dostaje pierwszy prawdziwy model do runtime bez czekania na najlepszy wynik GA.

## M3 Uruchomić kontrolowany trening
D1–D2. Zależności: M0, M2.

1. Wykorzystać trwające GA tylko po potwierdzeniu zgodnego kodu, splitów i enkodera. Nie restartować kosztownego treningu bez potrzeby.

2. W rankingu porównywać zgodne protokoły, kryterium najpierw walidacja i budżet FA, następnie złożoność/energia według wcześniej zapisanej reguły.

3. Zapisywać seed, checkpoint i config atomowo, wspierać resume z jawnego stanu. Nie wymuszać sukcesu przy nieosiągalnym budżecie FA/h.

**Odbiór:** Champion oznacza najlepszy według z góry określonej walidacji; jeśli budżet jest nieosiągalny, raport mówi infeasible.

**Przekazanie:** Patryk ma candidate model; Karolina ma manifest oraz historyczne punkty porównania.


===
# T04 Marcel krok 5 do 6
## M4 Ocenić ciągły strumień
D2–D3. Zależności: M3, K3.

1. Wczytać continuous manifest Kacpra, zachować stan SNN na granicach okien i odjąć tylko z góry określony warmup/gapy od ekspozycji.

2. Policzyć event recall, FA/h, false-alarm count, czas tła, one-to-one matching i opóźnienie od onset; elapsed treningu nie jest latency detekcji.

3. Zamrożony test uruchomić dopiero po wyborze modelu. Raportować przedział ufności FA/h i różnicę do historycznej metryki na klipach.

**Odbiór:** Każdy FA da się wskazać na osi czasu. Przy zerowym FA wynik ma dodatnią granicę górną niepewności, a nie obietnicę braku alarmów.

**Przekazanie:** Karolina dostaje CSV/JSON metryk; Andrzej identyczny zestaw do hardware.

## M5 Przekazać pakiet championa
D3. Zależności: M4, P1.

1. Spakować checkpoint, manifest, topologię, weights, decoder, calibration status i golden replay. Podać SHA256 każdego artefaktu.

2. Sprawdzić powtórne załadowanie w czystym procesie i identyczne wyjście. Duże artefakty przekazać przez uzgodniony storage, nie zwykły commit binariów.

3. PR #47 po poprawkach kierować do master. Po jego squash kolejny etap zaczynać z aktualnego master, żeby nie powielać starej historii.

**Odbiór:** Patryk i Wiktor wczytują dokładnie oceniony model; plik JSON nie obiecuje fizycznej kompatybilności bez kalibracji.

**Przekazanie:** Zamrożony raport naukowy, hash modelu i opis ograniczeń dla artykułu.


===
# T05 Kacper krok 1 do 2
Zgodny enkoder oraz ciągłe dane do pipeline’u Marcela. Branch: feat/encoder-features PR #50 oraz feat/continuous-dataset PR #48 — dwa istniejące zakresy.

## K1 Zamrozić działający wariant enkodera
D1 rano. Zależności: brak — można rozpocząć od razu.

1. Porównać trzykanałowy szkic ze starszego feat/encoder z wariantem features-improvement. Zidentyfikować faktyczny firmware i model, nie wybierać po nazwie pliku.

2. Wypisać sample_rate, preprocessing, gain/ADC, okno, kanały, progi, pulse width i encoder_hash. Nazwy kanałów w swap muszą odpowiadać mobility/autocorr, nie dawnym etykietom.

3. Sprawdzić kompilację i RAM/czas na rzeczywistym Uno ATmega328P; wyniki Cortex-M4F nie są dowodem wydajności Uno. Z Wiktorem ustalić format serial.

**Odbiór:** Jeden wariant ma jawny kontrakt i budżet czasu/RAM; przy braku zgodności blokuje się Start, a nie dopasowuje kanały heurystycznie.

**Przekazanie:** Wiktor dostaje opis i przykładową ramkę; Marcel zgodny encoder_hash do treningu.

## K2 Firmware i test zgodności
D1–D2. Zależności: K1, A1.

1. Dodać lub wydzielić serial output z timestamp i seq bez zakłócania próbkowania. Udokumentować przepustowość oraz przepełnienie bufora.

2. Na ustalonym wektorze porównać Python twin i firmware: cechy, impulsy i tolerancje. Zestaw obejmuje ciszę, impuls, sinus, szkło i nagłą zmianę amplitudy.

3. Z Wiktorem wykonać 30–60 minut pomiaru stabilności: przepełnienia, jitter, liczba próbek i restart. Nie podawać samego zgodnego pojedynczego klipu jako dowodu.

**Odbiór:** Firmware mieści się na Uno i dotrzymuje budżetu; zgodność ma zapisany raport, nie tylko ręczną obserwację LED.

**Przekazanie:** W1 ma gotowe Uno; Patryk ma identyczne wejście do symulacji.


===
# T05 Kacper krok 3 do 5
## K3 Dokończyć continuous dataset
D1–D2. Zależności: M1.

1. Na feat/continuous-dataset uzupełnić bazę o aktualny master bez utraty własnych zmian; branch jest 30 commitów za bazą. Na branchu współdzielonym preferować merge master do feature, nigdy odwrotnie.

2. W annotations.py i stream_builder.py wykluczyć grupy train oraz val z końcowego testu continuous. Rozdzielić strojenie continuous-val od continuous-test.

3. Zbudować manifest z source_id, group_id, onset/offset, gain, seed, warmup i ekspozycją; walidować także overlap źródeł i mixów VOICe.

4. Dodać testy rozłączności, minimalnych przerw, granic strumienia i deterministyczności; README ma opisywać rzeczywisty kod.

**Odbiór:** Ponowny build z tym samym seed daje ten sam manifest; walidator odrzuca wspólny group_id z train lub val; znany czas tła.

**Przekazanie:** Marcel pozostaje właścicielem całego pipeline’u; Kacper dostarcza builder i audyt jako jego wejście.

## K4 Baseline FFT dla Uno
D2–D3. Zależności: K2, A1.

1. Zaimplementować najprostszy wykonalny baseline cech częstotliwościowych i progu, z tą samą akwizycją oraz znanym kosztem SRAM.

2. Parametry dobrać wyłącznie na walidacji; odtworzyć strumienie M1/M4, zapisać decyzje i czas źródłowy.

3. Z Andrzejem zmierzyć moc baseline i akwizycji; zaznaczyć, czy porównanie obejmuje komunikację oraz Pi. FFT nie jest samodzielnym klasyfikatorem bez reguły decyzyjnej.

**Odbiór:** Baseline ma regułę, parametry, encoder/firmware hash i identyczny protokół oceny; pomiar nie miesza różnych urządzeń bez oznaczenia.

**Przekazanie:** Marcel otrzymuje decyzje do tego samego evaluator; Andrzej wynik dla bilansu energii.

## K5 PR i instrukcja dla Wiktora
D3. Zależności: K2, K3.

1. Dokończyć #48 i #50 jako odrębne PR: dane i firmware nie powinny mieć jednego nierozdzielnego diffu. Przed zmianą base sprawdzić porównanie z master.

2. W #50 usunąć przypadkową zmianę architecture_14_neurons_patryk_09_07/train_log.csv z zakresu kodu lub uzasadnić ją jako osobny artefakt; nie wybierać seed po teście.

3. Przekazać Wiktorowi komendę budowania, wgrywania, diagnostyki ADC i interpretacji spike; nie wymagać znajomości notebooków.

**Odbiór:** Oba PR mają testy i dowody oraz wymagane review; po squash nie dopisywać nowych zmian na tych samych dawnych branchach.

**Przekazanie:** Runbook Uno, fixture serial i raport datasetu; następne poprawki z nowego master.


===
# T06 Andrzej krok 1 do 2
Fizyczne Lu.i kalibracja oraz wiarygodne pomiary energii. Branch: feat/andrzej-hardware-energy.

## A1 Inwentaryzacja i bezpieczne stanowisko
D1 rano. Zależności: brak — można rozpocząć od razu.

1. Spisać liczbę/sprawność Lu.i, model Pi 5 i opcjonalnego Zero, kamerę, kable USB/OTG/CSI, zasilacze, buzzer i dostępny miernik/oscyloskop.

2. Potwierdzić pinout i zakres MAX4466→Uno ADC, masę i napięcia; Pi komunikuje się z Uno po USB. GPIO Pi nie przyjmuje 5 V.

3. Ustalić sterowanie buzzerem i wymagany driver na podstawie rzeczywistego modułu. Zrobić zdjęcie realnego okablowania i listę ustawień.

**Odbiór:** Wiktor może uruchomić stanowisko z instrukcji; braki są zapisane pierwszego dnia, zanim zacznie się integracja.

**Przekazanie:** Kacper zna tor ADC; Wiktor zna adapter kamery i wyjść; zespół wie, czy Zero jest opcją.

## A2 Kalibracja pojedynczych neuronów
D1–D2. Zależności: A1, P1.

1. Nadać board_id, zapisać nastawy trymerów i napięcie zasilania. Podać impulsy o kontrolowanej amplitudzie, czasie i szerokości.

2. Zmierzyć odpowiedź pobudzającą/hamującą, tau membrany/synaps, próg, reset, refractory, zakres nasycenia i LED. Zachować surowe przebiegi.

3. Przekazać osobny zestaw do dopasowania i walidacji Patryka. Z góry wspólnie ustalić tolerancje i sposób porównania czasów spike.

**Odbiór:** Każda wartość ma jednostkę, warunki i board_id. Jeśli nie da się zmierzyć Vmem, raport nie deklaruje zweryfikowanego przebiegu potencjału.

**Przekazanie:** Patryk dostaje CSV i opis; Karolina zakres fizycznej skali LED.


===
# T06 Andrzej krok 3 do 5
## A3 Pomiar energii urządzeń i sieci
D2–D3. Zależności: A1, K2.

1. Zdefiniować granice pomiaru: Uno+mikrofon, sieć Lu.i, Pi idle/capture/transmit/alarm. Zmierzyć pobór z LED i jawnie opisać ich udział.

2. Użyć napięcia/prądu w czasie; podać częstotliwość próbkowania miernika i niepewność. Powtórzyć ten sam przebieg kilka razy.

3. Dla krótkich zdarzeń uwzględnić ograniczenie miernika USB; nie wyliczać impulsowej energii ze zbyt wolnego pojedynczego odczytu.

4. Chmurę mierzyć jako koszt/zasoby albo oddzielną estymację; nie utożsamiać liczby spike z energią fizycznej płytki.

**Odbiór:** Raport rozdziela measured/estimated i podaje W oraz J/Wh przy tej samej ekspozycji. Brak pomiaru oznacza unavailable.

**Przekazanie:** Marcel otrzymuje tabelę do artykułu; Karolina dane Energy z granicą systemu.

## A4 Sieć fizyczna lub dane partnerskie
D3. Zależności: A2, M5.

1. Odwzorować kompatybilną topologię championa i nastawy; weryfikować fan-in, fan-out, znak i timing. Nie każda sieć 0–50 z edytora pasuje do fizycznego zestawu.

2. Odtworzyć te same wejściowe spike co w symulacji, zapisać rzeczywiste output spike/decisions i źródło zegara.

3. Jeżeli używane są wyniki drugiego zespołu: zebrać board revision, nastawy, firmware, encoder/model hash, surowe dane, aparat pomiarowy i zgodę na wykorzystanie/atrybucję.

**Odbiór:** Identyczny typ Lu.i nie zastępuje zgodności konfiguracji. Dane nieporównywalne trafiają wyłącznie do wyników wstępnych z jasnym opisem.

**Przekazanie:** Marcel otrzymuje porównywalny raport albo jawny brak finalnego eksperymentu.

## A5 Instrukcja i finalny odbiór sprzętu
D3. Zależności: A3.

1. Opisać uruchomienie, kontrolę poziomów, bezpieczne wyłączenie i symptomy błędnego okablowania.

2. Z Wiktorem sprawdzić fizyczny LED/buzzer: brak alarmu po starcie, limit czasu, Stop i awaria połączenia.

3. Przekazać podpisane identyfikatorem sesji zdjęcia stanowiska i pliki kalibracji przez własny PR.

**Odbiór:** Druga osoba uruchamia stanowisko bez domysłów, a alarm nie pozostaje włączony po utracie chmury.

**Przekazanie:** Zespół ma materiał do demonstracji i metodologii artykułu.


===
# 8 Trening na MacBooku M5 Max 128 GB
Docelowym hostem treningu jest MacBook Pro M5 Max z 128 GB unified memory, wskazany przez Wiktora. Trening działa lokalnie w natywnym środowisku ARM64 macOS. Azure obsługuje runtime SNN, dashboard i vision; Raspberry Pi pozostaje lekkim edge. Nie używamy Docker Desktop do treningu wymagającego bezpośredniego MPS.

128 GB to pamięć współdzielona CPU i GPU, z której korzysta także system. Nie oznacza 128 GB wolnej pamięci GPU. Nie dobieramy liczby workerów z samej pojemności RAM i nie zakładamy, że Apple Neural Engine automatycznie wykonuje trening PyTorch. Standardową ścieżką GPU jest backend MPS; wsparcie konkretnego stosu potwierdza test na docelowym urządzeniu.

## Co trzeba poprawić w obecnym pipeline
hardware.py w feat/master-pipeline wybiera CPU dla wszystkich komputerów macOS w trybie auto. Historyczny komentarz o około trzykrotnie wolniejszym MPS dotyczy wcześniejszego pomiaru małej SNN, nie jest benchmarkiem tego M5 Max. Z kolei ręczne --device mps przy wielu workerach może skierować wiele procesów na jeden GPU, więc nie powinno być bezwarunkowym nowym defaultem.

Obecny benchmark_workers wybiera minimalny elapsed, chociaż liczba zadań rośnie razem z workers: total_tasks = workers × tasks_per_worker. Wyniki nie porównują takiej samej ilości pracy. Sztuczne mnożenie macierzy nie mierzy pełnego GA z ładowaniem danych, rozwijaniem SNN w czasie i dekoderem. Naprawa M0 jest obowiązkowa przed stwierdzeniem, że profil jest zoptymalizowany.

## Plan benchmarku
| Wariant | Stałe warunki i miara |
| CPU GA | workers 1, 2, 4, 8, 12, 16 ograniczone do dostępnych zasobów; torch threads=1; spawn |
| MPS GA | Jeden proces wykonujący trening GPU; ta sama lista genomów i budżet; bez puli wielu właścicieli GPU |
| Dotrenowanie championa | Oddzielny benchmark CPU/MPS; batch 128, 256, 512 tylko przy zachowaniu definicji eksperymentu |
| Dane | Ten sam encoder_hash, split, liczba próbek i seed; brak testu w selekcji |
| Pomiar | Warmup, potem co najmniej trzy powtórzenia; kandydaci/min, mediana czasu, peak memory, pressure i swap |
| GPU timing | Synchronizacja MPS przed startem/końcem pomiaru; rozdzielić startup i steady state |

Zmiana batch size może zmienić trening, więc jest parametrem eksperymentu, a nie darmową optymalizacją. Na czas porównania urządzeń najpierw utrzymujemy identyczny batch i liczbę aktualizacji. FP32 jest bazą; mixed precision, kompilacja i inny framework pozostają dodatkiem po wykazaniu zgodności oraz korzyści.
===
# 9 Profil i odbiór treningu
Marcel dodaje wersjonowany profil macbook_m5_max_128gb wraz z raportem pomiarów. Pole device_auto nie może ukrywać nieprzetestowanej heurystyki. Jeśli CPU wygrywa dla małych SNN, CPU jest prawidłowym wynikiem strojenia; jeśli MPS wygrywa dla dotrenowania, etapy mogą mieć różne urządzenia. Profil nie wymaga wykorzystania całych 128 GB.

## Pola raportu sprzętowego
| Pole | Zawartość |
| hardware | Model, liczba rdzeni CPU/GPU, RAM, macOS; bez numeru seryjnego |
| environment | ARM64, Python, torch, wersja kodu, zależności i dostępność MPS |
| ga_execution | device, workers, torch_threads, population, epochs, samples, seeds |
| winner_execution | device, batch_size, dtype, epochs i reguła wyboru checkpointu |
| data_cache | Format cache, rozmiar i liczba kopii na worker; ograniczenie pamięci |
| benchmark | Stały workload, powtórzenia, throughput, wall time, RSS, MPS allocated/driver memory |
| reliability | Checkpoint, resume, anulowanie, OOM, zapis błędów i brak cichego CPU fallback |
| reproducibility | Seed, manifest/hash danych, tolerancja różnic CPU/MPS, golden replay |

Na starcie stosujemy ostrożny budżet całego treningu około 80 GB jako propozycję organizacyjną i obserwujemy system. Nie jest to limit sprzętowy ani wyliczony dopuszczalny rozmiar alokacji MPS. W razie rosnącego swap lub memory pressure redukujemy cache, liczbę workerów lub batch. Nie ustawiamy PYTORCH_MPS_HIGH_WATERMARK_RATIO=0, aby wymusić większą alokację; dokumentacja ostrzega przed wyłączeniem limitu.

Nie zakładamy współdzielenia wszystkich danych przez procesy spawn: RealFitness jest inicjalizowany osobno w workerze, więc trzeba zmierzyć duplikację cache. MPS fallback dla nieobsługiwanych operacji ma być jawnie raportowany; benchmark hybrydowy nie może być nazwany czystym GPU. Jedna próba zakończona OOM nie powinna niszczyć ostatniego poprawnego checkpointu.

## Odbiór M0
Ten sam rzeczywisty zestaw genomów przechodzi baseline CPU i wybraną konfigurację bez NaN, brakujących prób ani zmiany kryterium jakości. Raport uzasadnia wybór urządzenia i workerów. Test wznowienia odtwarza stan eksperymentu. Nie deklarujemy przyspieszenia bez pomiaru na M5 Max 128 GB.

Źródła: https://developer.apple.com/metal/pytorch/ ; https://docs.pytorch.org/docs/stable/notes/mps.html ; https://docs.pytorch.org/docs/stable/mps_environment_variables.html ; https://docs.pytorch.org/docs/stable/mps.html . Konkretne wersje środowiska zamrażamy po smoke teście; nie przerywamy ani nie aktualizujemy zależności trwającego GA w tej sesji.

===
# 10 Reguły decyzji i ograniczenia terminu
## Co jest obowiązkowym wynikiem demonstratora
Działający E2E na Pi 5, jawny stan błędów, zgodny enkoder/model, angielski dashboard, wspólny login, działające PR-only i manifest wersji. Wyniki badawcze muszą zachować niezależny test i pochodzenie danych. Brak docelowej jakości klasyfikacji należy pokazać, nie maskować zmianą progu po teście.

## Czego nie dodajemy kosztem podstaw
Rozpoznawanie domowników pozostaje kolejną fazą. Migracja na Zero nastąpi dopiero po identyfikacji wersji i kabli. Nie wykonujemy rozbudowanej przebudowy całego historycznego repo, nowego systemu kont, nowej bazy SQL ani nowego klastra. Edytor 0–50 nie oznacza, że w trzy dni wytrenujemy i fizycznie skalibrujemy 50 neuronów.

## Punkty decyzyjne
| Moment | Warunek i reakcja |
| D1 rano | Brak określonego enkodera: najpierw K1, bez nowego GA na przypadkowych cechach |
| D1 wieczór | Uno nie nadąża: zmierzyć ograniczenie, użyć zgodnego prostszego wariantu; zmiana kanałów wymaga nowego modelu |
| D2 południe | Brak realnego E2E: zespół kończy integrację; wstrzymuje dodatkowe warianty wizualne i eksperymenty |
| D2 wieczór | Brak pełnego hardware: kontynuować symulację; dane partnerskie tylko z A4 i jawnie jako wstępne |
| D3 | Brak niezależnej finalnej oceny: demonstracja jest prototypem, a wynik artykułu ma opisane ograniczenie |

Reguła alarmu nadal wymaga rozróżnienia „osoba widoczna” od „osoba nieuprawniona”. Wspólne konto portalu tej decyzji nie rozwiązuje. Do zatwierdzenia polityki automatyczny alarm nie powinien opierać się na przypisaniu uprawnień przez sam model vision. Szczegóły są otwartą decyzją architektury, nie dodatkowym kontem użytkownika.

## Definicja zakończenia
Każda osoba oddaje swój PR, dowody testów i wskazane przekazanie. Integrator wykonuje test całego łańcucha po finalnym squash. Raport rozdziela actual hardware, simulation, replay oraz estimated energy. Artykuł zaczyna się od tego zamrożonego materiału; wcześniejsze eksperymenty pozostają zidentyfikowanymi źródłami porównawczymi.

===
# 11 Źródła i materiały dla zespołu
## Zweryfikowane punkty repozytorium
Snapshot zdalnych branchy i PR został sprawdzony przez GitHub API; szczegóły liczb i pełne SHA są w analiza_branchy.json. Wybór bazy: feat/testing-ideas @ c8c6eab1. Główne dowody: master_pipeline/ga_runner.py; dataset/continuous/eval/annotations.py i stream_builder.py; encoder/features-improvement/WNIOSKI.md oraz firmware/twin; rpi_agents/cloud/app/auth.py i położenie starego deploy.yml.

Otwarte prace: https://github.com/Industrial-Data-Science-AGH/SNN_Agent/pull/47 ; https://github.com/Industrial-Data-Science-AGH/SNN_Agent/pull/48 ; https://github.com/Industrial-Data-Science-AGH/SNN_Agent/pull/50 . Są odniesieniem do planu; nie zostały scalone w tej sesji.

## Ochrona GitHub
Aktywna reguła: https://github.com/Industrial-Data-Science-AGH/SNN_Agent/rules/23904463

Dokumentacja reguł: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets

REST API i weryfikacja: https://docs.github.com/en/rest/repos/rules

Reguły serwera sprawdzono przez rules/branches/master i branches/master. Testy lokalnych hooków wykonano w pomocniczym repo; nie próbowano łamać ochrony przez publikację nieautoryzowanego commitu na master.

## Bezpieczeństwo wspólnego dostępu
OWASP Authentication Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html

OWASP Session Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html

Źródła opisują mechanizmy; wartości TTL i limitu logowania w tym planie są propozycjami ustawień demonstratora. Nie wdrożono jeszcze mechanizmu logowania w kodzie.

## Zawartość pakietu
Plan_zespolu_SNN.docx — dokument do wspólnej edycji po imporcie do Google Docs. Mapa_projektu.html — lokalna mapa z rozwijanymi krokami. assets/mapa-zespolu.svg — pełna mapa kafelkowa do powiększania; assets/kolejnosc.svg — diagram zależności. taski/T01–T06.md — osobne checklisty właścicieli. zadania.json — strukturalny plan. hooks oraz install-hooks.sh — lokalne zabezpieczenie dla pozostałych klonów.
