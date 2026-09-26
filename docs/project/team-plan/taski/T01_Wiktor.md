# T01 Wiktor

Lekki edge, chmura, wspólny dostęp i integracja end to end

Branch: feat/wiktor-edge-cloud

Reviewer: Patryk dla API i bezpieczeństwa; Karolina dla UI

Zakres plików: rpi_agents/agent/**; rpi_agents/cloud/app/{auth,storage,worker,vision,policy}*.py; rpi_agents/cloud/infra/**; rpi_agents/deploy/**; .github/workflows/**; contracts/**

Punkt startowy: feat/dashboard: rpi_agents/agent, cloud/app, cloud/infra, deploy; nie importować całej historii brancha

## W0 Kontrakty i podstawa integracji

Kolejność: D1 rano. Zależności: brak.

1. Z Patrykiem i Kacprem zamrozić schema_version, channel_map, jednostki, zegar i encoder_hash. Zapisać kontrakty SpikeBatch, SNNDecision, NeuronFrame, CaptureCommand, VisionResult, AlarmCommand oraz manifest modelu.

2. W osobnym krótkim PR utworzyć contracts/ z JSON Schema i minimalnymi fixture: cisza, spike, duplikat, luka, trigger, vision unavailable. Dodać mock API do pracy bez urządzeń.

3. Przenieść wybrane rpi_agents z feat/dashboard do nowego brancha. Nie kopiować .env, wyników, starych ADR jako obowiązującego projektu ani całej historii dev. Zapisać źródłowy SHA.

4. Dodać root .github/workflows/pr.yml: pull_request na master, read-only token, testy kontraktów i lekkie testy modułów, bez sekretów i bez kosztownych treningów. Dopiero po udanym uruchomieniu ustawić wymagany check pr-gate.

Odbiór: Pierwszy PR pokazuje walidację poprawnych fixture i odrzucenie błędnego kanału/wersji. Workflow rzeczywiście uruchamia się z głównego .github/workflows, a nie z rpi_agents/.github/workflows.

Przekazanie: Patryk i Karolina dostają fixtures oraz stabilne nazwy endpointów; wszyscy pobierają master po squash PR.

## W1 Bridge Uno i Raspberry Pi 5

Kolejność: D1. Zależności: W0, K1.

1. Z Andrzejem sprawdzić urządzenia USB, mikrofon i wyjścia; zapisać adapter kamery i serial device w konfiguracji.

2. Z Kacprem odebrać ramki Uno. Walidować seq, boot_id, zakresy i długość; bufor ograniczony, jawny gap zamiast cichej utraty. Batching z czasem źródłowym, nie czasem nadejścia HTTP.

3. Zaimplementować lekki proces systemd: reconnect USB/HTTPS, trwały mały outbox, heartbeat i obsługę SIGTERM. Nie instalować torch ani lokalnego vision.

4. Oddzielić CameraAdapter, SerialAdapter i AlarmAdapter od logiki. Capture wykonać dopiero po ważnej komendzie SNN; czas TTL i deduplikacja po command_id.

Odbiór: Na Pi 5 działa serial replay i fizyczne Uno; ponowne podłączenie nie gubi tożsamości sesji. Powtórzona komenda nie robi kolejnych zdjęć.

Przekazanie: Patryk otrzymuje zapis wejścia; Karolina otrzymuje DeviceStatus; ten sam program ma działać na przyszłym Pi Zero po zmianie konfiguracji.

## W2 Trwałe zdarzenia i polecenia

Kolejność: D1–D2. Zależności: W0.

1. Przygotować API urządzenia, session/event_id, własność obiektów i idempotentne zapisy do Table/Blob. Zdefiniować limity JPEG i batchy.

2. Zapis zdarzenia i outbox powiązać jednym planem odzyskiwania; publikować zadania do Queue. Worker deduplikuje, odnawia visibility timeout i kończy retry w kolejce błędów.

3. Outbox skanować przy uruchomieniu i okresowo podczas pracy; przed scale-to-zero potwierdzić opróżnienie. Przy awarii pomiędzy zapisem a enqueue potrzebny jest niezależny okresowy reconciler albo jawne utrzymywanie minReplicas=1 aż do odrobienia zaległości.

4. Komenda alarmu zawiera TTL, event_id i command_id. Pi potwierdza applied/expired/failed, wyjścia wyłączają się po lokalnym limicie niezależnie od chmury.

Odbiór: Test crash-before-enqueue kończy się odtworzeniem zadania; duplikat nie tworzy dwóch zdarzeń ani ponownego buzzera; obce device_id jest odrzucone.

Przekazanie: Karolina otrzymuje listę zdarzeń i statusy ACK; W3 dostaje kolejkę vision.

## W3 Foundry i reguła alarmu

Kolejność: D2. Zależności: W1, W2, P2.

1. Wybrać dostępny na subskrypcji model vision, zapisać model/deployment/version i sprawdzić poprawny JPEG oraz timeout. Dostęp z managed identity, jeśli wdrożenie go wspiera.

2. Model zwraca wyłącznie obserwacje zgodne ze schematem: glass_visible, person_visible, uncertainty. Nie nadaje osobie statusu unauthorized bez osobnej informacji.

3. Politykę alarmu trzymać w kodzie. Do czasu rozstrzygnięcia uzbrojenia/nieuprawnionej osoby: Review required. SNN=true samo nie uruchamia alarmu.

4. Przenieść e-mail do workera; adresaci z konfiguracji, załącznik wskazuje event_id. Retry nie może obiecywać exactly-once e-mail przy niejednoznacznym timeout.

Odbiór: Cztery scenariusze glass/person oraz błąd vision dają zapisane statusy. Błąd API nigdy nie jest zamieniany w pewne włamanie. Alarm spełnia zatwierdzoną regułę i gaśnie po limicie.

Przekazanie: Karolina otrzymuje realne wyniki i obraz do Event details; Andrzej rejestruje pobór w fazie capture/vision/alarm.

## W4 Azure i wspólny login

Kolejność: D2. Zależności: W0, C1.

1. Sprawdzić dostępne regiony az CLI, modele Foundry i quota. Zbudować kontenery API/SNN/UI oraz workera; utworzyć Blob/Table/Queue i Key Vault.

2. Dodać formularz Username/Password dla jednej wspólnej pary, sesję serwerową i Logout. Brak rejestracji, ról i kont per osoba. Hasło lub jego weryfikator tylko w Key Vault; brak domyślnych danych logowania.

3. Włączyć HTTPS, Secure/HttpOnly/SameSite cookie, ochronę CSRF, TTL sesji, ograniczenie prób logowania oraz redakcję logów. Pi używa osobnego poświadczenia urządzenia.

4. Na test aktywować minReplicas=1 dla runtime SNN. Po Stop i drain wrócić do min0. Jedna aktywna epoka i lease również podczas zmiany rewizji; bez obietnicy zerowego kosztu całej subskrypcji.

Odbiór: Login działa z dwóch przeglądarek tą samą parą, bezpieczny logout unieważnia daną sesję, nieautoryzowany odczyt zdjęć/telemetrii jest blokowany. Po restarcie sekret nie jest drukowany ani dodany do obrazu.

Przekazanie: Zespół dostaje URL; hasło Wiktor przekazuje własnym kanałem, poza rozmową z LLM. Karolina podłącza ekran loginu.

## W5 Odbiór i wersja demonstracyjna

Kolejność: D3. Zależności: W3, W4, P5, C5, M5, K2, A3.

1. Uruchomić skryptowany test E2E: Uno → Pi → SNN → capture → Foundry → policy → email/LED/buzzer. Zachować event_id oraz czasy wszystkich faz.

2. Sprawdzić przerwanie Wi-Fi, restart kontenera, odłączenie USB, timeout vision, duplikat i TTL. Awaria ma widoczny stan, bez fikcyjnej ciągłości.

3. Zamrozić release po zatwierdzonych PR, zapisać SHA obrazu/modelu/enkodera i parametry sesji. Nie poprawiać kodu bezpośrednio na urządzeniu bez przeniesienia przez PR.

4. Przygotować checklistę migracji: model Zero, architektura CPU, OS, USB/OTG, zasilanie, kamera i instalacja zależności. Decyzja o migracji dopiero po identyfikacji sprzętu.

Odbiór: Demonstracja powtarza się z czystego wdrożenia; awarie mają ślad i dają się odtworzyć. Raport rozdziela wyniki rzeczywiste od replay.

Przekazanie: Cały zespół: instrukcja uruchomienia, rollback obrazu/modelu, manifest release i lista ograniczeń.