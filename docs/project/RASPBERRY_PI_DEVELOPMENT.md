# Praca z agentem i testy na Raspberry Pi

## Zalecany układ

Kod, przegląd i cięższe narzędzia AI uruchamiamy na Macu. Raspberry Pi 5 jest zdalnym stanowiskiem do testów integracyjnych przez SSH. To pozwala Claude Code i Codex uruchamiać te same jawne polecenia testowe, bez instalowania ich na Pi i bez obciążania ewentualnego Pi Zero. Usługa edge pozostaje mała, niezależna od narzędzia programisty i wymienia dane z chmurą przez kontrakt W0. Po potwierdzeniu sprzętu ten sam pakiet może być testowany na Pi Zero, z osobną weryfikacją adapterów kamery i GPIO.

## Przygotowanie stanowiska przez Wiktora

1. Zanotuj model Pi, architekturę OS (`uname -m`), wersję Python, typ kamery, numer seryjny Uno, interfejs USB i pinout LED/buzzera. Pi 5 i Pi Zero mogą wymagać innych sterowników lub obrazu systemu.
2. Włącz SSH w Raspberry Pi OS, utwórz osobne konto programistyczne i dodaj klucz publiczny Maca do `~/.ssh/authorized_keys`. Sprawdź fingerprint hosta przed pierwszym logowaniem. Nie publikuj hasła, klucza prywatnego, adresu IP ani tokenów w repo, promptach i logach. Dla pracy poza LAN użyj zaufanego VPN/tunelu, bez wystawiania portu 22 do Internetu.
3. Dodaj lokalny wpis `Host snn-pi` w `~/.ssh/config` z `HostName`, `User`, `IdentityFile` i `IdentitiesOnly yes`; w repo nie umieszczaj prawdziwych wartości. Sprawdź `ssh snn-pi 'hostname; uname -m; python3 --version'`.
4. Na Pi utwórz katalog roboczy i `python3 -m venv .venv` oraz instaluj zależności według plików projektu. Sekrety urządzenia trzymaj poza checkoutem, w pliku dostępnym tylko dla konta usługi albo systemowym magazynie; backend pobiera sekrety z Key Vault przez odpowiednią tożsamość. Klient Pi nie powinien znać klucza Foundry ani hasła portalu.
5. Zmapuj porty urządzeń przez trwałe identyfikatory, np. `/dev/serial/by-id/`, zamiast założyć stałe `/dev/ttyACM0`. Zweryfikuj kamerę poleceniem dostępnego sterownika (`rpicam-hello` dla obsługiwanej CSI lub `v4l2-ctl`/`lsusb` dla USB). Test LED i buzzera uruchamiaj kontrolowaną komendą z limitem czasu i stanem OFF w `finally`.

## Pętla implementacyjna

Zacznij od aktualnego `origin/master` i osobnego `feat/` brancha. Codex lub Claude Code edytuje kod lokalnie, uruchamia szybkie testy kontraktowe i statyczne, a dopiero potem kopiuje **wyłącznie potrzebne pliki aplikacji** na Pi lub testuje commit pobrany przez Pi. Unikaj synchronizacji całego katalogu wraz z `.git`, datasetem, modelami i sekretami. Do zmian niezatwierdzonych można użyć `rsync` z jawną listą katalogów do katalogu testowego; do testu integracyjnego i raportu użyj commita z hashem i `git fetch` na Pi. Nie aktualizuj działającej instancji przez przypadkowy `git pull`.

Na Pi uruchom w kolejności: test serial i timestampów na fixture W0; test kamery; test GPIO w trybie kontrolowanym; test komunikacji z mock API; test sesji z Azure; pełny scenariusz E2E. Zapisz commit, model Pi, firmware Uno, wersję kontraktu, model SNN, czas i log testu. Testy mock nie są wynikami naukowymi. Eksperymenty z rzeczywistym alarmem wykonuj świadomie, przy obecności człowieka, z limitem czasu buzzera. Awaria analizy obrazu nie może sama wywoływać alarmu.

Jeśli SSH nie jest jeszcze skonfigurowane, najpierw można rozwijać kod na Macu z mock API i fixture W0. Testy sprzętowe wymagają fizycznie osiągalnego Pi, dostępu SSH i przypiętych urządzeń. Edytor Remote SSH jest opcją dla wygody na Pi 5; przy Pi Zero preferuj lokalny edytor i zdalne polecenia, bo serwer edytora zużywa pamięć. Zobacz [instrukcję SSH Raspberry Pi](https://www.raspberrypi.com/documentation/computers/remote-access.html) oraz [wymagania VS Code Remote SSH](https://code.visualstudio.com/docs/remote/ssh).
