# Pakiet planowania SNN Agent

Plan_zespolu_SNN.docx zawiera plan, analizę branchy, PR, wspólny login i profil M5 Max 128 GB. Mapa_projektu.html działa lokalnie po otwarciu w przeglądarce; rozwiń kafelek, by zobaczyć odbiór i przekazanie. SVG w assets można powiększać.

Sześć tasków właścicieli zawiera 32 kroki. Są planem, nie raportem wdrożenia. Istniejący kod aplikacji pozostał niezmieniony. Utworzono master z c8c6eab1, ustawiono go domyślnym i aktywowano ruleset 23904463. W tym klonie zainstalowano lokalne hooki.

## Hooki w innych klonach

Z katalogu repozytorium uruchom ścieżkę do install-hooks.sh z tego pakietu. Instalator nie nadpisuje istniejących hooków ani core.hooksPath. Hooki blokują lokalny commit/merge na master i push do zdalnego master; nie są zabezpieczeniem serwera i można je lokalnie ominąć. Reguła GitHub obowiązuje niezależnie od ich instalacji.

## Pierwsze kroki

1. Pobrać origin/master; bez przełączania lub resetowania cudzej pracy.
2. W0 kontrakty i rzeczywisty check pr-gate przez PR. Dopiero potem wymagany status check.
3. M0 profilowanie MacBooka M5 Max 128 GB, bez zatrzymywania trwającego GA.
4. Kontynuować checklisty właścicieli i integrować przez PR.

Branche osób, wdrożenie loginu i poprawki treningu nie zostały jeszcze wykonane. Otwarte PR #47, #48, #50 pozostają na dotychczasowej bazie do świadomego retargetowania przez właścicieli.
