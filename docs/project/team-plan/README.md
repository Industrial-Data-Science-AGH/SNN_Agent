# Pakiet planowania SNN Agent

**Aktualny stan po W0:** W0 jest scalone z `master`. Priorytet eksperymentu i status fizycznych płytek opisuje [aktualny zakres](../CURRENT_ASSUMPTIONS.md); poniższa historia i plan zadań pochodzą z wcześniejszego etapu. W szczególności pomiar fizycznej sieci Lu.i jest rozszerzeniem, nie warunkiem ukończenia podstawowego porównania.

Plan_zespolu_SNN.docx zawiera plan, analizę branchy, PR, wspólny login i profil M5 Max 128 GB. Mapa_projektu.html działa lokalnie po otwarciu w przeglądarce; rozwiń kafelek, by zobaczyć odbiór i przekazanie. SVG w assets można powiększać.

Sześć tasków właścicieli zawiera 32 kroki. Są planem, nie raportem wdrożenia. Istniejący kod aplikacji pozostał niezmieniony. Utworzono master z c8c6eab1, ustawiono go domyślnym i aktywowano ruleset 23904463. W tym klonie zainstalowano lokalne hooki.

## Hooki w innych klonach

Z katalogu repozytorium uruchom ścieżkę do install-hooks.sh z tego pakietu. Instalator nie nadpisuje istniejących hooków ani core.hooksPath. Hooki blokują lokalny commit/merge na master i push do zdalnego master; nie są zabezpieczeniem serwera i można je lokalnie ominąć. Reguła GitHub obowiązuje niezależnie od ich instalacji.

## Pierwsze kroki

1. Pobrać origin/master; bez przełączania lub resetowania cudzej pracy.
2. Przeczytać [W0](../W0_START_HERE.md) i [aktualne założenia](../CURRENT_ASSUMPTIONS.md), następnie wyciąć własny branch `feat/`.
3. Wykonać M0 profilowanie MacBooka M5 Max 128 GB, bez zatrzymywania trwającego GA.
4. Kontynuować checklisty właścicieli i integrować przez PR z zielonym `pr-gate`.

Stan poszczególnych branchy, PR, loginu i treningu należy sprawdzić przed pracą w GitHub; powyższy plan nie jest raportem bieżącego wdrożenia.
