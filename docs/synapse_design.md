# Projekt Synaps (Połączeń między neuronowych)

Synapsa w analogowym SNN pełni rolę wagi i jest de facto zrealizowana mechanizmem dzielnika napięciowego (Rezystor wejściowy N2 + Rezystor Szeregowy).

## 1. Wyliczenie Wagi
Napięcie błony docelowej $V_{mem} = V_{spike} \cdot \frac{R_{in}}{R_{syn} + R_{in}}$
Waga $\omega$ połączenia to wynikowy stosunek tych impedancji.

## 2. Dobór rezystorów
- Ostrzeżenie na najtańsze komponenty (np. 5% z dużych kitów THT). Różnica 50 ohm na 1k wpływa na przesunięcie próg V_th. 
- Do obwodów synaptycznych używaj **wyłącznie rezystorów 1% (Szereg E96)**.

## 3. Rezystancja Upływu ($R_{leak}$) 
Czas degradacji ładunku w kondensatorze membranowym (Leakage tau $\tau = R_{leak} \cdot C_{mem}$).
Należy zmatchować dobór ze stałą czasową wymaganą do usłyszenia "rozbitego szkła" (ok. 50-200ms burst duration).

## 4. Opcja konfigurowalna
By nie marnować płytek na re-spin przy zmianie parametrów sieci z fazy symulacyjnej, wlutować w miejsce $R_{syn}$ 2-pinowy pin header żeński, by łatwo modyfikować wagi na płytce stykowej lub wymieniając przewlekane rezystory.
