# T05 Kacper

Zgodny enkoder oraz ciągłe dane do pipeline’u Marcela

Branch: feat/encoder-features PR #50 oraz feat/continuous-dataset PR #48 — dwa istniejące zakresy

Reviewer: Marcel dla danych; Wiktor i Andrzej dla Uno

Zakres plików: encoder/features-improvement/**; dataset/continuous/**; tests/encoder/**; encoder/uno_baseline/** (nowy)

Punkt startowy: fixed_encoder z feat/encoder; features-improvement/twin i tools; continuous/eval i tests

## K1 Zamrozić działający wariant enkodera

Kolejność: D1 rano. Zależności: brak.

1. Porównać trzykanałowy szkic ze starszego feat/encoder z wariantem features-improvement. Zidentyfikować faktyczny firmware i model, nie wybierać po nazwie pliku.

2. Wypisać sample_rate, preprocessing, gain/ADC, okno, kanały, progi, pulse width i encoder_hash. Nazwy kanałów w swap muszą odpowiadać mobility/autocorr, nie dawnym etykietom.

3. Sprawdzić kompilację i RAM/czas na rzeczywistym Uno ATmega328P; wyniki Cortex-M4F nie są dowodem wydajności Uno. Z Wiktorem ustalić format serial.

Odbiór: Jeden wariant ma jawny kontrakt i budżet czasu/RAM; przy braku zgodności blokuje się Start, a nie dopasowuje kanały heurystycznie.

Przekazanie: Wiktor dostaje opis i przykładową ramkę; Marcel zgodny encoder_hash do treningu.

## K2 Firmware i test zgodności

Kolejność: D1–D2. Zależności: K1, A1.

1. Dodać lub wydzielić serial output z timestamp i seq bez zakłócania próbkowania. Udokumentować przepustowość oraz przepełnienie bufora.

2. Na ustalonym wektorze porównać Python twin i firmware: cechy, impulsy i tolerancje. Zestaw obejmuje ciszę, impuls, sinus, szkło i nagłą zmianę amplitudy.

3. Z Wiktorem wykonać 30–60 minut pomiaru stabilności: przepełnienia, jitter, liczba próbek i restart. Nie podawać samego zgodnego pojedynczego klipu jako dowodu.

Odbiór: Firmware mieści się na Uno i dotrzymuje budżetu; zgodność ma zapisany raport, nie tylko ręczną obserwację LED.

Przekazanie: W1 ma gotowe Uno; Patryk ma identyczne wejście do symulacji.

## K3 Dokończyć continuous dataset

Kolejność: D1–D2. Zależności: M1.

1. Na feat/continuous-dataset uzupełnić bazę o aktualny master bez utraty własnych zmian; branch jest 30 commitów za bazą. Na branchu współdzielonym preferować merge master do feature, nigdy odwrotnie.

2. W annotations.py i stream_builder.py wykluczyć grupy train oraz val z końcowego testu continuous. Rozdzielić strojenie continuous-val od continuous-test.

3. Zbudować manifest z source_id, group_id, onset/offset, gain, seed, warmup i ekspozycją; walidować także overlap źródeł i mixów VOICe.

4. Dodać testy rozłączności, minimalnych przerw, granic strumienia i deterministyczności; README ma opisywać rzeczywisty kod.

Odbiór: Ponowny build z tym samym seed daje ten sam manifest; walidator odrzuca wspólny group_id z train lub val; znany czas tła.

Przekazanie: Marcel pozostaje właścicielem całego pipeline’u; Kacper dostarcza builder i audyt jako jego wejście.

## K4 Baseline FFT dla Uno

Kolejność: D2–D3. Zależności: K2, A1.

1. Zaimplementować najprostszy wykonalny baseline cech częstotliwościowych i progu, z tą samą akwizycją oraz znanym kosztem SRAM.

2. Parametry dobrać wyłącznie na walidacji; odtworzyć strumienie M1/M4, zapisać decyzje i czas źródłowy.

3. Z Andrzejem zmierzyć moc baseline i akwizycji; zaznaczyć, czy porównanie obejmuje komunikację oraz Pi. FFT nie jest samodzielnym klasyfikatorem bez reguły decyzyjnej.

Odbiór: Baseline ma regułę, parametry, encoder/firmware hash i identyczny protokół oceny; pomiar nie miesza różnych urządzeń bez oznaczenia.

Przekazanie: Marcel otrzymuje decyzje do tego samego evaluator; Andrzej wynik dla bilansu energii.

## K5 PR i instrukcja dla Wiktora

Kolejność: D3. Zależności: K2, K3.

1. Dokończyć #48 i #50 jako odrębne PR: dane i firmware nie powinny mieć jednego nierozdzielnego diffu. Przed zmianą base sprawdzić porównanie z master.

2. W #50 usunąć przypadkową zmianę architecture_14_neurons_patryk_09_07/train_log.csv z zakresu kodu lub uzasadnić ją jako osobny artefakt; nie wybierać seed po teście.

3. Przekazać Wiktorowi komendę budowania, wgrywania, diagnostyki ADC i interpretacji spike; nie wymagać znajomości notebooków.

Odbiór: Oba PR mają testy i dowody oraz wymagane review; po squash nie dopisywać nowych zmian na tych samych dawnych branchach.

Przekazanie: Runbook Uno, fixture serial i raport datasetu; następne poprawki z nowego master.