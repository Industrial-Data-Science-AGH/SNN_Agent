# T04 Marcel

Pipeline badań i niezmienny champion na MacBooku M5 Max 128 GB

Branch: feat/master-pipeline — kontynuacja PR #47

Reviewer: Patryk; Kacper dla continuous evaluation

Zakres plików: master_pipeline/**; ga_neuron_search/{fitness,winner,export_winner}.py tylko gdy konieczne; tests/pipeline/**; raporty bez dużych binariów; master_pipeline/hardware.py i profiles/macbook_m5_max_128gb.json

Punkt startowy: master_pipeline/ga_runner.py, pipeline.py, pipeline_config.py, rank_runs.py, test_pipeline.py

## M0 Profil treningu M5 Max 128 GB

Kolejność: D1 rano. Zależności: brak.

1. Potwierdzić ARM64 bez Rosetty, model CPU/GPU, macOS, Python i torch; sprawdzić is_built/is_available MPS. Nie przebudowywać środowiska trwającego treningu. Utworzyć odrębny lock i profil macbook_m5_max_128gb.

2. Naprawić benchmark_workers: każda konfiguracja ocenia ten sam zestaw rzeczywistych genomów, epok, próbek i seedów. Obecne tasks_per_worker mnoży ilość pracy przez liczbę workerów, a dummy matmul nie reprezentuje SNN.

3. Zmierzyć CPU GA z 1/2/4/8/12/16 workerami w granicach wykrytej maszyny i jednym wątkiem torch na worker. MPS przetestować osobno z jednym procesem treningowym, dla batch 128/256/512. Wybrać per etap na podstawie kandydatów/min, czasu epoki, jakości i pamięci.

4. Raportować RSS, memory pressure, swap, pamięć MPS i czas po synchronizacji GPU; zacząć od budżetu procesu około 80 GB jako ustawienia ostrożnego, nie fizycznego limitu. Bez wyłączania limitu MPS. Zapisać checkpoint/resume i porównanie CPU/MPS na identycznym wektorze.

Odbiór: Benchmark na rzeczywistym M5 Max 128 GB wybiera profil; manifest podaje sprzęt, wersje i parametry. Samo auto=cpu lub wymuszenie mps nie jest odbiorem optymalizacji.

Przekazanie: Patryk dostaje wersję torch i test zgodności; Wiktor uruchamia właściwy profil lokalnie. Azure wykonuje inferencję i obsługę aplikacji, a nie ten trening.

## M1 Ustalić protokół eksperymentu

Kolejność: D1 rano. Zależności: K1.

1. Zamrozić train/val/test, wersję enkodera i cel wyboru modelu na walidacji. Zachować lineage source/group_id.

2. Sprawdzić że żaden checkpoint, seed ani próg nie jest wybrany po końcowym teście. Wyniki historyczne oznaczyć osobno.

3. Zapisać macierz porównań SNN symulowane, Lu.i fizyczne i Uno FFT z identycznymi strumieniami oraz oddzielnymi granicami energii.

Odbiór: Konfiguracja i manifest pozwalają odtworzyć selekcję; test nie trafia do funkcji fitness ani strojenia progu.

Przekazanie: Kacper otrzymuje protokół budowy continuous; Andrzej listę przebiegów pomiarowych.

## M2 Naprawić final evaluation i eksport

Kolejność: D1. Zależności: M1.

1. W ga_runner.py usunąć podstawianie testu do val_data w run_final_evaluation_stage i etykietę canonical_test_placeholder.

2. Rozdzielić trening, wybór, eksport i eval. Eksport ma serializować wybrany checkpoint, nie uruchamiać train_winner jeszcze raz.

3. Zachować jednocześnie source_commit, seed, dataset_manifest_hash, encoder_hash, checkpoint_hash, decoder i jednostki. Test sprawdza niezmienność wag po eksporcie.

4. Sprawdzić przekazywanie config.train.fitness_seeds: obecny run_ga_stage ma literal fitness_seeds=3. Test ma dowodzić, że profil sprzętowy i konfiguracja eksperymentu nie są ignorowane.

Odbiór: Wywołanie eval/export nie zmienia parametrów ani czasu uczenia; test wykrywa próbę użycia testu do wyboru modelu.

Przekazanie: Patryk dostaje pierwszy prawdziwy model do runtime bez czekania na najlepszy wynik GA.

## M3 Uruchomić kontrolowany trening

Kolejność: D1–D2. Zależności: M0, M2.

1. Wykorzystać trwające GA tylko po potwierdzeniu zgodnego kodu, splitów i enkodera. Nie restartować kosztownego treningu bez potrzeby.

2. W rankingu porównywać zgodne protokoły, kryterium najpierw walidacja i budżet FA, następnie złożoność/energia według wcześniej zapisanej reguły.

3. Zapisywać seed, checkpoint i config atomowo, wspierać resume z jawnego stanu. Nie wymuszać sukcesu przy nieosiągalnym budżecie FA/h.

Odbiór: Champion oznacza najlepszy według z góry określonej walidacji; jeśli budżet jest nieosiągalny, raport mówi infeasible.

Przekazanie: Patryk ma candidate model; Karolina ma manifest oraz historyczne punkty porównania.

## M4 Ocenić ciągły strumień

Kolejność: D2–D3. Zależności: M3, K3.

1. Wczytać continuous manifest Kacpra, zachować stan SNN na granicach okien i odjąć tylko z góry określony warmup/gapy od ekspozycji.

2. Policzyć event recall, FA/h, false-alarm count, czas tła, one-to-one matching i opóźnienie od onset; elapsed treningu nie jest latency detekcji.

3. Zamrożony test uruchomić dopiero po wyborze modelu. Raportować przedział ufności FA/h i różnicę do historycznej metryki na klipach.

Odbiór: Każdy FA da się wskazać na osi czasu. Przy zerowym FA wynik ma dodatnią granicę górną niepewności, a nie obietnicę braku alarmów.

Przekazanie: Karolina dostaje CSV/JSON metryk; Andrzej identyczny zestaw do hardware.

## M5 Przekazać pakiet championa

Kolejność: D3. Zależności: M4, P1.

1. Spakować checkpoint, manifest, topologię, weights, decoder, calibration status i golden replay. Podać SHA256 każdego artefaktu.

2. Sprawdzić powtórne załadowanie w czystym procesie i identyczne wyjście. Duże artefakty przekazać przez uzgodniony storage, nie zwykły commit binariów.

3. PR #47 po poprawkach kierować do master. Po jego squash kolejny etap zaczynać z aktualnego master, żeby nie powielać starej historii.

Odbiór: Patryk i Wiktor wczytują dokładnie oceniony model; plik JSON nie obiecuje fizycznej kompatybilności bez kalibracji.

Przekazanie: Zamrożony raport naukowy, hash modelu i opis ograniczeń dla artykułu.