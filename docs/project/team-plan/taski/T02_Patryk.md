# T02 Patryk

Stanowy symulator Lu.i i kontrakt cyfrowego bliźniaka

Branch: feat/patryk-lui-runtime

Reviewer: Marcel dla modelu; Andrzej dla parametrów fizycznych

Zakres plików: snn_runtime/** (nowy); tests/runtime/**; contracts/model*.json i contracts/neuron*.json po uzgodnieniu z Wiktorem

Punkt startowy: ga_neuron_search/net.py; snn_pipeline/snn_model.py; architecture_14_neurons_patryk_09_07/snn_hw_pipeline.py; dokumentacja Lu.i

## P1 Zdefiniować model i wejście runtime

Kolejność: D1 rano. Zależności: W0, K1.

1. Wybrać jeden adapter pakietu modelu; wymagać topology, weights, dt, reset, refractory, tau i decoder, bez ukrytych domyślnych wartości.

2. Zmapować kanały wejściowe na porty neuronów, odróżnić trzy fizyczne wejścia Lu.i od liczby cech enkodera.

3. Opisać jednostki napięcia/czasu i przejście z parametrów treningu na model referencyjny. Pozostawić oznaczenie uncalibrated do pomiarów Andrzeja.

Odbiór: Niekompletny model, niezgodny encoder_hash i błędny port kończą się kontrolowanym błędem przed Start.

Przekazanie: Marcel zna format eksportu; Karolina zna ID neuronów i pól.

## P2 Stanowy streaming i dekoder

Kolejność: D1–D2. Zależności: P1.

1. Wydzielić Runtime.load/reset/step/checkpoint bez zależności od UI. Utrzymywać Vmem, prądy synaps i historię dekodera pomiędzy batchami.

2. Zaimplementować kolejność po seq i source time, duplikaty, jawne luki oraz session_epoch. Nie resetować stanu na każde HTTP.

3. Dostarczyć Wiktorowi adapter route SNN i wynik zawierający decision_id, source_time, model_hash. Nie tworzyć równoległego drugiego API.

Odbiór: Ten sam strumień podzielony na różne batch sizes daje zgodne spike’y i decyzje w ustalonej tolerancji; luka generuje gap/warmup.

Przekazanie: Wiktor może podłączyć prawdziwy bridge; Karolina może odtworzyć krótką sesję.

## P3 Telemetria i edytowalna topologia

Kolejność: D2. Zależności: P2.

1. Emitować NeuronFrame z Vmem, progami, spike timestamp, stanem synaps, mode i calibration_id. Określić downsampling wykresu niezależnie od czasu symulacji.

2. Zapewnić snapshot i zdarzenia z rosnącym event sequence do live/replay. Telemetria UI może pomijać klatki, lecz nie ukrywać luk w danych wejściowych.

3. Walidować draft topologii 0–50: 0 oznacza pusty edytor, nie działający klasyfikator. Zmiana liczby neuronów nie przebudowuje po cichu championa; nowa sesja wymaga nowego ważnego modelu.

Odbiór: Karolina widzi wierny raster i potencjał na znanym wektorze testowym; Pause view nie zatrzymuje backendu.

Przekazanie: Udokumentowane mapowanie LED oraz przykład 8 i 50 neuronów.

## P4 Porównać z fizycznym Lu.i

Kolejność: D2–D3. Zależności: P2, A2.

1. Odtworzyć wejściowe impulsy pomiarów Andrzeja z amplitudą i szerokością; uwzględnić ich znaczenie dla wagi.

2. Dopasować parametry na zbiorze kalibracyjnym i porównać na oddzielnym przebiegu: czasy spike, przebieg Vmem, nasycenie i reset.

3. Raportować błąd oraz zakres ważności; bez pomiarów oznaczyć model functional simulation, nie zweryfikowany twin 1:1.

Odbiór: Raport zawiera identyfikatory płytek, nastawy, warunki i tolerancje ustalone przed oceną; błąd nie jest ukryty średnią.

Przekazanie: Karolina wyświetla calibration status; Andrzej potwierdza zgodność warunków.

## P5 Zamrożony runtime i golden replay

Kolejność: D3. Zależności: P3, M5.

1. Wczytać dokładny checkpoint Marcela bez dodatkowego treningu i porównać decoder output z eksportem.

2. Zapisać krótki golden replay z przewidywanym rastrem, decyzją i tolerancją. Uruchomić go lokalnie i w kontenerze Azure.

3. Przekazać lock zależności i wynik testu reset/restart/duplicate; trening nie może działać w runtime request path.

Odbiór: Hash modelu pozostaje taki sam od oceny do Azure. Golden replay przechodzi w obu środowiskach.

Przekazanie: Wiktor dostaje kontrakt uruchomienia; zespół jeden identyfikator championa.