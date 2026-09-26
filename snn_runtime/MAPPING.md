# Mapowanie eksportu `hw_*.json` na `ModelManifest` (zadanie P1, punkt 1)

Kontrakt `contracts/v1/ModelManifest.schema.json` jest tym, co runtime przyjmuje.
Nasz trening produkuje dziś `hw_*.json` z `snn_hw_pipeline.py export`. Ten dokument
mówi, co się z czym pokrywa, czego brakuje i kto to ma uzupełnić.

Producentem `ModelManifest` jest wg `contracts/README.md` Marcel. Poniższa tabela
jest specyfikacją dla jego eksportera, a nie samym eksporterem.

Sprawdzone na realnym eksporcie `models/hw_v2_recallfa_s0.json` (branch
`fah-metric-kn`), sieć 7 kanałów, 8 płytek, topologia 7-4-3-1, fan-in 3.
Wynik konwersji leży w `tests/runtime/fixtures/lui8-v2-manifest.json`: 8 neuronów,
24 połączenia (12 z kanałów, 12 między neuronami), przechodzi
`contracts.validation.validate` i `snn_runtime.load_manifest`.

Konwerter referencyjny: `snn_runtime/tools/make_reference_manifest.py`. Komenda,
którą powstał fixture, jest na końcu tego pliku.

## 1. Co się przenosi wprost

| Pole kontraktu | Źródło w `hw_*.json` | Przeliczenie |
|---|---|---|
| `runtime.dt_us` | `dt_s` | `× 1e6`, u nas 0.01 s → 10000 |
| `encoder_profile.channel_map[].channel` / `.feature` | `channels[]` | `index` = pozycja na liście |
| `topology.neurons[].neuron_id` | klucz w `boards` | `H0..H3, G0..G2, D` |
| `topology.neurons[].tau_mem_us` | `boards[].tau_mem_ms` | `× 1000` |
| `topology.neurons[].tau_syn_us` | `boards[].tau_syn_ms` | `× 1000` |
| `topology.neurons[].v_leak` | `boards[].v_leak` | bez zmian |
| `topology.neurons[].v_threshold` | `v_th` (globalne) | ta sama wartość dla każdego neuronu |
| `topology.connections[].target_port` | `synapses[].port` | `J1→1, J2→2, J3→3` |
| `topology.connections[].source_id` | `synapses[].from` | `source_kind` = `channel`, gdy nazwa jest w `channels`, inaczej `neuron` |
| `topology.connections[].sign` | `synapses[].sign` | `+ → excitatory`, `- → inhibitory` |
| `topology.connections[].weight` | `synapses[].w_sim` | **wartość bezwzględna**, patrz pułapka P1 |
| `provenance.seed` | `model_provenance.seed` | bez zmian |
| `provenance.dataset_manifest_hash` | `model_provenance.train_data.manifest_sha256` | **dokleić prefiks** `sha256:` |

## 2. Czego w eksporcie nie ma, a kontrakt wymaga

Każdy z tych wierszy to decyzja, którą ktoś musi podjąć. Konwerter referencyjny
świadomie odmawia wymyślania ich za kogoś i żąda ich jako argumentów.

| Pole | Kto decyduje | Uwaga |
|---|---|---|
| `model_id`, `status` | Marcel | `status` to `demo`/`sandbox`/`evaluated_champion`, patrz pułapka P7 |
| `encoder_profile.profile_id`, `provenance` | Kacper (K1) | `provenance` = `measured` dopiero po pomiarze na płytce |
| `encoder_profile.implementation_sha` | Kacper | **40 hex**, pełny SHA commita. Nasz eksport ma skrócony `git_commit: "aaed1647"` |
| `encoder_profile.config_sha256` | Kacper | mamy `encoder_sha256` pliku `encoder_twin.py`, użyłem go z prefiksem; do potwierdzenia, czy to jest to samo, co kontrakt nazywa configiem |
| `encoder_profile.sample_rate_hz`, `hop_samples`, `pulse_width_us` | Kacper | u nas 19231 / 192; `pulse_width_us` nie jest nigdzie w repo, wpisałem 500 jako wartość do potwierdzenia |
| `artifacts[].sha256`, `provenance.checkpoint_hash` | Marcel | eksport podaje ścieżkę `.pt`, nie podaje hasha |
| `runtime.implementation`, `version`, `integrator` | Patryk | `integrator: lui-order2-hard-reset` |
| `runtime.potential_unit`, `neurons[].potential_unit` | Patryk | `a.u.`, patrz pułapka P4 |
| `decoder.*` | **nikt dziś** | reguła decyzji nie jest w ogóle eksportowana, patrz pułapka P5 |
| `topology.topology_version`, `mode` | Marcel | `mode: hardware_compatible` dla sieci, która ma iść na płytki |
| `neurons[].v_reset` | Patryk | `0.0`, LuiNet resetuje twardo do zera |
| `neurons[].refractory_us` | Patryk | `0`, LuiNet nie modeluje refrakcji (`grep refrac snn_hw_pipeline.py` = 0 trafień) |
| `connections[].weight_unit`, `delay_us` | Patryk | `a.u.` i `0` |
| `provenance.training_run_id`, `evaluation_hash`, `calibration_id` | Marcel / Andrzej | brak, patrz pułapka P7 |

## 3. Co mamy, a kontrakt nie ma na to miejsca

`boards[].led_bar_pct`, `boards[].scale_k`, `synapses[].pot_pct`,
`synapses[].pulses_to_fire_100Hz`. To są **nastawy trymerów i pasków LED**, czyli
to, co człowiek ustawia śrubokrętem. `ModelManifest` ma `additionalProperties:
false`, więc nie da się ich tam dołożyć bez zmiany schematu.

Propozycja: nie pchać ich do `ModelManifest`. To jest artefakt kalibracyjny
związany z konkretnym egzemplarzem płytek, a nie z modelem, i ma inny cykl życia
(zmienia się przy każdej wymianie trymera, bez zmiany wag). Powinien być osobnym
plikiem wpisanym w `artifacts[]` ze swoim hashem. Do uzgodnienia z Wiktorem
i Andrzejem (A2).

## 4. Pułapki semantyczne

**P1. `weight` ma `minimum: 0`.** Nasze `w_sim` jest liczbą ze znakiem, a `sign`
niesie tę samą informację drugi raz. Eksporter musi wysłać `abs(w_sim)`, inaczej
schemat odrzuci pakiet. Konwerter dodatkowo sprawdza, czy znak `w_sim` zgadza się
z polem `sign`, bo w eksporcie są to dwa niezależne pola i mogą się rozjechać.

**P2. `v_reset < v_threshold` jest wymagane semantycznie** (`contracts/validation.py`).
U nas `v_reset = 0.0`, `v_threshold = 1.0`, więc przechodzi, ale eksport nie ma
dziś pola `v_reset` w ogóle.

**P3. `dt_us` i profil enkodera to dwa niezależne pola mówiące to samo.**
`runtime.dt_us` = 10000 µs, a `hop_samples / sample_rate_hz = 192 / 19231` daje
9984 µs. Rozjazd 0,16 %, czyli około jednej sekundy dryfu na dziesięć minut
strumienia. Kontrakt nie wymusza ich zgodności, więc robi to runtime
(`DT_ENCODER_MISMATCH`, tolerancja 1 % w `snn_runtime/units.py`). Który z nich jest
prawdziwy, rozstrzyga K1: albo enkoder ma wysyłać ramkę co 10,000 µs, albo trening
ma używać `dt = 9.984` ms.

**P4. `potential_unit` to nie kosmetyka.** Nasze potencjały są ułamkami `V_th`
(`V_TH = 1.0`, próg sprzętowo VDD/2), czyli `a.u.`, nie wolty. Runtime raportuje
wtedy `score_kind: "uncalibrated"` w `SNNDecision`. Pakiet, który deklaruje `"V"`
bez `provenance.calibration_id`, jest odrzucany (`CALIBRATION_MISSING`). Przejście
na wolty wymaga pomiaru Andrzeja (A2), nie zmiany w eksporterze.

**P5. Reguła decyzji nie jest eksportowana.** `decoder.window_us`, `threshold`
i `cooldown_us` nie mają odpowiednika w `hw_*.json`. Dziś reguła `k:w` żyje wyłącznie
jako argument CLI `eval_stream.py --rules` i stała `DEFAULT_REFRAC = 500` ramek
w `snn_pipeline/stream_eval.py`. To znaczy, że **model i reguła, przy której zmierzono
jego FA/h, są dziś przechowywane osobno i nic ich nie wiąże.** W fixture wpisałem
`k=1`, okno 1 ramki, cooldown 500 ramek (5 s). Eksporter musi zapisywać tę regułę
razem z modelem, bo bez niej liczba FA/h nie jest odtwarzalna.

**P6. Kontrakt nie ma pola na neuron decyzyjny.** Jest `decoder`, ale nic nie mówi,
czyje spiki dekoduje. Runtime wnioskuje to jako jedyny neuron bez połączeń
wychodzących (u nas `D`) i odmawia, gdy takich neuronów jest zero albo więcej niż
jeden (`NO_DECISION_NEURON`, `AMBIGUOUS_DECISION_NEURON`). To jest obejście.
Właściwe rozwiązanie to pole `topology.decision_neuron` w wersji 1.1 kontraktu.
**Do zgłoszenia Wiktorowi.**

**P7. Żaden nasz dzisiejszy model nie może być `evaluated_champion`.** Kontrakt
wymaga wtedy niepustych `training_run_id`, `dataset_manifest_hash`,
`evaluation_hash` i `seed`. Mamy `seed` i `dataset_manifest_hash`, nie mamy
`training_run_id` ani `evaluation_hash`. Dlatego fixture jest `sandbox`, a nie
championem. Test `test_our_best_model_cannot_yet_claim_champion` pilnuje, żeby to
nie przeszło przypadkiem.

**P8. Limity, w których się mieścimy z zapasem.** Kontrakt dopuszcza do 50 neuronów,
150 połączeń, 64 kanały i `target_port` 1..3. Mamy 8 / 24 / 7 / 3. Fan-in 3 jest
w schemacie wymuszony wprost (`target_port` maks. 3 plus reguła „jeden wpis na port"),
więc fizyczne ograniczenie płytki Lu.i jest już częścią kontraktu.

## 5. Jak odtworzyć fixture

```sh
.venv-w0/bin/python -m snn_runtime.tools.make_reference_manifest \
    --export models/hw_v2_recallfa_s0.json \
    --out tests/runtime/fixtures/lui8-v2-manifest.json \
    --model-id lui8-v2-recallfa-s0 --status sandbox \
    --artifact-path v2_recallfa_s0.pt --checkpoint models/v2_recallfa_s0.pt \
    --profile-id encoder-v2-atmega328p --profile-provenance reference \
    --encoder-sha aaed1647ce3ed394035bafb8d60ab0ac657f9f34 \
    --encoder-config-sha256 sha256:1be666b59b91cdc8141f033111eb9b24369dd202c93244df815d9993aae79763 \
    --topology-version lui8-7-4-3-1 \
    --dataset-manifest-hash sha256:b3dcc1101a6522ec567ff210e5a943f7aa606d4dc5b01e3e67de73ed70227cd3 \
    --seed 0
```

`models/hw_v2_recallfa_s0.json` i `models/v2_recallfa_s0.pt` są na branchu
`fah-metric-kn`, który nie jest jeszcze scalony do `master`.

## 6. Co P2 zmieniło w tej tabeli

Do P1 powyższe braki były opisem. Od P2 runtime naprawdę całkuje sieć i naprawdę
decyduje, więc trzy z nich przestały być notatką, a stały się liczbą, która
działa na produkcji.

**`decoder.*` nie jest już kosmetyczny.** Konwerter wpisuje dziś placeholdery:
`threshold = 1`, `window_us = 1 ramka`, `cooldown_us = 5 s` (za stałą
`DEFAULT_REFRAC = 500` w `stream_eval.py`). To znaczy „alarmuj na pierwszy spike
D i milcz przez pięć sekund", czyli w najgorszym razie **720 alarmów na godzinę**.
Operacyjny punkt pracy z `models/WYNIKI.md` był mierzony przy zupełnie innej
regule i tego związku nic dziś nie przechowuje. Dopóki Marcel nie eksportuje
`decoder.*` razem z wagami, manifest i zmierzone FA/h opisują dwa różne
detektory. To jest najpoważniejsza pozycja z tej tabeli.

**`neurons[].refractory_us` jest teraz wykonywane.** Runtime zaokrągla refrakcję
w górę do pełnych ramek i w tym czasie trzyma neuron przy `v_reset`. Przy `0`,
czyli tym, co dziś eksportujemy, nic się nie zmienia, ale jeśli A2 zmierzy
niezerową refrakcję na płytce, ta wartość zacznie zmieniać decyzje.

**`connections[].delay_us` musi leżeć na siatce ramki.** Runtime odmawia
pakietu z opóźnieniem, które nie jest wielokrotnością `dt_us`
(`DELAY_NOT_ON_GRID`), i odmawia pętli zbudowanej z samych połączeń
bezopóźnieniowych (`CYCLIC_TOPOLOGY`), bo taka ramka nie ma kolejności
wyliczania. Nasza sieć jest jednokierunkowa i ma same zera, więc to nic dziś
nie kosztuje; ogranicza za to edytor topologii z P3, jeśli Karolina ma pozwalać
na rekurencję.

Jedna rzecz, której runtime świadomie **nie** robi: nie wymyśla rozgrzewki
enkodera. Sesja startuje z membraną w spoczynku, co jest stanem prawdziwym, a
nie założeniem, więc pierwszy batch jest od razu `valid`. Jeżeli enkoder na
ATmedze potrzebuje N ramek, zanim jego `floor`/`MAD` coś znaczą (a wygląda na
to, że potrzebuje), to jest własność enkodera i musi trafić do
`EncoderProfile`, a nie zostać zgadnięta tutaj. To pytanie do K1.
