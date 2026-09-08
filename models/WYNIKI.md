# WYNIKI — trening pipeline na czystym benchmarku (spikes_v2 / v2.0.0)

Dziennik przebiegów `snn_hw_pipeline train` na **czystym** zbiorze (artefakt
`spikes_v2` zbudowany z `dataset/versions/v2.0.0`, bez przecieku, 771 grup w teście).
Metryka decyzyjna: **recall @ budżet FA/h** (`snn_pipeline/stream_eval.py`), dekoder
**k=1** spójnie. Próg akceptacji: `docs/DATASET_CONTRACT.md`
(recall ≥ 0.70 @ ≤ 6 FA/h na test, per-kind).

> Jak czytać: notuj zawsze **(artefakt, k/reguła, metryka)**. Nie mieszaj z
> `spikes_manifest7` (WYCOFANE, przeciek) ani ze `spikes_ext` (poligon selekcji cech).

---

## Wiersz 1 — 2026-09-08 — baseline recall_fa, seed 0

- **Artefakt:** `spikes_v2` (manifest `b3dcc110…`, enkoder `1be666b5…`, v2.0.0,
  10853/10853 plików, walidator ZDANE, K10 OK: 90% pozytywów w 63/96 grup, max 2%).
- **Trening:** `--select-metric recall_fa --stream-budget 6 --epochs 120 --seed 0`
  (domyślnie **`spk_w=0.0`**, `pos_weight=3.0`, HAT 0–47 / QAT 48–119).
- **Pliki:** `models/v2_recallfa_s0.pt`, `models/hw_v2_recallfa_s0.json`,
  `models/train_v2_recallfa_s0.csv`.

### Metryka DECYZYJNA (test, 884 szkło / 983 tło / 2,76 h tła)

| reguła | recall (wykryte szkło) | fałszywe alarmy/h | FA datasec / esc50 / voice |
|---|---|---|---|
| k=1 (okno 10 ms) | **86,3%** | **219/h** | 51% / 37% / 29% |
| k=2 (okno 1 s) | 75,3% | 155/h | 41% / 25% / 19% |
| k=3 (okno 2,5 s) | 69,3% | 139/h | 38% / 24% / 12% |

**recall @ 1 FA/h = 0,000  ·  recall @ 6 FA/h = 0,000  → BUDŻET NIEOSIĄGALNY.**
Neuron D: śr. 21,3 spików/klip na szkle, **8,9 na tle**.

### Metryki pomocnicze (ramkowe / okienkowe)
- best (ep 24, QAT), pełna walidacja: rec 0,943 · prec 0,525 · f1 0,674
- test: rec 0,890 · prec 0,456 · **f1 0,603**

### Wniosek
Model **świetnie łapie szkło (86%)**, ale robi **~140–220 fałszywych alarmów/h**
— **~30× ponad budżet 6/h**. W tej formie **nieużywalny**. Ostrzejszy dekoder nie
ratuje (k=3: wciąż 139/h, recall spada do 69%) → **spór k=1 vs k=2 bezprzedmiotowy,
dopóki nie ściągniemy strzelania na tle**. Problem to **precyzja** (0,46), nie recall.

**Uwaga metodyczna:** recall@6FA/h = 0 przez cały trening → selekcja checkpointu po
tej metryce była **jałowa** (brak sygnału). Do fazy „jeszcze poza budżetem" trzeba
miększego sygnału selekcji (patrz rekomendacje #5).

---

## Rekomendacje — co możemy poprawić (priorytetowo)

1. **[najtańsze, najcelniejsze] Włączyć człon spikowy w treningu: `--spk-w 0.5`.**
   Pipeline trenuje z `spk_w=0`, więc model nie ma ŻADNEJ presji, by milczeć na tle
   — stąd 200 FA/h. Człon spikowy karze spiki D na tle i wymusza serię na szkle.
   GA już to ma (`spk_w=0.5`). To pierwszy eksperyment: dotrenować i zmierzyć FA/h.
2. **Celować w PRECYZJĘ, nie recall.** recall 86% jest aż za wysoki, precyzja 0,46
   za niska. Zejść z `--pos-weight` (dziś 3,0; przy 9,5% pozytywów za mocno pcha
   recall) — sweep np. 1,0 / 1,5 / 2,0 i patrzeć na FA/h, nie na F1.
3. **Twardszy próg neuronu D bez zmian w treningu:** `eval_stream --d-leak-delta`
   obniża V_leak D (fizycznie: niższy pasek LED na płytce D). Szybki test, o ile
   spada FA/h przy sztywniejszym D — zero kosztu treningu.
4. **Zmienić metrykę selekcji na fazę „poza budżetem".** Gdy recall@budget=0,
   selekcja nie ma sygnału. Dodać fallback: „minimalne FA/h przy recall ≥ próg"
   albo „recall przy najniższym osiągalnym FA/h", żeby checkpoint nie był losowy.
5. **Regularyzacja aktywności / rzadkość na tle.** Rozważyć wyższą karę za średnią
   aktywność warstw na klipach tła (dziś `reg` w `loss_fn` jest łagodny).
6. **[dopiero po 1–3] GA z metryką `recall_fa`.** Jeśli trening nie ściągnie FA/h,
   obecna topologia 7→4→3→1 może nie mieć pojemności, by odróżnić trudne tło
   (datasec FA 51%). Wtedy `run_search --metric recall_fa` poszuka lepszej topologii.

**Kolejność:** najpierw #1 (`--spk-w 0.5`) — tanie i wprost w sedno. Potem #3
(`--d-leak-delta`, zero treningu) i #2 (sweep `pos_weight`). #4 to poprawka kodu.
