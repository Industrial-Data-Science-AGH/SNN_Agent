---

## CI/CD — stan aktualny i co robimy dalej

Hej, piszę żebyście wiedzieli jak działa nasz pipeline i co każdy musi zrobić żeby artefakty z waszego komponentu pojawiały się automatycznie na GitHubie.

---

### Jak to działa ogólnie

Każdy push na `dev`, `feat/**`, `integration/**` oraz każdy PR na `dev`/`main` odpala automatycznie pipeline na GitHub Actions. Pipeline składa się z 4 jobów:

```
lint → test → collect-component-artifacts
                └→ simulation (tylko integration/** i PR)
```

Na końcu każdego runu w zakładce **Actions → wasz run → Artifacts** pojawia się paczka `component-outputs-<numer>` z plikami z całego pipeline'u posegregowanymi per komponent.

---

### Struktura artefaktów

Pipeline zbiera outputy do tej struktury:

```
ci_artifacts/
├── dataset/       ← z data/
├── encoder/       ← z encoder/encoder_output/
├── network/       ← z output/
├── agents/        ← z software/agent_output/
├── logs/          ← z logs/
├── simulation/    ← pliki z output/ po odpaleniu run_pipeline.py
└── summary/
    └── manifest.md   ← lista zebranych plików, branch, commit SHA
```

Jeśli jakiś folder nie istnieje — pojawia się `README.txt` z info że brakuje outputu. Artefakty zbierają się **zawsze**, nawet jeśli testy nie przejdą.

---

### Co już działa

- lint (ruff) ✓
- testy jednostkowe — `test_ci_basics.py`, `test_decoder.py` ✓
- zbieranie `encoder/encoder_output/` (`negative.csv`, `positive.csv`) ✓
- manifest z branch + commit SHA + lista plików ✓

---

### Czego brakuje i co każdy powinien zrobić

**Enkoder**
Encoder output już działa. Upewnijcie się tylko że wyniki zawsze lądują w `encoder/encoder_output/` i są commitowane — ten folder jest czytany przez CI.

**Sieć / snn_pipeline**
Output z sieci powinien trafiać do katalogu `output/` — ten folder jest gitignorowany, więc pliki muszą być generowane podczas runu CI. Żeby to działało, `snn_pipeline/run_pipeline.py` musi dać się odpalić bez danych ESC-50 (np. na mocku) i bez GPU. Aktualnie `torch` nie jest w zależnościach dev — trzeba go dodać do `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "numpy>=2.4.2",
    "pytest>=9.0.2",
    "ruff>=0.15.2",
    "torch>=2.0.0",   # ← dodać
]
```

**Agenci**
Outputy agentów powinny trafiać do `software/agent_output/`. Ten folder aktualnie nie istnieje. Jak ktoś implementuje warstwę agentów — niech zapisuje wyniki tam.

**Testy**
Każdy komponent powinien mieć swój plik testowy w `tests/`. Aktualnie mamy tylko `test_ci_basics.py` i `test_decoder.py`. Brakuje:
- `tests/test_encoder.py`
- `tests/test_network.py`
- `tests/test_agents.py`

Format wyjścia każdego komponentu jest zdefiniowany w `docs/ci-cd-plan.md` — tam są też przykładowe asserty które powinny znaleźć się w testach.

---

### Gdzie żyje ci.yml i kiedy go zmieniać

Plik `.github/workflows/ci.yml` musi być na branchu `main` żeby działał dla wszystkich branchy. Aktualnie jest na `integration/ci-cd-test` — trzeba go zmergować do `main` przez PR.

Workflow **nie merguje automatycznie** żadnego kodu. CI zbiera dowody (testy, artefakty, outputy), decyzję o PR na `dev` lub `main` podejmuje człowiek.

---

### TL;DR — co każdy musi zrobić

| Komponent | Co zrobić |
|---|---|
| Enkoder | Upewnić się że output ląduje w `encoder/encoder_output/` i jest commitowany |
| Sieć | Dodać `torch` do dev deps, `run_pipeline.py` musi działać na mocku bez GPU, output do `output/` |
| Agenci | Stworzyć `software/agent_output/`, zapisywać tam wyniki |
| Wszyscy | Dopisać testy do `tests/test_<komponent>.py` zgodnie z kontraktem z `docs/ci-cd-plan.md` |
| Ktokolwiek | ci.yml z `feat/ci-cd-pipeline` na `main` żeby CI działało globalnie |

---