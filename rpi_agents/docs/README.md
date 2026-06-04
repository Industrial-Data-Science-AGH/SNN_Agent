# Wake-Up AI RPi Agents Documentation Bundle

This folder contains the project-analysis documentation from the repository
review performed on 2026-06-04. The docs are intentionally kept here instead of
the repository root.

Wake-Up AI is a research/prototype repository for an ultra-low-power
"analog trigger, digital brain" edge-AI system. The current codebase is
centered on glass-break detection: audio is converted into spike trains,
a small spiking neural network is trained with hardware constraints, and
the resulting weights are exported into PCB-friendly resistor and
microcontroller formats.

## Docs In This Folder

- [project_analysis.md](project_analysis.md): full source-level analysis of the
  current repository.
- [docs_index.md](docs_index.md): index of existing project docs and generated
  artifacts.
- [arduino_encoder_plan.md](arduino_encoder_plan.md): corrected notes for the
  checked-in GPIO-based Arduino encoder prototype.
- [web_dashboard_ui.md](web_dashboard_ui.md): current state of the dashboard UI.
- [spi_protocol_status.md](spi_protocol_status.md): status note for the planned
  2-byte SPI trigger packet.

## Current Repository Map

```text
../snn_pipeline/                 Main HAT/QAT SNN training and export pipeline
  config.py                      Hardware, audio, training, dataset, and path constants
  data_pipeline.py               ESC-50 download, preprocessing, augmentation, DataLoaders
  spike_encoders.py              Rate coding and TTFS spike encoders
  snn_model.py                   4-neuron glass-break SNN using snntorch
  losses.py                      Hardware-aware loss with F1, precision, recall, E24 terms
  hat_trainer.py                 Hardware-aware training with Gumbel/E24 quantization
  qat_trainer.py                 Quantization-aware fine-tuning
  evaluation.py                  Metrics, thermal drift, power estimate, benchmark table
  export.py                      JSON, CSV, and Arduino header export
  hil_validation.py              HIL tolerance simulation and MCP4151 programming table

../encoder/                      Arduino prototype sketches and encoder plan
../docs/                         Hardware notes, SPI protocol, proposal PDF
../software/web_dashboard/api/   Minimal FastAPI health endpoint
../software/web_dashboard/ui/    Vite/React dashboard shell
../software/host_agent/          Mock agent and toy Chroma RAG scripts
../software/infra/               Experimental Docker assets
../snn_simulator/                Older simulator sandbox, not the main pipeline
../tests/                        Manual hardware characterization template
```

## Quick Setup Notes

The project declares Python 3.12 in `../.python-version` and
`../pyproject.toml`. ML dependencies are currently in `../requirements.txt`.

```bash
cd ..
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small forward-pass smoke test:

```bash
python test_forward.py
```

Run the main training pipeline:

```bash
python -m snn_pipeline.run_pipeline --phase all
```

Useful shorter runs:

```bash
python -m snn_pipeline.run_pipeline --phase A --data-limit 20
python -m snn_pipeline.run_pipeline --phase all --quick
python -m snn_pipeline.run_pipeline --phase B --epochs 5
```

Phase A can download ESC-50 from GitHub. The dataset is about 600 MB and
requires network access.

## Known Project State

- `pyproject.toml` does not list the ML dependencies yet. Use
  `requirements.txt` for now.
- `software/infra/docker-compose.yml` and its Dockerfiles still reference
  `../backend` and `../frontend`, while the actual dashboard paths are
  `software/web_dashboard/api` and `software/web_dashboard/ui`.
- `snn_simulator/main.py` references a missing `models.snn_torch` module and
  appears to be an older sandbox. Use `snn_pipeline/` for active work.
- `debug_hat.txt` records a previous quick HAT run with recall `0.0`; treat it
  as a debugging artifact, not a validated model result.
- There is no maintained tracked BOM file in the current tree. `docs/BOM.md`
  has been removed in the working tree, and `docs/lu_i_hardware_bought.md`
  is currently the only hardware-purchase/reference note present.

