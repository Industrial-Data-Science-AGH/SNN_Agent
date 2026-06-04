# Project Documentation Index

Start Here

- [README.md](README.md): analysis bundle overview and quick commands.
- [project_analysis.md](project_analysis.md): full source-level analysis.
- [../docs/Projekt Neuromorficzny Wake-Up AI.pdf](../docs/Projekt%20Neuromorficzny%20Wake-Up%20AI.pdf):
  original Polish feasibility/business/architecture report.

## Existing Hardware Notes

- [../docs/backplane_design.md](../docs/backplane_design.md): backplane topology,
  connector assumptions, grounding, routing, and test-point notes for 3-5 analog
  neurons.
- [../docs/synapse_design.md](../docs/synapse_design.md): resistor-divider
  synapse model, resistor guidance, leakage notes, and configurable `R_syn`
  recommendation.
- [../docs/spi_protocol.md](../docs/spi_protocol.md): planned Raspberry Pi /
  host SPI trigger packet. Current Arduino encoder code does not yet implement
  this packet.
- [../docs/lu_i_hardware_bought.md](../docs/lu_i_hardware_bought.md): external
  Lu.i electronic neuron reference note. Images referenced by that note are not
  vendored here.

## Firmware And Test Notes

- [arduino_encoder_plan.md](arduino_encoder_plan.md): corrected current-state
  notes for the Arduino encoder/decoder prototype.
- [../encoder/ENCODER_PLAN.md](../encoder/ENCODER_PLAN.md): existing encoder
  plan in the repository.
- [../encoder/snn_encoder/snn_encoder.ino](../encoder/snn_encoder/snn_encoder.ino):
  microphone amplitude to GPIO spike prototype.
- [../encoder/snn_decoder/snn_decoder.ino](../encoder/snn_decoder/snn_decoder.ino):
  GPIO spike counter prototype.
- [../tests/neuron_characterization_20260322.md](../tests/neuron_characterization_20260322.md):
  manual analog-neuron characterization sheet.

## Generated Or Missing Artifacts

The training/export pipeline writes runtime artifacts under ignored directories
such as `data/`, `output/`, and `checkpoints/`. Expected generated files include:

- `output/weights.json`
- `output/weights.csv`
- `output/snn_config.h`
- `output/mcp4151_table.csv`
- `output/hat_learning_curves.png`
- `output/sensitivity_heatmap.png`

There is currently no maintained tracked BOM file. `docs/BOM.md` has been
removed in the working tree.
