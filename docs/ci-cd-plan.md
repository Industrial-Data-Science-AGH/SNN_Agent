# Pipeline Contract

## 1. Purpose

This document defines the data contract between the main components of the SNN Agent pipeline.

The goal is to make every pipeline stage return a predictable output format, so that the next stage can consume it without guessing the structure.

The planned data flow is:

```text
dataset → encoder → SNN/network → network output → agents → metrics/report
```

This contract is especially important for CI/CD, because each component can be tested independently and its output can be collected automatically as a GitHub Actions artifact.

---

## 2. General pipeline stages

| Stage          | Responsibility                                | Expected output          |
| -------------- | --------------------------------------------- | ------------------------ |
| Dataset        | Provides raw or preprocessed input data       | Dataset sample           |
| Encoder        | Converts input data into spike representation | Encoded spike train      |
| SNN/network    | Runs inference/training/simulation            | Network output           |
| Agent layer    | Uses network result for higher-level decision | Agent decision           |
| Metrics/report | Evaluates output and produces summary         | Metrics and report files |

---

# 3. Dataset output

## 3.1 Responsibility

The dataset stage provides input data for the encoder.

It may contain:

* raw signal samples,
* already preprocessed features,
* labels,
* metadata required for further processing.

In the repository, dataset-related files may be stored in:

```text
data/
dataset/
encoder/encoder_tests/
```

Depending on the branch, dataset files may also be generated during experiments.

## 3.2 Expected output format

A single dataset sample should follow this structure:

```json
{
  "sample_id": "sample_001",
  "signal": [0.12, 0.18, 0.31, 0.24],
  "sampling_rate": 16000,
  "label": 1,
  "metadata": {
    "source": "mock_dataset",
    "description": "example input sample"
  }
}
```

## 3.3 Field description

| Field           | Type        | Description                       |
| --------------- | ----------- | --------------------------------- |
| `sample_id`     | string      | Unique sample identifier          |
| `signal`        | list[float] | Raw signal or feature vector      |
| `sampling_rate` | int/null    | Sampling frequency, if applicable |
| `label`         | int/string  | Ground-truth label                |
| `metadata`      | object      | Optional additional information   |

## 3.4 Contract rules

The dataset output must satisfy:

```text
sample_id must not be empty
signal must not be empty
label must be present
signal values must be numeric
```

---

# 4. Encoder output

## 4.1 Responsibility

The encoder converts dataset input into a spike-based representation that can be consumed by the SNN/network stage.

The encoder is responsible for translating continuous or discrete input data into spike trains.

For the current mockup, the target network size is:

```text
7 neurons
```

## 4.2 Repository location

Encoder-related files may be stored in:

```text
encoder/
encoder/encoder_output/
snn_pipeline/spike_encoders.py
experiments/encoder_sim.py
```

The current encoder output should be collected from:

```text
encoder/encoder_output/
```

## 4.3 Expected output format

```json
{
  "sample_id": "sample_001",
  "spikes": [
    [0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0]
  ],
  "n_neurons": 7,
  "n_timesteps": 5,
  "encoding": "rate",
  "label": 1
}
```

## 4.4 Field description

| Field         | Type            | Description                                                 |
| ------------- | --------------- | ----------------------------------------------------------- |
| `sample_id`   | string          | ID copied from dataset sample                               |
| `spikes`      | list[list[int]] | Spike matrix                                                |
| `n_neurons`   | int             | Number of encoded neurons                                   |
| `n_timesteps` | int             | Number of time steps                                        |
| `encoding`    | string          | Encoding method, for example `rate`, `latency`, `threshold` |
| `label`       | int/string      | Ground-truth label passed further                           |

## 4.5 Contract rules

The encoder output must satisfy:

```text
n_neurons == 7
len(spikes) == 7
all spike rows have the same length
spike values must be only 0 or 1
n_timesteps must match spike row length
sample_id must be preserved from dataset input
label must be preserved for metrics
```

## 4.6 Example CI checks

```python
assert encoder_output["n_neurons"] == 7
assert len(encoder_output["spikes"]) == 7
assert all(value in [0, 1] for row in encoder_output["spikes"] for value in row)
```

---

# 5. SNN/network output

## 5.1 Responsibility

The SNN/network stage receives encoded spike trains and performs inference, simulation or training.

The network stage should not depend on raw dataset format. It should consume only the encoder output.

## 5.2 Repository location

Network-related files may be stored in:

```text
snn_pipeline/
snn_pipeline/snn_model.py
snn_pipeline/run_pipeline.py
snn_simulator/
```

Generated network outputs may be stored in:

```text
output/
```

## 5.3 Expected output format

```json
{
  "sample_id": "sample_001",
  "neuron_spike_counts": [2, 2, 2, 1, 2, 2, 1],
  "output_neuron": 4,
  "decision_score": 0.82,
  "triggered": true,
  "label": 1
}
```

## 5.4 Field description

| Field                 | Type       | Description                                 |
| --------------------- | ---------- | ------------------------------------------- |
| `sample_id`           | string     | ID copied from encoder output               |
| `neuron_spike_counts` | list[int]  | Number of spikes produced by each neuron    |
| `output_neuron`       | int        | Selected/winning output neuron              |
| `decision_score`      | float      | Confidence or activation score              |
| `triggered`           | bool       | Whether the network triggered wake-up logic |
| `label`               | int/string | Ground-truth label passed further           |

## 5.5 Contract rules

The SNN/network output must satisfy:

```text
sample_id must be preserved
neuron_spike_counts must contain 7 values
output_neuron must be within valid neuron index range
decision_score should be between 0.0 and 1.0
triggered must be boolean
label must be preserved for metrics
```

## 5.6 Example CI checks

```python
assert len(network_output["neuron_spike_counts"]) == 7
assert 0 <= network_output["output_neuron"] < 7
assert 0.0 <= network_output["decision_score"] <= 1.0
assert isinstance(network_output["triggered"], bool)
```

---

# 6. Agent output

## 6.1 Responsibility

The agent layer receives the network output and converts it into a higher-level system decision.

The agent does not need to know the full internal network state. It should operate on the network decision, confidence score and trigger status.

In the target architecture, this stage may represent:

* wake-up decision,
* event classification,
* software agent reaction,
* future LLM/VLM processing,
* notification or report generation.

## 6.2 Repository location

Agent-related files may be stored in:

```text
software/
software/agent_output/
```

If the agent layer is not implemented yet, GitHub Actions should still prepare an empty artifact folder with a README file.

## 6.3 Expected output format

```json
{
  "sample_id": "sample_001",
  "agent_decision": "wake_up",
  "predicted_label": 1,
  "confidence": 0.82,
  "triggered": true,
  "action": "activate_reactor",
  "label": 1
}
```

## 6.4 Field description

| Field             | Type       | Description                                            |
| ----------------- | ---------- | ------------------------------------------------------ |
| `sample_id`       | string     | ID copied from network output                          |
| `agent_decision`  | string     | High-level decision, for example `wake_up` or `ignore` |
| `predicted_label` | int/string | Final predicted class                                  |
| `confidence`      | float      | Confidence score                                       |
| `triggered`       | bool       | Whether the system should be activated                 |
| `action`          | string     | Suggested next action                                  |
| `label`           | int/string | Ground-truth label passed to metrics                   |

## 6.5 Contract rules

The agent output must satisfy:

```text
sample_id must be preserved
agent_decision must be one of: wake_up, ignore
confidence should be between 0.0 and 1.0
triggered must be boolean
predicted_label must be present
label must be preserved for metrics
```

## 6.6 Example decision logic

```text
if confidence >= 0.80 and triggered == true:
    agent_decision = "wake_up"
    action = "activate_reactor"
else:
    agent_decision = "ignore"
    action = "stay_idle"
```

---

# 7. Metrics and report output

## 7.1 Responsibility

The metrics/report stage evaluates the pipeline output.

It compares predictions with ground-truth labels and generates technical metrics, summaries and files that can be collected by CI/CD.

This stage supports human decision-making. It should not automatically merge branches.

## 7.2 Repository location

Metrics and reporting logic may be stored in:

```text
snn_pipeline/metrics.py
snn_pipeline/evaluation.py
snn_pipeline/export.py
```

Reports and generated files may be stored in:

```text
reports/
output/
ci_artifacts/
```

## 7.3 Expected metrics output

```json
{
  "accuracy": 0.91,
  "precision": 0.88,
  "recall": 0.86,
  "false_positives": 3,
  "false_negatives": 2,
  "avg_latency_ms": 14.5,
  "hardware_ready": true,
  "passed": true
}
```

## 7.4 Field description

| Field             | Type  | Description                                  |
| ----------------- | ----- | -------------------------------------------- |
| `accuracy`        | float | Overall classification quality               |
| `precision`       | float | How many detected events were correct        |
| `recall`          | float | How many real events were detected           |
| `false_positives` | int   | Number of false alarms                       |
| `false_negatives` | int   | Number of missed events                      |
| `avg_latency_ms`  | float | Average decision latency                     |
| `hardware_ready`  | bool  | Whether output is valid for hardware mapping |
| `passed`          | bool  | Whether quality gates were satisfied         |

## 7.5 Expected report output

```json
{
  "run_id": "github-actions-run-id",
  "branch": "integration/ci-cd-test",
  "commit": "commit-sha",
  "status": "PASS",
  "metrics_file": "reports/metrics.json",
  "summary_file": "reports/summary.md",
  "artifacts_dir": "ci_artifacts/"
}
```

## 7.6 Quality gates

Planned quality gates:

```text
all lint checks pass
all unit tests pass
pipeline simulation finishes successfully
recall >= 0.85
precision >= 0.80
false_positives <= accepted threshold
hardware_ready == true
```

At the current stage, quality gates should support human review. They should not automatically merge code into `dev` or `main`.

---

# 8. Hardware-aware output

## 8.1 Responsibility

The hardware-aware stage validates whether trained or simulated network parameters can be mapped to physical hardware constraints.

In this project, important hardware constraints include:

* E24 resistor mapping,
* valid resistor range,
* unsupported weights handled as `open`,
* hardware-ready network configuration.

## 8.2 Repository location

Hardware-aware logic may be stored in:

```text
snn_pipeline/e24_quantizer.py
snn_pipeline/hil_validation.py
snn_pipeline/export.py
```

## 8.3 Expected output format

```json
{
  "sample_id": "sample_001",
  "e24_valid": true,
  "mapped_weights": [10000, 22000, "open", 47000],
  "open_weights": 1,
  "hardware_ready": true
}
```

## 8.4 Contract rules

The hardware-aware output must satisfy:

```text
mapped_weights may contain E24 resistor values or "open"
open means no physical resistor/no connection
invalid or too-small weights should not be forced into E24 mapping
hardware_ready must be false if mapping violates constraints
```

---

# 9. CI/CD artifact strategy

## 9.1 Purpose

GitHub Actions should automatically collect outputs from all important pipeline stages and publish them as artifacts.

This allows the team to inspect:

* dataset snapshots,
* encoder outputs,
* network outputs,
* agent outputs,
* logs,
* metrics,
* reports,
* generated hardware files.

## 9.2 Artifact folders

The CI workflow should collect outputs into:

```text
ci_artifacts/
├── dataset/
├── encoder/
├── network/
├── agents/
├── logs/
├── simulation/
└── summary/
```

## 9.3 Source-to-artifact mapping

| Source folder/file           | Artifact destination       |
| ---------------------------- | -------------------------- |
| `data/`                      | `ci_artifacts/dataset/`    |
| `encoder/encoder_output/`    | `ci_artifacts/encoder/`    |
| `output/`                    | `ci_artifacts/network/`    |
| `software/agent_output/`     | `ci_artifacts/agents/`     |
| `logs/`                      | `ci_artifacts/logs/`       |
| `benchmark_table.txt`        | `ci_artifacts/simulation/` |
| `hat_learning_curves.png`    | `ci_artifacts/simulation/` |
| `hat_weight_table.txt`       | `ci_artifacts/simulation/` |
| `mcp4151_table.csv`          | `ci_artifacts/simulation/` |
| `qat_learning_curves.png`    | `ci_artifacts/simulation/` |
| `sensitivity_heatmap.png`    | `ci_artifacts/simulation/` |
| `snn_config.h`               | `ci_artifacts/simulation/` |
| `weight_error_histogram.png` | `ci_artifacts/simulation/` |
| `weights.csv`                | `ci_artifacts/simulation/` |
| `weights.json`               | `ci_artifacts/simulation/` |

## 9.4 Manifest

Each CI run should generate:

```text
ci_artifacts/summary/manifest.md
```

The manifest should contain:

```text
branch name
commit SHA
GitHub run ID
data flow description
list of collected files
```

---

# 10. Branch and review logic

The CI/CD pipeline should not automatically merge code.

The intended process is:

```text
feat/* → PR to dev → CI validates → human reviews → merge decision
dev → PR to main → CI validates → human reviews → release decision
```

The pipeline provides evidence for the reviewer:

* tests passed,
* simulation passed,
* artifacts were generated,
* output files are available,
* metrics can be inspected.

The final merge decision belongs to the team.

---

# 11. Current implementation plan

## Stage 1: CI foundation

Implemented or planned:

* lint with Ruff,
* format check with Ruff,
* pytest unit tests,
* CI execution on `feat/**`,
* CI execution on `integration/**`,
* CI execution on PRs to `dev` and `main`.

## Stage 2: Artifact collection

Planned:

* collect dataset outputs,
* collect encoder outputs,
* collect network outputs,
* collect agent outputs,
* collect logs,
* generate manifest,
* upload artifacts from GitHub Actions.

## Stage 3: Simulation pipeline

Planned:

* run full SNN simulation on `integration/**`,
* run optional simulation on PRs,
* upload generated SNN simulation files as artifacts.

## Stage 4: Quality gates

Planned:

* validate encoder output shape,
* validate 7-neuron contract,
* validate spike values,
* validate network output,
* validate hardware mapping,
* validate metrics thresholds.

---

# 12. Summary

This contract defines the expected data exchange between all major pipeline stages:

```text
dataset → encoder → SNN/network → network output → agents → metrics/report
```

Each stage must return a predictable output format.

The CI/CD pipeline should:

* run tests,
* collect component outputs,
* run simulation on integration branches,
* upload artifacts,
* provide reviewers with clear evidence,
* support but not replace human merge decisions.
