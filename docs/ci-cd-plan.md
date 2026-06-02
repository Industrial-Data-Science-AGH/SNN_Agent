# CI/CD Plan

## 1. Current CI state

The project already uses GitHub Actions.

Current workflow:

- runs on pushes to `dev` and `feat/**`,
- runs on pull requests to `dev`,
- performs lint checks with Ruff,
- checks formatting with Ruff,
- runs tests with pytest.

Current jobs:

- `lint`
- `test`

## 2. Branching strategy

The project should follow a simple GitFlow-like structure:

```text
feat/* → dev → main
```

Branch roles:

- main - stable release branch,
- dev - integration branch
- feat/* - feature branchces for separate modules
- docs - documentation

All feature branches should be merged into dev through Pull Requests.

## 3. Branching strategy

The current CI/CD branch introduces:

- CI execution for feature branches,
- basic repository validation tests,
- fixed decoder logging path for cross-platform compatibility,
- ignored runtime logs and cache files.

The decoder logging issue was caused by a hardcoded Linux path:
```
/tmp/decoder.log
```

This failed beacuse it did not use PATH.

## 4. Target CI pipeline

The target CI pipeline should contain the following stages:
```text
lint
↓
unit tests
↓
integration tests
↓
SNN pipeline validation
↓
metrics validation
↓
report generation
```

## 5. Target model pipeline

The planned model validation flow is:
```text
dataset → encoder → SNN/model → metrics → report
```
This pipeline should validate whether changes in one module still work with the rest of the system.

## 6. Planned tests
### Encoder contract tests

To be added after the encoder is merged into `dev`.

Expected checks:

- encoder accepts valid input,
- encoder returns correct output shape,
- encoder returns the expected number of outputs,
- encoder does not return empty or invalid data.

Example requirement:
```text
encoder output count == 3
```

### Integration tests

To be added after the encoder, model and metrics modules are available on dev.

Expected flow:
```text
sample dataset → encoder → model → metrics
```

The integration test should check:

- dataset can be loaded,
- encoder output is compatible with the model input,
- model inference runs successfully,
- metrics are calculated correctly.

### Hardware-aware tests
To be added after the E24 / HAT logic is available on dev.

Expected checks:

- weights can be mapped to E24 resistor values,
- unsupported weights are marked as open weights,
- invalid resistor values are rejected,
- hardware constraints are respected.
  
## 7. Quality gates

In the future, PRs should be blocked if quality gates fail.

Planned quality gates:

- all lint checks pass,
- all unit tests pass,
- integration test passes,
- encoder output shape is valid,
- metrics meet minimum thresholds,
- hardware-aware mapping is valid.

Example target values:
```text
recall >= 0.85
encoder_outputs == 3
E24 mapping valid
```

## 8. CD approach

At the current stage, classic deployment is not required because the project does not yet have Dockerfiles, infrastructure or a production target.

For this project, CD should initially mean Continuous Validation / Continuous Experimentation.

The CD pipeline should:

- run the full model experiment,
- generate metrics,
- compare results with a baseline,
- publish the result as a GitHub Actions artifact.

## 9. Future artifacts

The pipeline should later generate:
```text
reports/metrics.json
reports/summary.md
```

These files should be uploaded as GitHub Actions artifacts.

## 10. Next steps

1. Keep the current CI foundation on feat/ci-cd-pipeline.
2. Merge or synchronize feature branches into dev.
3. Add encoder contract tests after encoder integration.
4. Add full pipeline integration tests after SNN/model integration.
5. Add metrics report generation.
6. Add quality gates based on technical and business metrics.


## TODO

- Everyone should do PR into dev.
```text
1. feat/encoder → dev
2. feat/decoder → dev
3. neuron-architecture → dev
4. hat-metrics-work → dev
5. feat/sim_train → dev
6. feat/ci-cd-pipeline → dev
```

- branch `integration/ci-cd-test` is only to test if everything will go right