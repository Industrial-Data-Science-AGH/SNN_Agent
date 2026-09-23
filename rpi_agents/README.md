# SNN integration foundation

Start with [W0 team handoff](../docs/project/W0_START_HERE.md) and
[contracts/mock quickstart](../contracts/README.md).

- agent/: dependency-light hardware interfaces and imported GPIO guard (W1).
- runtime/: stateful SNN interface for Patryk (P1).
- cloud/app/mock_api.py: local scripted fixtures for frontend/integration only.
- [MIGRATION.md](MIGRATION.md): source commit and selective legacy import.

No production server, cloud deployment, secrets, real inference or hardware
actuation is included in W0. Never deploy the demo API publicly.
