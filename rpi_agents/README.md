# rpi_agents — Wake-Up AI RPi 5 Agent

Rule-based state machine that wakes on an SNN hardware trigger, runs a
local prefilter + Gemini vision check, and fires actuators + email on a
confirmed intrusion.

State flow: `WAKE → CAPTURE → PREFILTER → [VISION] → DECIDE → re-halt`

See [`docs/docs_index.md`](docs/docs_index.md) for the full analysis and
[`../docs/`](../docs/) for hardware design.

---

## Mac dev setup (eval harness + unit tests only)

```bash
python -m venv .venv
source .venv/bin/activate

# picamera2 / gpiozero / lgpio install ONLY on the Pi (via apt).
# On Mac, install only the packages needed for the eval harness:
pip install numpy pytest python-dotenv google-genai opencv-python-headless

cd rpi_agents
python -m pytest tests/ -q
```

## Run the import-safety smoke test

```bash
cd rpi_agents
python -m agent.main --test
```

## Pi setup

```bash
sudo apt install -y python3-picamera2 python3-gpiozero python3-lgpio
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy secrets template and fill in values:
mkdir -p ~/.config/snn-agent
cp .env.example ~/.config/snn-agent/.env
chmod 600 ~/.config/snn-agent/.env
```

## Secrets

Secrets live in `~/.config/snn-agent/.env` on the Pi — never in the repo.
See [`.env.example`](.env.example) for the required keys.

## Architecture reference

- [`/Users/vityk/Documents/Knowledgebase/Projects/SNN_Agent/architecture.md`](../docs/project_analysis.md)
- [`docs/docs_index.md`](docs/docs_index.md)
