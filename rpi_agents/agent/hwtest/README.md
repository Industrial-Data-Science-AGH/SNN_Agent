# Hardware test helpers (W1)

## Mega stand-in (`mega_standin/`)

`mega_standin.ino` is **not the encoder**. It emits Uno serial protocol v1 with synthetic masks (50 priming
hops, then a burst of 8 hops every 100) so the Raspberry Pi bridge can be tested on a real UART before the real
Uno firmware exists. It has no microphone input and no pulse outputs. Anything measured with it is a stand-in
result and must be labelled as such in reports; it says nothing about the encoder, the SNN or the Uno.

The same stream shape is produced in software by `rpi_agents/agent/synthetic.py`, which the replay tests use.

Fault injection over serial (one character): `I` reprint the boot line, `G` lose 10 hops silently, `T` lose 10
hops and report them in `txdrop`, `L` merge one hop into a 384-sample frame.

### Build and flash (Arduino Mega 2560, from the Pi that the board is plugged into)

```sh
arduino-cli core install arduino:avr
arduino-cli compile --fqbn arduino:avr:mega:cpu=atmega2560 mega_standin
arduino-cli upload -p /dev/serial/by-id/<mega> --fqbn arduino:avr:mega:cpu=atmega2560 mega_standin
```

Use the `/dev/serial/by-id/...` path, not `/dev/ttyUSB0`. Uploading overwrites the sketch on the board. Note that
many Mega clones use a CH340 converter without a serial number, so that by-id name is the same for every such
board: with two of them plugged in, pick the port by its physical location instead.

## Probe (`python -m rpi_agents.agent.probe`)

Reads a serial port for a few seconds and prints one JSON summary of what the bridge would see: boots, frames,
priming, gaps with their cause, rejected lines, anomalies, stalls, batches and the device hop time.

```sh
python -m rpi_agents.agent.probe --port /dev/serial/by-id/<mega> --seconds 10
python -m rpi_agents.agent.probe --port /dev/serial/by-id/<mega> --seconds 10 --send G --send-at 4
```

Opening the port resets an Arduino (DTR), so every run starts with a new boot line and, on a Mega, about one
second of bootloader delay. Commands sent before that delay are lost, hence `--send-at`.
