# Arduino Encoder Plan

Current Status

The checked-in encoder is a GPIO spike prototype, not an SPI packet sender.

- `snn_encoder/snn_encoder.ino` reads microphone amplitude on `A0`.
- It emits a voltage spike on digital pin `D2`.
- It supports two compile-time modes:
  - `MODE_RATE_CODING`: emits a burst where spike count grows with amplitude.
  - `MODE_TTFS`: emits one fast spike when amplitude and amplitude delta cross
    the configured thresholds.
- `snn_decoder/snn_decoder.ino` counts falling edges on digital pin `D3`.

The planned Raspberry Pi / host packet protocol is documented in
`../docs/spi_protocol.md`. Treat that as the next integration target.

## Current GPIO Wiring

```text
Arduino encoder
  A0  <- microphone analog output
  D2  -> analog neuron input or decoder input
  GND -> shared ground

Arduino decoder
  D3  <- analog neuron output or encoder D2 during bench tests
  GND -> shared ground
```

For Arduino-to-Raspberry-Pi or Arduino-to-analog-board testing, keep all grounds
common. If a 5 V Arduino talks to 3.3 V GPIO, use a level shifter or a safe
resistor divider.

## Encoder Constants

Current sketch constants:

```cpp
#define MIC_PIN A0
#define SPIKE_OUT_PIN 2
#define MODE_TTFS
#define RC_WINDOW_MS 50
#define RC_NOISE_FLOOR 80
```

`MODE_TTFS` is enabled by default. To test rate coding, comment out
`MODE_TTFS` and uncomment `MODE_RATE_CODING`.

## TTFS Prototype Behavior

The prototype computes peak-to-peak amplitude over a 50 ms window:

```text
energy = max(analogRead(A0)) - min(analogRead(A0))
delta = energy - last_energy
```

It emits one 0.5 ms digital spike when:

```text
energy > RC_NOISE_FLOOR
delta > 30
```

After a TTFS spike, the sketch waits 100 ms as a refractory period.

## Rate-Coding Prototype Behavior

When rate coding is enabled, amplitude above the noise floor is mapped to
1-10 spikes:

```text
num_spikes = map(energy, RC_NOISE_FLOOR, 1023, 1, 10)
```

Each spike is a 0.5 ms HIGH pulse on `D2` with a 2 ms gap between spikes.

## Bench Test

1. Upload `snn_encoder/snn_encoder.ino` to the encoder Arduino.
2. Upload `snn_decoder/snn_decoder.ino` to the decoder Arduino.
3. Connect encoder `D2` to decoder `D3`.
4. Connect grounds.
5. Open the serial monitors at 115200 baud.

Expected encoder output:

```text
[ENCODER] Init SNN Hardware Encoder...
Mode: Time-to-First-Spike (TTFS)
TTFS Spike, Energy: ...
```

Expected decoder output:

```text
[DECODER] Started listening to SNN output...
Spikes in last second: ...
```

## Oscilloscope Expectations

On encoder pin `D2`:

```text
idle:  LOW
spike: HIGH for about 0.5 ms, then LOW
```

For analog-neuron input testing, verify:

- Spike amplitude after level shifting.
- Pulse width at the neuron input.
- No repeated false spikes in quiet conditions.
- Shared ground integrity.

## Planned SPI Trigger Protocol

The project-level SPI trigger protocol is a 2-byte packet:

```text
byte 0: magic bits + trigger bit + neuron fired flags
byte 1: timestamp LSB
```

See `../docs/spi_protocol.md` for the bit layout. That protocol still needs
firmware implementation. Do not use the older 6-byte `0xAE` packet idea as the
current source of truth.

## Calibration Checklist

1. Measure microphone baseline in quiet conditions.
2. Set `RC_NOISE_FLOOR` above the quiet peak-to-peak amplitude.
3. Generate controlled impulse sounds and watch encoder serial output.
4. Verify `D2` pulse width and amplitude on an oscilloscope.
5. Connect the analog neuron and confirm it fires only on intended spikes.
6. Record thresholds and drift in `../tests/neuron_characterization_20260322.md`.
