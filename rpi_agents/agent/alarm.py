"""GPIO alarm output (LED and buzzer) with a local time limit. gpiozero is imported lazily.

Safety rules, all enforced here and none of them dependent on the cloud:
- Nothing touches a pin until the first apply(), and outputs are forced OFF when they are created.
- Every apply() arms a monotonic timer that switches the outputs off after the requested duration, capped by
  `max_ms`. A dead network, a crashed backend or a killed bridge cannot leave the alarm on for long: the
  timer lives in this process, and off() is also registered with atexit.
- Re-applying cannot keep the alarm on forever: continuous on-time is capped by `max_continuous_ms`.
- Asking for an output that is not wired raises AlarmUnavailable BEFORE anything is energised.
- off() always attempts every output, even if one of them fails, and is safe to call at any time.

Pin numbers are BCM. The defaults (LED GPIO17, buzzer GPIO27) are assumed from the previous agent and must be
checked against the real wiring before the first real activation; the buzzer was not wired in that project.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Callable, Protocol

log = logging.getLogger("snn_edge.alarm")
DEFAULT_LED_PIN, DEFAULT_BUZZER_PIN = 17, 27
CONTRACT_MAX_MS = 30_000  # AlarmCommand.parameters.duration_ms maximum


class AlarmUnavailable(RuntimeError):
    """The requested output is not configured, the hardware failed, or a safety limit forbids it."""


class Output(Protocol):
    def on(self) -> None: ...
    def off(self) -> None: ...
    def close(self) -> None: ...


def _gpiozero_output(kind: str) -> Callable[[int], Output]:
    def make(pin: int) -> Output:
        from rpi_agents.agent.gpio import configure_gpio

        configure_gpio()
        import gpiozero  # type: ignore[import-untyped]

        return getattr(gpiozero, kind)(pin)

    return make


class GpioAlarm:
    """Implements ports.AlarmAdapter for an LED and an optional buzzer."""

    def __init__(
        self,
        *,
        led_pin: int | None = DEFAULT_LED_PIN,
        buzzer_pin: int | None = None,
        max_ms: int = 10_000,
        max_continuous_ms: int = 30_000,
        make_led: Callable[[int], Output] | None = None,
        make_buzzer: Callable[[int], Output] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        timer: Callable[..., threading.Timer] = threading.Timer,
    ):
        if led_pin is None and buzzer_pin is None:
            raise ValueError("configure at least one output pin")
        if not 1 <= max_ms <= CONTRACT_MAX_MS or max_continuous_ms < max_ms:
            raise ValueError(f"max_ms must be 1..{CONTRACT_MAX_MS} and max_continuous_ms >= max_ms")
        self._pins = {"led": led_pin, "buzzer": buzzer_pin}
        self._makers = {"led": make_led or _gpiozero_output("LED"), "buzzer": make_buzzer or _gpiozero_output("Buzzer")}
        self._max_ms, self._max_continuous_ms = max_ms, max_continuous_ms
        self._mono, self._timer_factory = monotonic, timer
        self._lock = threading.RLock()
        self._outputs: dict[str, Output] = {}
        self._timer: threading.Timer | None = None
        self._on_since: float | None = None
        self._registered = False

    @property
    def led_available(self) -> bool:
        return self._pins["led"] is not None

    @property
    def buzzer_available(self) -> bool:
        return self._pins["buzzer"] is not None

    def _output(self, name: str) -> Output:
        if name not in self._outputs:
            device = self._makers[name](self._pins[name])
            device.off()  # never start energised
            self._outputs[name] = device
        return self._outputs[name]

    def initialize(self) -> None:
        """Create every configured output now, forced OFF. Call once at start-up when the alarm is enabled."""
        with self._lock:
            try:
                for name in ("led", "buzzer"):
                    if self._pins[name] is not None:
                        self._output(name)
            except Exception as exc:
                raise AlarmUnavailable(f"cannot initialise outputs: {type(exc).__name__}") from None

    def apply(self, *, duration_ms: int, led: bool, buzzer: bool) -> None:
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 1:
            raise ValueError("duration_ms must be a positive integer")
        if not (led or buzzer):
            raise ValueError("an alarm needs at least one active output")
        wanted = [name for name, on in (("led", led), ("buzzer", buzzer)) if on]
        for name in wanted:  # refuse before energising anything
            if self._pins[name] is None:
                raise AlarmUnavailable(f"{name} is not configured")
        with self._lock:
            now = self._mono()
            used_ms = 0.0 if self._on_since is None else (now - self._on_since) * 1000
            allowance_ms = self._max_continuous_ms - used_ms
            if allowance_ms <= 0:
                self.off()
                raise AlarmUnavailable("continuous on-time limit reached; the alarm was switched off")
            granted_ms = min(duration_ms, self._max_ms, allowance_ms)
            try:
                outputs = {name: self._output(name) for name in wanted}
                for name in ("led", "buzzer"):
                    if name in outputs:
                        outputs[name].on()
                    elif name in self._outputs:
                        self._outputs[name].off()  # this command asks for fewer outputs than before
            except Exception as exc:
                try:
                    self.off()
                except AlarmUnavailable:
                    pass  # already failing; the original error is the one to report
                raise AlarmUnavailable(f"output failed: {type(exc).__name__}") from None
            if self._on_since is None:
                self._on_since = now
            if self._timer is not None:
                self._timer.cancel()
            self._timer = self._timer_factory(granted_ms / 1000, self.off)
            self._timer.daemon = True
            self._timer.start()
            if not self._registered:
                atexit.register(self._off_at_exit)
                self._registered = True
            log.warning("alarm on: led=%s buzzer=%s for %d ms", led, buzzer, granted_ms)

    def off(self) -> None:
        """Switch everything off. Attempts every output; raises the first failure afterwards."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._on_since = None
            first_error: Exception | None = None
            for name, device in list(self._outputs.items()):
                try:
                    device.off()
                except Exception as exc:  # keep going: the other output must still be switched off
                    log.error("failed to switch %s off: %s", name, type(exc).__name__)
                    first_error = first_error or exc
            if first_error is not None:
                raise AlarmUnavailable(f"output did not switch off: {type(first_error).__name__}")

    def _off_at_exit(self) -> None:
        try:
            self.off()
        except Exception:  # nothing useful can be done at interpreter exit; off() already logged the failure
            pass

    def close(self) -> None:
        try:
            self.off()
        finally:
            with self._lock:
                for device in self._outputs.values():
                    try:
                        device.close()
                    except Exception:  # closing is best effort after off()
                        pass
                self._outputs.clear()


class SourceBox:
    """The bridge's currently-open serial source, if any -- shared mutable state between the
    main thread (which owns connect/reconnect) and the command thread (which drives the alarm).

    `SerialAlarm` is built before the `Bridge` exists (both are constructor arguments to
    `Bridge.__init__`), so it cannot hold a reference to the bridge or to a live source
    directly. This box is built first, handed to both, and the bridge's main loop updates
    `.current` on every connect/disconnect -- the same indirection `GpioAlarm` does not need
    because GPIO pins do not go away on a serial reconnect."""

    def __init__(self) -> None:
        self.current = None  # ByteSource | None; only set by the bridge's main thread


class SerialAlarm:
    """Implements ports.AlarmAdapter by sending the target board's own alarm command over the
    same serial link the bridge already reads $F frames from (task, 27.09.2026: the target
    hardware keeps microphone, LED and buzzer together on the encoder board -- see
    architecture_14_neurons_patryk_09_07/encoder_v2.ino 'A<ms>'/'Z' handlers -- instead of
    wiring LED/buzzer to separate Raspberry Pi GPIO pins the way GpioAlarm does).

    One real difference from GpioAlarm: the board drives LED and buzzer as ONE output (a
    single 'A<ms>' line lights both together), because that is what the firmware implements.
    Asking for only one of the two still energises both -- there is no way to ask the board
    for just one -- so `led_available`/`buzzer_available` are both true together, or both
    false, never split.

    Same safety discipline as GpioAlarm despite the device also self-timing (encoder_v2.ino's
    LED/buzzer auto-off after the requested duration): a monotonic timer here independently
    sends 'Z' (immediate off) after the granted duration, so a firmware bug or a lost line
    cannot be the only thing standing between "alarm" and "off" -- the same "not dependent on
    the cloud, and not dependent on the device either" property GpioAlarm documents for itself.
    """

    def __init__(
        self,
        source_box: SourceBox,
        *,
        max_ms: int = 10_000,
        max_continuous_ms: int = 30_000,
        monotonic: Callable[[], float] = time.monotonic,
        timer: Callable[..., threading.Timer] = threading.Timer,
    ):
        if not 1 <= max_ms <= CONTRACT_MAX_MS or max_continuous_ms < max_ms:
            raise ValueError(f"max_ms must be 1..{CONTRACT_MAX_MS} and max_continuous_ms >= max_ms")
        self._box = source_box
        self._max_ms, self._max_continuous_ms = max_ms, max_continuous_ms
        self._mono, self._timer_factory = monotonic, timer
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._on_since: float | None = None
        self._registered = False

    @property
    def led_available(self) -> bool:
        return True

    @property
    def buzzer_available(self) -> bool:
        return True

    def initialize(self) -> None:
        """Force OFF now. Unlike GpioAlarm there is no pin to reserve ahead of time -- the
        board is only reachable once the bridge has a live serial connection, which may not
        exist yet at start-up, so this is best-effort and does not raise when the port is
        down (the board itself starts with outputs LOW, per encoder_v2.ino setup())."""
        try:
            self.off()
        except AlarmUnavailable:
            pass

    def apply(self, *, duration_ms: int, led: bool, buzzer: bool) -> None:
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 1:
            raise ValueError("duration_ms must be a positive integer")
        if not (led or buzzer):
            raise ValueError("an alarm needs at least one active output")
        with self._lock:
            now = self._mono()
            used_ms = 0.0 if self._on_since is None else (now - self._on_since) * 1000
            allowance_ms = self._max_continuous_ms - used_ms
            if allowance_ms <= 0:
                self.off()
                raise AlarmUnavailable("continuous on-time limit reached; the alarm was switched off")
            granted_ms = min(duration_ms, self._max_ms, allowance_ms)
            self._write(f"A{int(granted_ms)}\n")
            if self._on_since is None:
                self._on_since = now
            if self._timer is not None:
                self._timer.cancel()
            self._timer = self._timer_factory(granted_ms / 1000, self.off)
            self._timer.daemon = True
            self._timer.start()
            if not self._registered:
                atexit.register(self._off_at_exit)
                self._registered = True
            log.warning("alarm on (serial): led=%s buzzer=%s for %d ms", led, buzzer, granted_ms)

    def off(self) -> None:
        """Always safe to call: no live connection means there is nothing to switch off,
        which is not a failure the way a write that starts and then breaks is."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._on_since = None
            self._write("Z\n", when_unconnected="ignore")

    def _write(self, line: str, *, when_unconnected: str = "raise") -> None:
        source = self._box.current
        if source is None:
            if when_unconnected == "ignore":
                return
            raise AlarmUnavailable("no live serial connection to the board right now")
        try:
            source.write(line.encode("ascii"))
        except Exception as exc:
            raise AlarmUnavailable(f"serial write failed: {type(exc).__name__}") from None

    def _off_at_exit(self) -> None:
        try:
            self.off()
        except Exception:  # nothing useful can be done at interpreter exit; off() already logged the failure
            pass

    def close(self) -> None:
        try:
            self.off()
        except AlarmUnavailable:
            pass  # port already gone; nothing more to do
