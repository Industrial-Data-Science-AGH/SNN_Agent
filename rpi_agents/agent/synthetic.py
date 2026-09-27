"""Synthetic Uno-protocol streams for replay tests and for the Mega stand-in.

This is NOT encoder output and carries no scientific meaning: the masks are fixed patterns that exercise
the bridge (silence, bursts, lost lines, a merged late frame, resets, corrupt lines). The Mega stand-in
sketch (hwtest/mega_standin) emits the same shape of stream so the two can be compared line by line.
Standard library only.
"""

from __future__ import annotations

from rpi_agents.agent.serial_protocol import crc8

FS_HZ, HOP, N_CH, PULSE_US = 19231, 192, 7, 6000
PRIMING_HOPS = 50
BURST_EVERY, BURST_LEN, BURST_MASK = 100, 8, 0b1110000  # hf_lo, hf_hi and flux channels


def _line(body: str, eol: bytes) -> bytes:
    data = body.encode("ascii")
    return b"$" + data + b"*" + b"%02X" % crc8(data) + eol


def boot_line(build_id: str = "5717A0D0", chset: str = "swap", eol: bytes = b"\n") -> bytes:
    return _line(f"B,1,{build_id},{FS_HZ},{HOP},{N_CH},{PULSE_US},{chset}", eol)


def hop_time_us(seq: int, base_us: int = 0) -> int:
    """micros() at the end of hop `seq` on the ideal 192/19231 s grid."""
    return (base_us + ((seq + 1) * HOP * 1_000_000 + FS_HZ // 2) // FS_HZ) & 0xFFFFFFFF


def frame_line(
    seq: int, *, mask: int = 0, n: int = HOP, flags: int = 0, txdrop: int = 0, base_us: int = 0,
    eol: bytes = b"\n",
) -> bytes:  # fmt: skip
    return _line(f"F,{seq},{hop_time_us(seq, base_us)},{n},{mask:02X},{flags:X},{txdrop}", eol)


def _damage(line: bytes, eol: bytes) -> bytes:
    """Same line with its last checksum digit changed, so the CRC no longer matches."""
    body = line[: -len(eol)]
    return body[:-1] + (b"1" if body[-1:] == b"0" else b"0") + eol


def _mask(seq: int) -> int:
    return BURST_MASK if seq >= PRIMING_HOPS and seq % BURST_EVERY < BURST_LEN else 0


def scenario(name: str, hops: int = 200, *, eol: bytes = b"\n", base_us: int = 0) -> bytes:
    """One boot followed by `hops` hops (two boots for `reset`). Names:

    silence  priming, then no spikes
    glass    priming, then a burst of BURST_LEN hops every BURST_EVERY hops
    gap      glass with hops 80..89 lost and the Uno reporting txdrop=10
    lost     glass with hops 80..89 lost silently (no txdrop, as if the cable dropped them)
    late     glass with hop 99 merged into a 384-sample frame at seq 100
    corrupt  glass with the checksum of every 50th line damaged
    reset    glass, then a second boot with seq restarting at 0
    """
    if name not in ("silence", "glass", "gap", "lost", "late", "corrupt", "reset"):
        raise ValueError(f"unknown scenario {name!r}")
    out = [boot_line(eol=eol)]
    txdrop = 0
    for seq in range(hops):
        prime = int(seq < PRIMING_HOPS)
        mask = 0 if name == "silence" else _mask(seq)
        if name in ("gap", "lost") and 80 <= seq < 90:
            txdrop += name == "gap"
            continue
        if name == "late" and seq == 99:
            continue
        n = 2 * HOP if name == "late" and seq == 100 else HOP
        line = frame_line(seq, mask=mask, n=n, flags=prime, txdrop=txdrop, base_us=base_us, eol=eol)
        if name == "corrupt" and seq % 50 == 49:
            line = _damage(line, eol)
        out.append(line)
    if name == "reset":
        out.append(boot_line(eol=eol))
        out += [frame_line(s, mask=_mask(s), flags=int(s < PRIMING_HOPS), eol=eol) for s in range(hops)]
    return b"".join(out)
