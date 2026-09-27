#!/usr/bin/env python3
"""
isr_cycles.py — statyczna analiza czasu ISR (ATmega328P) z deasemblacji.

Liczy liczbę cykli NAJKRÓTSZEJ i NAJDŁUŻSZEJ ścieżki przez ISR(ADC_vect)
(__vector_21), łącznie z wywołaniami funkcji libgcc (__mulhisi3, __adddi3, ...),
pętlami przesunięć (ldi rX,K ... dec rX; brne) i prologiem/epilogiem (push/pop).

Użycie:
    python3 tools/isr_cycles.py build/orig/fw.asm            # tabela
    python3 tools/isr_cycles.py build/orig/fw.asm --json out.json --label baseline
    python3 tools/isr_cycles.py fw.asm --fs 19231 --f-cpu 16000000

Model: liczby z tabeli instrukcji ATmega328P (AVRe+, PC 16-bit). Opóźnienie wejścia
w przerwanie (4 cykle) + JMP w tablicy wektorów (3 cykle) dodawane osobno (ENTRY_CYCLES).
Wynik to PRZEWIDYWANIE — do porównania z pomiarem na płytce (measurements.json).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

ENTRY_CYCLES = 4 + 3     # reakcja na przerwanie (4) + jmp w tablicy wektorów (3)

C1 = """add adc sub sbc subi sbci and andi or ori eor com neg inc dec mov movw ldi cp cpc cpi
tst lsl lsr rol ror asr swap bset bclr sei cli sec clc sez clz sen cln sev clv set clt seh clh
ses cls bst bld nop in out wdr sleep""".split()
C2 = "adiw sbiw lds sts ld st ldd std push pop sbi cbi rjmp ijmp mul muls mulsu fmul fmuls fmulsu".split()
C3 = "lpm elpm jmp rcall icall".split()
C4 = "call ret reti".split()
BRANCH = "brbs brbc breq brne brcs brcc brsh brlo brmi brpl brge brlt brhs brhc brts brtc brvs brvc brie brid".split()
SKIP = "cpse sbrc sbrs sbic sbis".split()
CYC = {**{m: 1 for m in C1}, **{m: 2 for m in C2}, **{m: 3 for m in C3}, **{m: 4 for m in C4}}

LINE = re.compile(r"^\s*([0-9a-f]+):\t([0-9a-f ]+?)\s*\t(\S+)\s*([^;]*)(?:;\s*(.*))?$")


def parse_asm(path):
    ins, sym, cur = {}, {}, None
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:", raw)
        if m:
            sym[m.group(2)] = int(m.group(1), 16)
            continue
        m = LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        addr = int(m.group(1), 16)
        size = len(m.group(2).split())
        mn, ops, cm = m.group(3), m.group(4).strip(), (m.group(5) or "")
        tgt = None
        if mn in BRANCH or mn in ("rjmp", "jmp", "call", "rcall"):
            t = re.search(r"0x([0-9a-f]+)", cm) or re.search(r"^0x([0-9a-f]+)$", ops)
            if t:
                tgt = int(t.group(1), 16)
        ins[addr] = dict(addr=addr, size=size, mn=mn, ops=ops, tgt=tgt)
    return ins, sym


class Analyzer:
    def __init__(self, ins, sym):
        self.ins, self.sym = ins, sym
        self.addrs = sorted(ins)
        self.loops = self._find_loops()
        self.memo = {}

    def nxt(self, a):
        return a + self.ins[a]["size"]

    def _find_loops(self):
        heads = {}
        for a, i in self.ins.items():
            if i["mn"] in BRANCH and i["tgt"] is not None and i["tgt"] <= a:
                heads[i["tgt"]] = a
        return heads          # head -> adres skoku wstecznego (szczegóły liczone leniwie)

    def _loop_info(self, head):
        back = self.loops[head]
        reg = None
        p = head
        while p <= back:
            if self.ins[p]["mn"] == "dec":
                reg = self.ins[p]["ops"]
            p = self.nxt(p)
        prev = [x for x in self.addrs if self.nxt(x) == head]
        k = None
        if reg and prev and self.ins[prev[0]]["mn"] == "ldi" and self.ins[prev[0]]["ops"].startswith(reg + ","):
            k = int(self.ins[prev[0]]["ops"].split(",")[1].strip(), 0)
        if k is None:
            raise SystemExit(f"pętla @0x{head:x}..0x{back:x}: nie umiem ustalić liczby iteracji")
        body, p = 0, head
        while p <= back:
            if self.ins[p]["mn"] in BRANCH and p != back:
                raise SystemExit(f"pętla @0x{head:x} ma wewnętrzny skok — nieobsługiwane")
            body += 1 if p == back else self._c(p)
            p = self.nxt(p)
        return back, k, k * body + (k - 1)     # (k-1) dodatkowych cykli za wzięty skok

    def _c(self, a):
        mn = self.ins[a]["mn"]
        if mn not in CYC and mn not in BRANCH and mn not in SKIP:
            raise SystemExit(f"nieznana instrukcja '{mn}' @0x{a:x} — dodaj do tabeli cykli")
        return CYC.get(mn, 1)

    def cost(self, a, depth=0):
        """(min, max) cykli od a do najbliższego ret/reti."""
        if a in self.memo:
            return self.memo[a]
        if depth > 400:
            raise SystemExit("zbyt głęboka rekurencja")
        i = self.ins.get(a)
        if i is None:
            raise SystemExit(f"skok pod nieznany adres 0x{a:x}")
        mn = i["mn"]
        if a in self.loops:
            back, k, tot = self._loop_info(a)
            lo, hi = self.cost(self.nxt(back), depth + 1)
            res = (tot + lo, tot + hi)
        elif mn in ("ret", "reti"):
            res = (4, 4)
        elif mn in ("call", "rcall"):
            clo, chi = self.cost(i["tgt"], depth + 1)
            lo, hi = self.cost(self.nxt(a), depth + 1)
            c = CYC[mn]
            res = (c + clo + lo, c + chi + hi)
        elif mn in ("rjmp", "jmp"):
            lo, hi = self.cost(i["tgt"], depth + 1)
            res = (CYC[mn] + lo, CYC[mn] + hi)
        elif mn in BRANCH:
            tl, th = self.cost(i["tgt"], depth + 1)
            fl, fh = self.cost(self.nxt(a), depth + 1)
            res = (min(1 + fl, 2 + tl), max(1 + fh, 2 + th))
        elif mn in SKIP:
            n1 = self.nxt(a)
            n2 = self.nxt(n1)
            skip_c = 1 + self.ins[n1]["size"] // 2
            fl, fh = self.cost(n1, depth + 1)
            sl, sh = self.cost(n2, depth + 1)
            res = (min(1 + fl, skip_c + sl), max(1 + fh, skip_c + sh))
        else:
            lo, hi = self.cost(self.nxt(a), depth + 1)
            c = self._c(a)
            res = (c + lo, c + hi)
        self.memo[a] = res
        return res

    def count_prologue(self, a):
        n = 0
        while self.ins[a]["mn"] in ("push", "in", "eor") or (self.ins[a]["mn"] == "push"):
            if self.ins[a]["mn"] == "push":
                n += 1
            a = self.nxt(a)
        return n


def analyze(asm_path, vector="__vector_21"):
    ins, sym = parse_asm(asm_path)
    if vector not in sym:
        raise SystemExit(f"brak symbolu {vector} w {asm_path}")
    an = Analyzer(ins, sym)
    lo, hi = an.cost(sym[vector])
    # rozmiar kodu ISR (bajty) — od symbolu do reti
    a, size, called = sym[vector], 0, set()
    while True:
        i = ins[a]
        size += i["size"]
        if i["mn"] in ("call", "rcall") and i["tgt"] is not None:
            called.add(i["tgt"])
        if i["mn"] == "reti":
            break
        a = an.nxt(a)
    names = {v: k for k, v in sym.items()}
    return dict(vector=vector, body_min=lo, body_max=hi, entry=ENTRY_CYCLES,
                total_min=lo + ENTRY_CYCLES, total_max=hi + ENTRY_CYCLES,
                pushes=an.count_prologue(sym[vector]), code_bytes=size,
                helpers=sorted(names.get(t, hex(t)) for t in called))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asm")
    ap.add_argument("--vector", default="__vector_21")
    ap.add_argument("--label", default="")
    ap.add_argument("--f-cpu", type=int, default=16_000_000)
    ap.add_argument("--fs", type=float, default=19231.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = analyze(a.asm, a.vector)
    us = lambda c: 1e6 * c / a.f_cpu
    budget = 1e6 / a.fs
    print(f"[{a.label or a.asm}] {a.vector}: {r['pushes']} push, kod {r['code_bytes']} B, helpery: {', '.join(r['helpers']) or '-'}")
    print(f"  cykle (z wejściem {r['entry']}): min {r['total_min']}  max {r['total_max']}")
    print(f"  czas @{a.f_cpu/1e6:g} MHz:   min {us(r['total_min']):.2f} us  max {us(r['total_max']):.2f} us"
          f"   | budżet 1/fs = {budget:.2f} us  -> zajętość CPU (max) {100*us(r['total_max'])/budget:.1f}%")
    if a.json:
        r.update(label=a.label, f_cpu=a.f_cpu, fs=a.fs)
        json.dump(r, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
