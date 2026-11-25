"""Fill {{key}} / {{key|fmt}} placeholders in paper/*.tex.in from numbers.json.

    python -m eaagn.render --numbers paper/numbers.json --paper paper

fmt: '3f' (default), '2f', 'pct' (x100, 1 decimal), 'p' (bootstrap p-value), 'i' (integer).
Missing keys render as a visible ``[?key]`` marker so nothing unverified ships silently.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

N_BOOT = 2000


def fmt(v, f):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if isinstance(v, str):
        return " ".join(v.replace("@", "at").split())
    if f == "pct":
        return f"{100 * v:.1f}"
    if f == "p":
        return f"<{2 / N_BOOT:.3f}" if v <= 0 else f"{v:.3f}"  # two-sided resolution = 2/N_BOOT
    if f == "i":
        return f"{int(round(v))}"
    if f.endswith("e"):
        m, e = f"{v:.{int(f[:-1]) - 1}e}".split("e")
        return "$" + m + r"\times10^{" + str(int(e)) + "}$"
    if f.endswith("f"):
        s = f"{v:.{int(f[:-1])}f}"
        return r"\ensuremath{-}" + s[1:] if s.startswith("-") else s  # valid in text and math mode
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    N = json.load(open(a.numbers))
    missing = set()

    def sub(m):
        key, _, f = m.group(1).partition("|")
        if key not in N:
            missing.add(key)
            return r"\textbf{[?" + key.replace("_", r"\_") + "]}"
        return fmt(N[key], f or "3f")

    for src in glob.glob(os.path.join(a.paper, "*.tex.in")):
        txt = open(src, encoding="utf-8").read()
        out = re.sub(r"\{\{([^}]+)\}\}", sub, txt)
        open(src[:-3], "w", encoding="utf-8").write(out)
    if missing:
        print("MISSING:", sorted(missing))
    else:
        print("all placeholders filled")


if __name__ == "__main__":
    main()
