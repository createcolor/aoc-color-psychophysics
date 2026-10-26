"""Print the main results from output/, one section per result.

Usage:
    PYTHONPATH=.:scripts python scripts/summary.py
Also writes them to output/summary.txt.
"""
from __future__ import annotations

import json

import numpy as np

from common import CENTERS, OUT

lines: list[str] = []


def say(text: str = "") -> None:
    lines.append(text)


def load(name: str):
    return json.loads((OUT / name).read_text())


def span(values, fmt=".2f") -> str:
    return f"{min(values):{fmt}} to {max(values):{fmt}}"


def primaries(s) -> str:
    return "/".join(map(str, sorted(s)))


val = load("validation.json")
disp = load("display_model.json")["results"]
res = load("resolution_sets.json")
gate = load("standard_anchor_gate_refine.json")["results"]
k3 = load("k3_standard_anchor_best.json")["results"]
xval = load("primary_selection_crossvalidate.json")["results"]
sel = {c: load(f"standard_anchor_selection/{c}.json") for c in CENTERS}
blue_k3 = [r for r in load("ring_step_selection/blue_k3.json")["rows"] if "J_ring" in r]
obs = {c: {"mean": gate[c]["adopted"]["test_mean"], "worst": gate[c]["adopted"]["test_worst"]} for c in CENTERS}

say("1. Validation: set color differences are reproduced in alternation (2026-10-17 campaign, dE00)")
say(f"   commanded separations {span(val['commanded_separation_range'])}, "
    f"{span([val[c]['readings_per_pair'] for c in CENTERS], 'd')} readings per pair, 14 blocks over 54 h")
say(f"   {'':7s} {'primaries':12s} {'error at center':>16s} {'pair error':>13s} {'per pair mean/max':>18s}")
for c in CENTERS:
    r = val[c]
    say(f"   {c.capitalize():7s} {primaries(int(w) for w in r['primaries_nm']):12s} "
        f"{r['center_mean']:9.2f} ± {r['center_sd']:.2f} {r['pair_mean']:6.2f} ± {r['pair_sd']:.2f} "
        f"{r['per_pair_mean']:12.2f}/{r['per_pair_max']:.2f}")
say(f"   largest single-reading pair error {max(val[c]['pair_max'] for c in CENTERS):.2f}; "
    f"block means of the center error spread {span([val[c]['center_block_spread'] for c in CENTERS])} "
    f"(linear trend {span([val[c]['center_block_trend'] for c in CENTERS])}), "
    f"of the pair error {span([val[c]['pair_block_spread'] for c in CENTERS])}")

say("\n2. Primaries for color matching: the resolution criterion (CIE 1964 residual / minimal step, dE00)")
for c in CENTERS:
    r = res[c]
    say(f"   {c.capitalize():7s} {primaries(r['set']):16s} {r['resid']:5.2f} {r['step']:5.2f}   "
        f"(best k=3 J {r['J_ring_k3']:.3f}, k=4 J {r['J_ring_k4']:.3f})")
say("   display, 12-bit model: " + ", ".join(
    f"{c} {disp[c]['resid_12bit']:.2f}/{disp[c]['step_12bit']:.2f}" for c in CENTERS))
r_ = np.corrcoef([r["step_center"] for r in blue_k3],
                 [r["ring_mean_step"] for r in blue_k3])[0, 1]
say(f"   blue, three primaries: {len(blue_k3)} sets render the full ring; "
    f"center step vs mean ring step r = {r_:.3f}")

say("\n3. Primaries for color difference: the observer criterion (76 held-out observers, dE00)")
means = {c: np.array([r["test_mean"] for r in sel[c]["rows"]]) for c in CENTERS}
say(f"   across the feasible four-primary sets the mean departure varies "
    f"{span([m.max() / m.min() for m in means.values()], '.1f')}-fold (yellow {span(means['yellow'])}; "
    f"SD {span([100 * m.std() / m.mean() for m in means.values()], '.0f')} % of the mean)")
for c in CENTERS:
    a = gate[c]["adopted"]
    say(f"   {c.capitalize():7s} {primaries(gate[c]['set']):16s} mean {obs[c]['mean']:.2f} worst {obs[c]['worst']:.2f}"
        f"   residual {a['resid']:.2f} step {a['step']:.2f}   best k=3 {primaries(k3[c]['set'])} {k3[c]['test_mean']:.2f}")
y = gate["yellow"]
say(f"   yellow, the solver's drive vs the adopted drive: mean {y['solver']['test_mean']:.2f} -> "
    f"{y['adopted']['test_mean']:.2f}, worst {y['solver']['test_worst']:.2f} -> {y['adopted']['test_worst']:.2f}")

say("\n4. The criteria compared, with the reference display as baseline (held-out mean / worst, dE00)")
say(f"   {'':7s} {'resolution':>13s} {'observer':>13s} {'display':>13s}")
for c in CENTERS:
    rr, dd = res[c]["observer_test"], sel[c]["display_test"]
    say(f"   {c.capitalize():7s} {rr['mean']:7.2f} {rr['worst']:5.2f} {obs[c]['mean']:7.2f} {obs[c]['worst']:5.2f} "
        f"{dd['mean']:7.2f} {dd['worst']:5.2f}")
say(f"   resolution/observer mean {span([res[c]['observer_test']['mean'] / obs[c]['mean'] for c in CENTERS], '.1f')}; "
    f"the observer criterion costs at most {max(gate[c]['adopted']['step'] - res[c]['step'] for c in CENTERS):.2f} "
    f"in minimal step and saves up to "
    f"{max(res[c]['observer_test']['mean'] - obs[c]['mean'] for c in CENTERS):.2f} in mean departure")

say("\n5. Transfer: selected on other observers (CIE 2015, Stiles & Burch, age model), scored on the 151")
say("   cost over the in-sample optimum: " + ", ".join(f"{c} {xval[c]['gap']:.2f}" for c in CENTERS)
    + f"  ({span([v['n_candidates'] for v in xval.values()], 'd')} candidates per center)")

text = "\n".join(lines)
print(text)
(OUT / "summary.txt").write_text(text + "\n")
