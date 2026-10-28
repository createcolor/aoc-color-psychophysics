"""Validation: how accurately the colorimeter reproduces set color differences
while the two stimuli alternate, how stable it is over a 54-hour experiment
(2026-10-17), and how the reference display does under the same protocol
(2026-10-28). Paper: Table 1, Section 5, Fig. 4 headers.

Both data files hold, per color center, every pair of the protocol
(8 directions x 5 set steps) with all its readings: the measured center and
test color (xyY), the commanded test color, and the center error recorded by
the control software (D50). Readings are grouped into measurement blocks by
the gaps (> 10 min) in the sweep times: 14 blocks of 24 min for the
colorimeter, two cycles of about an hour for the display, whose model is its
measured primaries plus its measured black level.

  error at the center    mean +/- SD of the recorded center error over all readings
  pair-separation error  |dE00(measured center, measured test) - dE00(target,
                         commanded test)| per reading, mean +/- SD, both in
                         CIELAB against D50 (CIE 1964 10 deg) at 100 cd/m^2.
                         The colorimeter's control software placed the test
                         colors with a D65 white, so the commanded separations,
                         not the set steps, are the reference.
  per pair (Fig. 4)      |mean over readings of the measured separation -
                         commanded separation|: mean and max over the 40 pairs
  stability              spread (max - min) of the block means, and their
                         linear trend over the experiment (fitted slope x blocks)

Usage:
    PYTHONPATH=.:scripts python scripts/validation.py
Writes output/validation.json.
"""
from __future__ import annotations

import json

# common before colour: it silences colour's matplotlib warning
from common import CENTERS, DATA, OUT

import numpy as np
from colour import CCS_ILLUMINANTS, XYZ_to_Lab, xyY_to_XYZ
from colour.difference import delta_E_CIE2000

RUNS = {"colorimeter": DATA / "validation_experiment_2026-10-17.json",
        "display": DATA / "display_validation_2026-10-28.json"}
D50 = CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["D50"]


def lab(xyY) -> np.ndarray:
    return XYZ_to_Lab(xyY_to_XYZ(np.asarray(xyY, dtype=float)) / 100.0, illuminant=D50)


def device(d: dict) -> dict:
    """The statistics above for one device's run, per center and pooled."""
    out, commanded = {}, []
    for c in CENTERS:
        v = d["results"][c]
        block = np.concatenate([[0], np.cumsum(np.diff(v["times_seconds"]) > 600)])
        t_lab = lab(v["target_xyY"])
        center, pair, pair_block, per_pair = [], [], [], []
        for dv in v["directions"].values():
            for e in dv["steps"].values():
                cmd = float(delta_E_CIE2000(t_lab, lab(e["target_second_xyY"])))
                commanded.append(cmd)
                n = min(len(e["center_measured_xyY"]), len(e["second_measured_xyY"]))
                sep = np.asarray(delta_E_CIE2000(lab(e["center_measured_xyY"][:n]),
                                                 lab(e["second_measured_xyY"][:n])), dtype=float)
                center.append(np.asarray(e["center_delta_E"][:n], dtype=float))
                pair.append(np.abs(sep - cmd))
                pair_block.append(block[:n])
                per_pair.append(abs(sep.mean() - cmd))
        center, pair, b = np.concatenate(center), np.concatenate(pair), np.concatenate(pair_block)
        cb = np.array([center[b == i].mean() for i in np.unique(b)])
        pb = np.array([pair[b == i].mean() for i in np.unique(b)])
        checks = d.get("reproduction_checks", {})
        out[c] = {
            "primaries_nm": sorted(h["wavelength_nm"] for h in checks[c]["headroom"]) if c in checks else None,
            "n_blocks": int(len(cb)), "readings_per_pair": min(map(len, pair_block)),
            "center_mean": float(center.mean()), "center_sd": float(center.std()),
            "pair_mean": float(pair.mean()), "pair_sd": float(pair.std()), "pair_max": float(pair.max()),
            "per_pair_mean": float(np.mean(per_pair)), "per_pair_max": float(np.max(per_pair)),
            "center_block_spread": float(cb.max() - cb.min()),
            "center_block_trend": float(np.polyfit(np.arange(len(cb)), cb, 1)[0] * len(cb)),
            "pair_block_spread": float(pb.max() - pb.min())}
    out["commanded_separation_range"] = [float(min(commanded)), float(max(commanded))]
    out["mean_center"] = float(np.mean([out[c]["center_mean"] for c in CENTERS]))
    out["mean_pair"] = float(np.mean([out[c]["pair_mean"] for c in CENTERS]))
    return out


def main() -> None:
    out = {}
    for name, path in RUNS.items():
        r = out[name] = device(json.loads(path.read_text()))
        print(f"{name}: commanded separations {r['commanded_separation_range'][0]:.2f}.."
              f"{r['commanded_separation_range'][1]:.2f} dE00 (D50); means over centers: "
              f"center {r['mean_center']:.2f}, pair {r['mean_pair']:.2f}")
        for c in CENTERS:
            s = r[c]
            print(f"  {c:6s} center {s['center_mean']:.2f} ± {s['center_sd']:.2f}  "
                  f"pair {s['pair_mean']:.2f} ± {s['pair_sd']:.2f} (max {s['pair_max']:.2f})  "
                  f"per pair {s['per_pair_mean']:.2f}/{s['per_pair_max']:.2f}  blocks {s['n_blocks']}: "
                  f"center spread {s['center_block_spread']:.2f} trend {s['center_block_trend']:+.2f}, "
                  f"pair spread {s['pair_block_spread']:.2f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
