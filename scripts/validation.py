"""Validation: how accurately the colorimeter reproduces set color differences
while the two stimuli alternate, and how stable it is over a 54-hour campaign
(2026-10-17). Paper: Table 1 (colorimeter columns), Section 5, Fig. 4 headers.

data/validation_campaign_2026-10-17.json holds, per color center, every pair
of the protocol (8 directions x 5 set steps) with all its readings: the
measured center and test color (xyY), the commanded test color, and the
center error recorded by the control software (D50). Readings are grouped
into the 14 measurement blocks by the gaps (> 10 min) in the sweep times.

  error at the center    mean +/- SD of the recorded center error over all readings
  pair-separation error  |dE00(measured center, measured test) - dE00(target,
                         commanded test)| per reading, mean +/- SD, both in
                         CIELAB against D50 (CIE 1964 10 deg) at 100 cd/m^2.
                         The control software placed the test colors with a
                         D65 white, so the commanded separations, not the set
                         steps, are the reference.
  per pair (Fig. 4)      |mean over readings of the measured separation -
                         commanded separation|: mean and max over the 40 pairs
  stability              spread (max - min) of the 14 block means, and their
                         linear trend over the campaign (fitted slope x 14 blocks)

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

CAMPAIGN = DATA / "validation_campaign_2026-10-17.json"
D50 = CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["D50"]


def lab(xyY) -> np.ndarray:
    return XYZ_to_Lab(xyY_to_XYZ(np.asarray(xyY, dtype=float)) / 100.0, illuminant=D50)


def main() -> None:
    d = json.loads(CAMPAIGN.read_text())
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
        out[c] = {
            "primaries_nm": sorted(h["wavelength_nm"] for h in d["reproduction_checks"][c]["headroom"]),
            "n_blocks": int(len(cb)), "readings_per_pair": min(map(len, pair_block)),
            "center_mean": float(center.mean()), "center_sd": float(center.std()),
            "pair_mean": float(pair.mean()), "pair_sd": float(pair.std()), "pair_max": float(pair.max()),
            "per_pair_mean": float(np.mean(per_pair)), "per_pair_max": float(np.max(per_pair)),
            "center_block_spread": float(cb.max() - cb.min()), "center_block_trend": float(np.polyfit(np.arange(len(cb)), cb, 1)[0] * len(cb)),
            "pair_block_spread": float(pb.max() - pb.min())}
        r = out[c]
        print(f"{c:6s} {r['primaries_nm']}  center {r['center_mean']:.2f} ± {r['center_sd']:.2f}  "
              f"pair {r['pair_mean']:.2f} ± {r['pair_sd']:.2f} (max {r['pair_max']:.2f})  "
              f"per pair {r['per_pair_mean']:.2f}/{r['per_pair_max']:.2f}  blocks {r['n_blocks']}: "
              f"center spread {r['center_block_spread']:.2f} trend {r['center_block_trend']:+.2f}, "
              f"pair spread {r['pair_block_spread']:.2f}")
    out["commanded_separation_range"] = [float(min(commanded)), float(max(commanded))]
    out["colorimeter_mean_center"] = float(np.mean([out[k]["center_mean"] for k in CENTERS]))
    out["colorimeter_mean_pair"] = float(np.mean([out[k]["pair_mean"] for k in CENTERS]))
    print(f"commanded separations {min(commanded):.2f}..{max(commanded):.2f} dE00 (D50); "
          f"means over centers: center {out['colorimeter_mean_center']:.2f}, "
          f"pair {out['colorimeter_mean_pair']:.2f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
