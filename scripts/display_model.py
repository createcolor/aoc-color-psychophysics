"""The reference display's residual and minimal step under the CIE 1964 observer,
through a linear model of its three measured primary spectra: the baseline
for the colorimeter's resolution (paper: Table 2, display columns).

  * A: CIE 1964 10 deg XYZ of the measured EIZO CG245W primaries
    (data/eizo_primaries/, Argyll .sp), integrated with the converter;
  * per nominal color center, weights w = A^-1 XYZ(target), quantized to the
    panel's effective 12 bits (1/4095): residual = dE00(target, A w_q);
  * minimal step: the largest dE00 of a +/-1 LSB move on one channel, against
    the display's own rendered color; the native 10-bit step alongside.

Scale: one LSB is a fixed fraction of full drive, so both quantities depend on
the primaries' absolute luminance. The primaries are scaled to a 295 cd/m^2
white, the scale of the paper (the measured primaries sum to 202 cd/m^2 on the
converter's scale). The 12-bit residuals are rounding artefacts that move in
the second decimal with the scale.

Usage:
    PYTHONPATH=.:scripts python scripts/display_model.py
Writes output/display_model.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from colour.difference import delta_E_CIE2000

from ao.spectral_converter import xyY_to_XYZ
from common import DATA, NOMINAL, OUT, converter

WHITE_Y = 295.0


def read_sp(path: Path):
    """Argyll .sp emission measurement -> (wavelengths, values)."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    start = float(re.search(r'SPECTRAL_START_NM\s+"([\d.]+)"', txt).group(1))
    end = float(re.search(r'SPECTRAL_END_NM\s+"([\d.]+)"', txt).group(1))
    n = int(re.search(r'SPECTRAL_BANDS\s+"(\d+)"', txt).group(1))
    data = txt.split("BEGIN_DATA")[-1].split("END_DATA")[0].split()
    return np.linspace(start, end, n), np.array([float(x) for x in data[:n]])


def main() -> None:
    conv = converter()
    sc = conv._spectral_converter
    A = np.column_stack([
        np.asarray(sc.integrate_sd_to_XYZ(sc.data_to_sd(*read_sp(DATA / "eizo_primaries" / f"eizo_{n}.sp"))),
                   dtype=float) for n in ("red", "green", "blue")])
    A = A * WHITE_Y / float((A @ np.ones(3))[1])

    def lab(XYZ):
        return conv._xyz_to_lab(np.asarray(XYZ, dtype=float))

    results = {}
    for name, xyY in NOMINAL.items():
        t = np.asarray(xyY_to_XYZ(list(xyY)), dtype=float)
        w = np.linalg.solve(A, t)
        row = {"weights": [float(x) for x in w]}
        for bits in (12, 10):
            q = 2 ** bits - 1
            wq = np.round(w * q) / q
            lab_r = lab(A @ wq)
            row[f"resid_{bits}bit"] = float(delta_E_CIE2000(lab(t), lab_r))
            row[f"step_{bits}bit"] = max(float(delta_E_CIE2000(lab_r, lab(A @ (wq + s * np.eye(3)[i] / q))))
                                         for i in range(3) for s in (1.0, -1.0))
        results[name] = row
        print(f"{name:12s} resid {row['resid_12bit']:.3f}  step {row['step_12bit']:.3f}  "
              f"(10 bit: step {row['step_10bit']:.3f})")
    OUT.mkdir(exist_ok=True)
    (OUT / "display_model.json").write_text(json.dumps(
        {"white_Y": WHITE_Y, "results": results}, indent=1))


if __name__ == "__main__":
    main()
