"""Asano (2015) / CIE 2006 individual-observer CMF model, age-parameterized.

Self-contained port of luxpy's ``cie2006cmfsEx`` (toolboxes/indvcmf) so it can be
used without importing luxpy (whose top-level import is currently broken on
NumPy 2). Only the CIE-2006 base data files are needed; they are bundled under
``data/asano_cie2006_data/``.

Reference:
    Asano Y., Fairchild M.D., Blondé L. (2016). Individual Colorimetric Observer
    Model. PLoS ONE 11(2): e0145671.  Base data & LMS->XYZ matrices from the
    Asano observer database (rit.edu/cos/colorscience).

Main entry point:
    asano_xyz_cmfs(age, fieldsize=10) -> (wavelengths[79], xyz[79, 3])

The individual-deviation parameters (od/shift std-devs) default to 0, so varying
only ``age`` isolates the age effect (lens/ocular-media yellowing), holding the
rest of the observer at the CIE-2006 population mean.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import interpolate

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "asano_cie2006_data"
_WL = np.arange(390.0, 781.0, 5.0)  # 79 samples; native grid of the base data
_WL_CRIT = 620.0  # S-cone absorbance undefined above this wavelength

# LMS cone-fundamentals -> XYZ CMF matrices (Asano database).
_M_2d = np.array([[0.4151, -0.2424, 0.0425],
                  [0.1355, 0.0833, -0.0043],
                  [-0.0093, 0.0125, 0.2136]])
_M_10d = np.array([[0.4499, -0.2630, 0.0460],
                   [0.1617, 0.0726, -0.0011],
                   [-0.0036, 0.0054, 0.2291]])


def _load(fname: str) -> np.ndarray:
    """Load a CIE-2006 base data file (mixed comma/tab, CR line endings, ragged)."""
    with open(_DATA_DIR / fname, newline="") as fh:
        raw = fh.read()
    rows = [r for r in re.split(r"[\r\n]+", raw) if r.strip()]
    parsed = []
    for r in rows:
        toks = [t for t in re.split(r"[,\t ]+", r.strip()) if t != ""]
        parsed.append([float(t) for t in toks])
    width = max(len(r) for r in parsed)
    arr = np.full((len(parsed), width), np.nan)
    for i, r in enumerate(parsed):
        arr[i, : len(r)] = r
    return arr


# Base data, oriented as luxpy uses it (transposed: components along axis 0).
_RMD = _load("asano_cie2006_RelativeMacularDensity.dat").T   # (1, 79) relative macular density
_LMSA = _load("asano_cie2006_Alms.dat").T                    # (3, 79) log10 photopigment absorbance
_DOCUL = _load("asano_cie2006_docul.dat").T                  # (2, 79) ocular-media density components


def _lms_to_xyz_matrix(fieldsize: float) -> np.ndarray:
    fs = min(max(fieldsize, 2.0), 10.0)
    a = (10.0 - fs) / (10.0 - 2.0)
    return _M_2d * (1 - a) + a * _M_10d


def asano_lms_cmfs(age: float = 32.0, fieldsize: float = 10.0,
                   var_od_lens: float = 0.0, var_od_macula: float = 0.0,
                   var_od_L: float = 0.0, var_od_M: float = 0.0, var_od_S: float = 0.0,
                   var_shft_L: float = 0.0, var_shft_M: float = 0.0, var_shft_S: float = 0.0
                   ) -> np.ndarray:
    """Area-normalized individual LMS cone fundamentals, shape (3, 79)."""
    rmd = _RMD.copy()
    lmsa = _LMSA.copy()
    docul = _DOCUL.copy()

    # Field-size corrected macular density.
    pk_od_macula = 0.485 * np.exp(-fieldsize / 6.132) * (1 + var_od_macula / 100)
    corrected_rmd = rmd * pk_od_macula

    # Age-corrected lens / ocular-media density (the dominant age effect).
    if age <= 60:
        lomd = docul[:1] * (1 + 0.02 * (age - 32)) + docul[1:2]
    else:
        lomd = docul[:1] * (1.56 + 0.0667 * (age - 60)) + docul[1:2]
    lomd = lomd * (1 + var_od_lens / 100)

    # Peak-wavelength shifts (0 for the population mean).
    wl_shifted = np.vstack([_WL + var_shft_L, _WL + var_shft_M, _WL + var_shft_S])
    lmsa_shft = np.empty_like(lmsa)
    for i in range(3):
        # The S-cone absorbance is undefined (NaN) above ~620 nm; interpolate over
        # the finite knots only and extrapolate. The unstable tail is caught by the
        # sign-switch guard below and by ``alpha[2, wl>=620]=0``.
        finite = np.isfinite(lmsa[i])
        lmsa_shft[i] = interpolate.interp1d(
            wl_shifted[i][finite], lmsa[i][finite], kind="cubic",
            bounds_error=False, fill_value="extrapolate",
        )(_WL)
    # Kill unstable S-cone extrapolation (sign switch above 560 nm).
    ssw = np.hstack((0, np.sign(np.diff(lmsa_shft[2, :]))))
    lmsa_shft[2, np.where((ssw >= 0) & (_WL > 560))] = np.nan

    # Field-size corrected peak optical densities of the photopigments.
    pk_od_L = (0.38 + 0.54 * np.exp(-fieldsize / 1.333)) * (1 + var_od_L / 100)
    pk_od_M = (0.38 + 0.54 * np.exp(-fieldsize / 1.333)) * (1 + var_od_M / 100)
    pk_od_S = (0.30 + 0.45 * np.exp(-fieldsize / 1.333)) * (1 + var_od_S / 100)

    alpha = np.zeros_like(lmsa_shft)
    alpha[0] = 1 - 10 ** (-pk_od_L * (10 ** lmsa_shft[0]))
    alpha[1] = 1 - 10 ** (-pk_od_M * (10 ** lmsa_shft[1]))
    alpha[2] = 1 - 10 ** (-pk_od_S * (10 ** lmsa_shft[2]))
    alpha[2, np.where(_WL >= _WL_CRIT)] = 0

    # Correct to corneal incidence, then to energy terms.
    lms_barq = alpha * (10 ** (-corrected_rmd - lomd))
    lms_bar = lms_barq * _WL
    lms_bar[np.isnan(lms_bar)] = 0
    lms = 100 * lms_bar / np.nansum(lms_bar, axis=1, keepdims=True)
    return lms


def asano_xyz_cmfs(age: float = 32.0, fieldsize: float = 10.0, **variability) -> tuple[np.ndarray, np.ndarray]:
    """Individual-observer XYZ CMFs.

    :return: ``(wavelengths[79], xyz[79, 3])`` with negatives clipped to 0.
    """
    lms = asano_lms_cmfs(age=age, fieldsize=fieldsize, **variability)
    xyz = _lms_to_xyz_matrix(fieldsize) @ lms
    xyz[xyz < 0] = 0
    return _WL.copy(), xyz.T


if __name__ == "__main__":
    for a in (20, 32, 50, 70):
        wl, xyz = asano_xyz_cmfs(a, 10)
        print(f"age {a}: XYZ CMF peak(x,y,z) @ wl",
              [float(wl[np.argmax(xyz[:, i])]) for i in range(3)],
              "sum", np.round(xyz.sum(axis=0), 2))
