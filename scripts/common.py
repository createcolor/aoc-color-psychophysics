"""Shared pieces of the reproduction: paths, the device model, the observer
population, the display-derived color-center targets and the drive step.

Conventions used by every selection script (Section 6 of the paper):

  * CIE 1964 10 deg standard observer; CIELAB against the CIE D50 table
    spectrum at 100 cd/m^2, "as each observer sees it"; CIEDE2000.
  * The 151 color-normal observers of Asano, Fairchild & Blonde (2016),
    split once into 75 training and 76 held-out observers, stratified by age
    (seed 0). Selection uses the training half, every reported number the
    held-out half.
  * The target is the display-derived color center (the reference display's
    measured spectrum, integrated for the standard observer), converted to
    CIELAB against the standard observer's white. Each observer's percept is
    taken in that observer's own CIELAB; stimuli are luminance-anchored to the
    target.
  * Primaries are assigned to channels in descending wavelength (three
    primaries on channels 2-4, four on 1-4).
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import colour
import numpy as np
from colour.difference import delta_E_CIE2000

from ao.ao_converter import AoConverter
from ao.config import ADAPTATION_BRIGHTNESS
from ao.spectral_converter import XYZ_to_xyY, xyY_to_XYZ

from asano_cmf_model import asano_xyz_cmfs

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "ao" / "calibration" / "2026-10-14"
DATA = ROOT / "data"
OUT = ROOT / "output"
MONITOR_FILE = DATA / "monitor_2026-08-28.txt"
ASANO_FILE = DATA / "asano2015_color_normal_observers_XYZ10.npz"

CENTERS = ["gray", "yellow", "green", "blue", "red"]
# Nominal CIE color centers, CIE 1964 10 deg (x, y, Y in cd/m^2).
NOMINAL = {"gray": [0.314, 0.331, 30.0], "yellow": [0.388, 0.428, 69.3],
           "green": [0.248, 0.362, 24.0], "blue": [0.219, 0.216, 8.8],
           "red": [0.484, 0.342, 14.1]}
MIN_SEPARATION_NM = 20.0
RING_DE = 3.0         # radius of the 8-point neighborhood ring, dE00
DP = 0.1              # drive quantum (amplitude units)
SEED = 0              # observer split
N_TRAIN = 75          # the held-out half takes the remaining 76
N_TAU = 41            # samples along a four-primary coefficient family
BATCH = 4000          # candidates scored per batch in the transfer-cost run


def converter(cal: Path = CAL) -> AoConverter:
    return AoConverter(cal)


def primaries_of(sc) -> list[int]:
    """The calibrated primary wavelengths (nm), ascending."""
    return [int(w) for w in sorted(w for w in sc.spectra_table if not isinstance(w, str))]


# ------------------------------------------------------------------ display

def parse_monitor_spectra(path: Path = MONITOR_FILE) -> list[dict]:
    """The X-Rite dump of the display at the five centers -> [{name, wavelengths, spectrum}]."""
    colors, name, grid = [], None, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"Spectrum from ([\d.]+) to ([\d.]+) nm in (\d+) steps", line)
        if m:
            grid = np.linspace(float(m.group(1)), float(m.group(2)), int(m.group(3)))
            continue
        if line.lower().startswith(("peak value", "result is")):
            continue
        if "," in line and re.match(r"^[\s\-\d.eE,]+$", line):
            values = np.array([float(v) for v in line.split(",") if v.strip() != ""])
            wl = grid if (grid is not None and len(grid) == len(values)) else \
                np.linspace(370.0, 730.0, len(values))
            colors.append({"name": name or f"color_{len(colors)}",
                            "wavelengths": wl, "spectrum": values})
            grid = None
            continue
        name = line
    return colors


# ---------------------------------------------------------------- observers

def _xyz_cmfs(wavelengths, values) -> dict:
    return {"wavelengths": np.asarray(wavelengths, dtype=float),
            "xyz": np.asarray(values, dtype=float)}


def load_observers(age_model_ages=None, field_size: float = 10.0) -> list[dict]:
    """[{label, group, wavelengths, xyz[, age]}]: the CIE 1964 reference first,
    then the standard observers (CIE 2015, Stiles & Burch 1959 and 1955), the
    Asano model at ``age_model_ages`` (individual variation at zero), and the
    151 Asano color-normal observers (asano_observers())."""
    cie = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    observers = [{"label": "CIE 1964 10 deg (reference)", "group": "reference",
                  **_xyz_cmfs(cie.wavelengths, cie.values)}]
    cie2015 = colour.MSDS_CMFS["CIE 2015 10 Degree Standard Observer"]
    observers.append({"label": "CIE 2015 10 deg (2006)", "group": "standard",
                      **_xyz_cmfs(cie2015.wavelengths, cie2015.values)})
    sb10 = colour.MSDS_CMFS["Stiles & Burch 1959 10 Degree RGB CMFs"]
    sb10_wl = np.asarray(sb10.wavelengths, dtype=float)
    observers.append({"label": "Stiles & Burch 1959 10 deg", "group": "standard",
                      **_xyz_cmfs(sb10_wl, [colour.colorimetry.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(w)
                                            for w in sb10_wl])})
    sb2 = colour.MSDS_CMFS["Stiles & Burch 1955 2 Degree RGB CMFs"]
    sb2_wl = np.asarray(sb2.wavelengths, dtype=float)
    observers.append({"label": "Stiles & Burch 1955 2 deg", "group": "standard",
                      **_xyz_cmfs(sb2_wl, [colour.colorimetry.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(w)
                                           for w in sb2_wl])})
    for age in age_model_ages or []:
        wl, xyz = asano_xyz_cmfs(float(age), fieldsize=field_size)
        observers.append({"label": f"Asano model age={int(age)}", "group": "age_model",
                          "age": float(age), **_xyz_cmfs(wl, xyz)})
    z = asano_observers()
    for idx, age, xyz in zip(z["ids"], z["ages"], z["xyz"]):
        observers.append({"label": f"Asano CN #{int(idx)}", "group": "asano",
                          "age": float(age), **_xyz_cmfs(z["wavelengths"], xyz)})
    return observers


def asano_observers():
    """The Asano color-normal observers (ids, ages, wavelengths, 10 deg XYZ
    CMFs). Downloaded on first use from Zenodo record 3252742 via
    colour-datasets and cached in data/, so later runs need no network."""
    if not ASANO_FILE.exists():
        import colour_datasets
        items = list(colour_datasets.load("3252742")["Colour Normal Observers"].items())
        np.savez(ASANO_FILE, ids=np.array([int(i) for i, _ in items]),
                 ages=np.array([float(s.parameters["Age"]) for _, s in items]),
                 wavelengths=np.asarray(items[0][1].XYZ_10.wavelengths, dtype=float),
                 xyz=np.stack([np.asarray(s.XYZ_10.values, dtype=float) for _, s in items]))
    return np.load(ASANO_FILE)


def split_indices(ages, seed: int = SEED):
    """(train, test) indices into the Asano observers, stratified by age:
    ordered by age (ties broken by a seeded shuffle) and dealt alternately."""
    ages = np.asarray(ages, dtype=float)
    jitter = np.random.default_rng(seed).random(len(ages))
    order = np.lexsort((jitter, ages))
    train, test = order[0::2], order[1::2]
    if len(train) > N_TRAIN:                        # 151 deals 76/75; swap to 75/76
        train, test = test, train
    return np.sort(train), np.sort(test)


def observer_matrices(observers: list[dict], grid: np.ndarray) -> np.ndarray:
    """CMFs of every observer resampled onto ``grid`` -> (n_obs, n_wl, 3)."""
    out = np.zeros((len(observers), len(grid), 3))
    for i, o in enumerate(observers):
        for j in range(3):
            out[i, :, j] = np.interp(grid, o["wavelengths"], o["xyz"][:, j], left=0.0, right=0.0)
    return out


def _spectrum_to_XYZ(wl_s, spec, observer: dict) -> np.ndarray:
    wl_o, cmf = observer["wavelengths"], observer["xyz"]
    dx = float((wl_o[-1] - wl_o[0]) / (len(wl_o) - 1))
    spec_on_o = np.interp(wl_o, np.asarray(wl_s, dtype=float), np.asarray(spec, dtype=float),
                          left=0.0, right=0.0)
    return (spec_on_o[:, None] * cmf).sum(axis=0) * dx


def adapting_white(colors, ref_sc, reference, grid, dlam, ref_cmf, cmfs):
    """D50 at ADAPTATION_BRIGHTNESS as each observer in ``cmfs`` sees it, on the
    converter's luminance scale (anchored through the display's gray center)."""
    d50 = colour.SDS_ILLUMINANTS["D50"]
    spec = np.interp(grid, d50.wavelengths, d50.values, left=0.0, right=0.0)
    anchor = next(c for c in colors if c["name"] == "gray")
    Y_true = float(ref_sc.integrate_sd_to_XYZ(
        ref_sc.data_to_sd(anchor["wavelengths"], anchor["spectrum"]))[1])
    Y_anchor = _spectrum_to_XYZ(anchor["wavelengths"], anchor["spectrum"], reference)[1]
    Y_d50 = float(spec @ ref_cmf[:, 1]) * dlam
    spec = spec * (Y_anchor * ADAPTATION_BRIGHTNESS / Y_true) / Y_d50
    return np.einsum("w,owc->oc", spec, cmfs) * dlam


def labs_of(XYZ: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Lab of XYZ (..., n_obs, 3) against a per-observer white (n_obs, 3)."""
    total = white.sum(axis=1, keepdims=True)
    xy_w = white[:, :2] / np.where(total > 0, total, 1.0)
    wY = np.where(white[:, 1] > 0, white[:, 1], 1.0)
    flat = XYZ.reshape(-1, XYZ.shape[-2], 3) / wY[None, :, None]
    lab = colour.XYZ_to_Lab(flat.reshape(-1, 3), np.tile(xy_w, (flat.shape[0], 1)))
    return lab.reshape(XYZ.shape)


class Population:
    """The observer population and the five display-derived targets, set up
    once per calibration. ``center(name)`` gives a target; ``de(stim, t)`` the
    per-observer departure of a stimulus spectrum from it."""

    def __init__(self, conv: AoConverter):
        self.conv = conv
        self.sc = sc = conv._spectral_converter
        self.grid = np.asarray(sc.grid, dtype=float)
        self.dlam = float((self.grid[-1] - self.grid[0]) / (len(self.grid) - 1))
        self.colors = parse_monitor_spectra()
        observers = load_observers()
        reference = observers[0]
        asano = [o for o in observers if o["group"] == "asano"]
        self.cmfs = observer_matrices(asano, self.grid)
        self.ref_cmf = observer_matrices([reference], self.grid)[0]
        self.white = adapting_white(self.colors, sc, reference, self.grid, self.dlam,
                                    self.ref_cmf, self.cmfs)
        n_obs = self.cmfs.shape[0]
        self.white_t = np.tile(adapting_white(self.colors, sc, reference, self.grid, self.dlam,
                                              self.ref_cmf, self.ref_cmf[None]), (n_obs, 1))
        self.train, self.test = split_indices([o["age"] for o in asano])

    def center(self, name: str) -> dict:
        c = next(c for c in self.colors if c["name"].strip() == name)
        mon = np.interp(self.grid, c["wavelengths"], c["spectrum"], left=0.0, right=0.0)
        txyY = [float(v) for v in XYZ_to_xyY(np.asarray(
            self.sc.integrate_sd_to_XYZ(self.sc.data_to_sd(c["wavelengths"], c["spectrum"])),
            dtype=float))]
        Yt = float((np.einsum("w,wc->c", mon, self.ref_cmf) * self.dlam)[1])
        XYZ_t = np.asarray(xyY_to_XYZ(list(txyY)), dtype=float) * (Yt / txyY[2])
        lab_t = labs_of(np.tile(XYZ_t, (self.cmfs.shape[0], 1))[None], self.white_t)[0]
        return {"name": name, "mon": mon, "xyY": txyY, "Y": Yt, "lab": lab_t,
                "lab_conv": self.conv._xyz_to_lab(xyY_to_XYZ(txyY))}

    def de(self, stim: np.ndarray, t: dict) -> np.ndarray:
        """Per-observer dE00 of a stimulus from the target, luminance-anchored."""
        Y = float(np.einsum("w,wc->c", stim, self.ref_cmf)[1] * self.dlam)
        lab = labs_of(np.einsum("w,owc->oc", stim * (t["Y"] / Y), self.cmfs)[None] * self.dlam,
                      self.white)[0]
        return np.asarray(delta_E_CIE2000(t["lab"], lab), dtype=float)

    def de_display(self, t: dict) -> np.ndarray:
        """The same departure for the display's own spectrum."""
        lab = labs_of(np.einsum("w,owc->oc", t["mon"], self.cmfs)[None] * self.dlam, self.white)[0]
        return np.asarray(delta_E_CIE2000(t["lab"], lab), dtype=float)

    def resid(self, stim: np.ndarray, t: dict) -> float:
        """CIE 1964 residual of a stimulus: chromaticity distance to the target
        at the target's luminance."""
        xyY = XYZ_to_xyY(np.einsum("w,wc->c", stim, self.ref_cmf) * self.dlam)
        return float(self.conv._ciede2000(
            t["lab_conv"], self.conv._xyz_to_lab(xyY_to_XYZ([xyY[0], xyY[1], t["xyY"][2]]))))

    def stats(self, de: np.ndarray, half: str = "test") -> dict:
        idx = self.test if half == "test" else self.train
        return {"mean": float(de[idx].mean()), "worst": float(de[idx].max())}


def freqs_of(conv: AoConverter, wl_desc) -> np.ndarray:
    return conv._snap_frequencies_to_calibration_grid(
        conv._get_frequency(np.asarray(wl_desc, dtype=float)))


def ring(conv: AoConverter, xyY, radius: float = RING_DE) -> list[list[float]]:
    """The 8 colors at CIEDE2000 distance ``radius`` around xyY, at its luminance."""
    return [list(p) for p in conv._get_ellipse_points_ciede2000(list(xyY), radius, n_points=8)]


def step_from_powers(conv: AoConverter, powers, freqs, channels) -> float:
    """Minimal step: the worst dE00 a +/-DP move on any one channel produces,
    against the drive's own rendered color."""
    P = np.round(np.asarray(powers, dtype=float), 1)
    own = conv._xyz_to_lab(xyY_to_XYZ(XYZ_to_xyY(conv.P_f_to_XYZ(P, freqs, channels=channels))))
    worst = 0.0
    for i in range(len(P)):
        for signed in (DP, -DP):
            q = P.copy()
            q[i] += signed
            lab = conv._xyz_to_lab(xyY_to_XYZ(XYZ_to_xyY(conv.P_f_to_XYZ(q, freqs, channels=channels))))
            worst = max(worst, float(conv._ciede2000(own, lab)))
    return worst


def null_family(sc, wls, inten0, channels, ref_cmf):
    """The four-primary coefficient family through ``inten0``: the
    standard-observer null direction and the tau range that keeps every
    intensity within [0, channel ceiling]."""
    shapes = np.stack([np.asarray(sc._table_sd(wv, float(i), ch).values, float) / float(i)
                       for wv, i, ch in zip(wls, inten0, channels)])
    ceil = []
    for wv, ch in zip(wls, channels):
        table, wvs = sc._table_for(ch)
        ceil.append(float(np.max(table[wv][0])) if wv in wvs else np.inf)
    nullv = np.linalg.svd(np.einsum("kw,wc->ck", shapes, ref_cmf))[2][-1]
    lo, hi = -np.inf, np.inf
    for ci, ni, cl in zip(inten0, nullv, ceil):
        if ni > 1e-12:
            lo, hi = max(lo, -ci / ni), min(hi, (cl - ci) / ni)
        elif ni < -1e-12:
            lo, hi = max(lo, (cl - ci) / ni), min(hi, -ci / ni)
    return nullv, lo, hi


def candidates(A_ref: np.ndarray, target: np.ndarray, n_primaries: int):
    """Feasible (indices, coefficients) for every primary set of the given size:
    three primaries fix the coefficients, four leave a family sampled at N_TAU
    points; sets needing a negative contribution are dropped."""
    for combo in combinations(range(len(A_ref)), n_primaries):
        Ac = A_ref[list(combo)].T
        if n_primaries == 3:
            try:
                coefs = [np.linalg.solve(Ac, target)]
            except np.linalg.LinAlgError:
                continue
        else:
            part, *_ = np.linalg.lstsq(Ac, target, rcond=None)
            ns = np.linalg.svd(Ac)[2][-1]
            span = float(np.abs(part).max()) * 2.0
            coefs = [part + t * ns for t in np.linspace(-span, span, N_TAU)]
        for c in coefs:
            if np.all(c > 1e-9):
                yield combo, c
