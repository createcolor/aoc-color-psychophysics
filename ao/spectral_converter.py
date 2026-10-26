"""Spectral model of the colorimeter: the calibrated spectrum of every band at
every drive level (one table per channel), integrated for the CIE 1964 10 deg
observer, and the intensity solves for three and four primaries."""

import pickle

import numpy as np
from scipy.optimize import minimize

from colour import MSDS_CMFS, XYZ_to_Lab
from colour.difference import delta_E_CIE2000

from ao.config import ADAPTATION_BRIGHTNESS

LUMINOS_EFFICENCY = 0.683  # mW/m^2

# Fraction of a band's calibrated maximum used to seed the 3-primary fixed point.
# Any value from ~0.1 to ~0.9 converges to the same solution; the failure mode is
# seeding far below the range, not the precise level.
SEED_INTENSITY_FRACTION = 0.25

# How close a 4-primary solution must come to the target color, in dE00 within
# the solver's own model, to count as a solution when the primary solve has
# failed and the boundary retries are being judged.
FOUR_PRIMARY_COLOR_TOL = 0.02


class TableBounds(Exception):
    def __init__(self, message, max_intens=None, asked_intens=None, wavelength=None):
        super().__init__(message)
        self.message = message
        self.max_intens: float = max_intens
        self.asked_intens: float = asked_intens
        self.wavelength: float = wavelength


def xyY_to_XYZ(xyY):
    """CIE xyY -> XYZ, for single points (3,) and batches (..., 3)."""
    xyY = np.asarray(xyY, dtype=float)
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(y != 0, x * Y / y, 0.0)
        Z = np.where(y != 0, (1 - x - y) * Y / y, 0.0)
    return np.stack([X, Y, Z], axis=-1)


def XYZ_to_xyY(XYZ) -> np.ndarray:
    """CIE XYZ -> xyY, for single points (3,) and batches (..., 3)."""
    XYZ = np.asarray(XYZ, dtype=float)
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    total = X + Y + Z
    x = np.full_like(X, 0.33)
    y = np.full_like(Y, 0.33)
    mask = total != 0
    x[mask] = X[mask] / total[mask]
    y[mask] = Y[mask] / total[mask]
    return np.stack([x, y, Y], axis=-1)


class Spectrum:
    """Values on a wavelength grid; spectra on the same grid add."""

    def __init__(self, wavelengths, values):
        self.wavelengths = np.array(wavelengths)
        self.values = np.array(values)
        assert self.wavelengths.shape == self.values.shape

    def __add__(self, other):
        if not np.allclose(self.wavelengths, other.wavelengths):
            raise ValueError("Wavelength grids do not match for addition.")
        return Spectrum(self.wavelengths, self.values + other.values)


class SpectralConverter:
    """Table model of the colorimeter's spectra for the CIE 1964 10 deg observer.

    ``table_paths`` maps each 1-based channel to its pickled spectra table:
    ``{wavelength: (intensities, spectra), "wavelengths": grid, ...}``, one
    spectrum per calibrated drive level, the intensity axis ascending.
    """

    def __init__(self, table_paths: dict):
        self.spectra_tables: dict[int, dict] = {}
        self.wvs_tables: dict[int, list] = {}
        for channel, path in table_paths.items():
            with open(path, "rb") as file:
                table = pickle.load(file)
            self.spectra_tables[int(channel)] = table
            self.wvs_tables[int(channel)] = self._table_wvs(table)
        default_channel = min(self.spectra_tables)
        self.spectra_table = self.spectra_tables[default_channel]
        self.wvs = self.wvs_tables[default_channel]
        self.grid = self.spectra_table["wavelengths"]
        self.dx_grid = (self.grid[-1] - self.grid[0]) / (len(self.grid) - 1)
        self.wv_step = (self.grid.max() - self.grid.min()) / self.grid.shape[0]
        cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
        self.cmfs_on_grid = np.zeros((len(self.grid), 3))
        for i in range(3):
            self.cmfs_on_grid[:, i] = np.interp(
                self.grid, np.ravel(cmfs.wavelengths), np.ravel(cmfs.values[:, i]), left=0, right=0)
        self.cmfs_on_grid *= LUMINOS_EFFICENCY
        if "black_level_spectrum" in self.spectra_table:
            self.black_level_spectrum = Spectrum(self.grid, self.spectra_table["black_level_spectrum"])
            self.black_level_xyz = self.integrate_sd_to_XYZ(self.black_level_spectrum)
        else:
            self.black_level_spectrum = Spectrum(self.grid, np.zeros_like(self.grid))
            self.black_level_xyz = np.array([0, 0, 0])

    @staticmethod
    def _table_wvs(table):
        """Sorted wavelength keys of a spectra table (string keys are metadata)."""
        return sorted(k for k in table.keys() if not isinstance(k, str))

    def _table_for(self, channel: int):
        return self.spectra_tables[channel], self.wvs_tables[channel]

    def _table_sd(self, wv: float, intensity: float = 1.0, channel: int = 1):
        """Spectrum of band ``wv`` at ``intensity``, interpolated in the channel's
        table over intensity and, between calibrated bands, over wavelength."""
        table, wvs = self._table_for(channel)

        if intensity < 1e-2:
            return Spectrum(self.grid, np.zeros_like(self.grid))

        if wv in wvs:
            intens = table[wv][0]
            inds = np.searchsorted(intens, intensity)
            try:
                spectras_up = table[wv][1][inds]
                spectras_down = table[wv][1][inds - 1]
            except IndexError as exc:
                max_intens = np.max(table[wv][0])
                raise TableBounds(
                    f"Max intensity bound reached on wv {wv} with asked {intensity} and max {max_intens}",
                    max_intens=max_intens, asked_intens=intensity, wavelength=wv) from exc
            shift = spectras_up - spectras_down
            spectra = spectras_down + (intensity - intens[inds - 1]) / (intens[inds] - intens[inds - 1]) * shift
            return Spectrum(self.grid, spectra)

        inds = np.searchsorted(wvs, wv)
        wv_up = wvs[inds]
        wv_down = wvs[inds - 1]

        # interpolation over intensity
        intens_up = table[wv_up][0]
        inds = np.searchsorted(intens_up, intensity)
        try:
            spectras_up_up = table[wv_up][1][inds]
        except IndexError as exc:
            max_intens = np.max(table[wv_up][0])
            raise TableBounds(
                f"Max intensity bound reached on wv {wv_up} with asked {intensity} and max {max_intens}",
                max_intens=max_intens, asked_intens=intensity, wavelength=wv_up) from exc
        spectras_up_down = table[wv_up][1][inds - 1]
        shift = spectras_up_up - spectras_up_down
        spectras_up = (
            spectras_up_down + (intensity - intens_up[inds - 1]) / (intens_up[inds] - intens_up[inds - 1]) * shift
        )

        intens_down = table[wv_down][0]
        inds = np.searchsorted(intens_down, intensity)
        try:
            spectras_down_up = table[wv_down][1][inds]
        except IndexError as exc:
            max_intens = np.max(table[wv_down][0])
            raise TableBounds(
                f"Max intensity bound reached on wv {wv_up} with asked {intensity} and max {max_intens}",
                max_intens=max_intens, asked_intens=intensity, wavelength=wv_down) from exc
        spectras_down_down = table[wv_down][1][inds - 1]
        shift = spectras_down_up - spectras_down_down
        spectras_down = (
            spectras_down_down
            + (intensity - intens_down[inds - 1]) / (intens_down[inds] - intens_down[inds - 1]) * shift
        )

        # interpolation over wv
        wv_shift_steps = np.floor((wv_up - wv_down) / self.wv_step)
        spectras_up_shifted = np.roll(spectras_up, wv_shift_steps)
        shift = spectras_up_shifted - spectras_down
        spectra = spectras_down + (wv - wv_down) / (wv_up - wv_down) * shift

        wv_shift_steps = np.floor((wv - wv_down) / self.wv_step)
        spectra = np.roll(spectra, -wv_shift_steps)
        return Spectrum(self.grid, spectra)

    def wavelengths_to_sd(self, wavelengths, intensities, consider_black_level=True, channels=None) -> Spectrum:
        """Combined spectrum of bands (scalars or sequences) at the given
        intensities, each through its channel's table (default channel i+1)."""
        if not hasattr(wavelengths, "__iter__"):
            wavelengths = [wavelengths]
        if not hasattr(intensities, "__iter__"):
            intensities = [intensities]
        if channels is None:
            channels = [i + 1 for i in range(len(wavelengths))]
        elif not hasattr(channels, "__iter__"):
            channels = [channels]
        if len(wavelengths) != len(intensities) or len(channels) != len(wavelengths):
            raise ValueError("Wavelengths, intensities and channels must have the same length.")

        result = None
        for wavelength, intensity, channel in zip(wavelengths, intensities, channels):
            if intensity <= 0:
                continue
            model = self._table_sd(wavelength, intensity, channel)
            result = model if result is None else result + model
        if result is not None and consider_black_level:
            result = result + self.black_level_spectrum
        if result is None:
            result = Spectrum(self.grid, np.zeros_like(self.grid))
        return result

    def data_to_sd(self, wv, spectral_data) -> Spectrum:
        """Arbitrary spectral data interpolated onto the grid."""
        idx = np.argsort(wv)
        values = np.interp(self.grid, np.ravel(wv[idx]), np.ravel(spectral_data[idx]), left=0, right=0)
        return Spectrum(self.grid, values)

    def integrate_sd_to_XYZ(self, spectrum: Spectrum) -> np.ndarray:
        values = np.ravel(spectrum.values)
        XYZ = np.zeros(3)
        for i in range(3):
            XYZ[i] = np.sum(values * self.cmfs_on_grid[:, i]) * self.dx_grid
        return XYZ

    def wavelengths_to_XYZ(self, wavelengths, intensities, consider_black_level=True, channels=None):
        return self.integrate_sd_to_XYZ(self.wavelengths_to_sd(
            wavelengths, intensities, consider_black_level=consider_black_level, channels=channels))

    def _seed_intensity(self, wavelength, channel, fraction=SEED_INTENSITY_FRACTION):
        """Starting intensity for the fixed-point solve: a fixed fraction of the
        band's calibrated maximum, so the seed is the same relative drive on
        every calibration."""
        try:
            self._table_sd(wavelength, 1e6, channel)
        except TableBounds as exc:
            return float(fraction * exc.max_intens)
        except (IndexError, KeyError, ValueError):
            pass
        return 20.0

    def _solve_3_primary_for_XYZ(self, wavelengths, target_XYZ, number_of_iterations=9, channels=None):
        """Iterative 3-primary solve for a black-level-subtracted XYZ: linearize
        each primary at the current intensities and solve, nine times. A negative
        iterate is clamped to a small floor and the iteration continues; only the
        final solve decides whether the target is out of gamut."""
        if channels is None:
            channels = [i + 1 for i in range(len(wavelengths))]
        intensities = [self._seed_intensity(wv, ch) for wv, ch in zip(wavelengths, channels)]
        floor = [max(1e-6, 1e-3 * float(v)) for v in intensities]
        solution = np.asarray(intensities, dtype=float)
        for it in range(number_of_iterations):
            XYZ_primaries = []
            # past a band's maximum, linearize at 0.9 of it
            for i, (wavelength, intens) in enumerate(zip(wavelengths, intensities)):
                try:
                    XYZ_primary = self.wavelengths_to_XYZ(wavelength, intens, consider_black_level=False,
                                                          channels=channels[i])
                except TableBounds as exc:
                    intensities[i] = exc.max_intens * 0.9
                    XYZ_primary = self.wavelengths_to_XYZ(wavelength, exc.max_intens * 0.9,
                                                          consider_black_level=False, channels=channels[i])
                    if it == number_of_iterations - 1:
                        raise exc
                XYZ_primaries.append(XYZ_primary)
            solution = np.linalg.solve(np.array(XYZ_primaries).T, target_XYZ) * np.asarray(intensities, dtype=float)
            intensities = list(np.maximum(solution, floor))
        return solution

    def _solve_4_primary_maximin(self, target_XYZ, wavelengths, channels):
        """Non-negative intensities for 4 primaries that match the target XYZ
        while maximizing the smallest channel intensity (SLSQP); if that fails,
        retried from each one-primary-off boundary solution."""
        bounds = []
        for wv, ch in zip(wavelengths, channels):
            try:
                max_intensity = self._table_sd(wv, 1e6, ch)
            except TableBounds as exc:
                max_intensity = exc.max_intens * 0.95
            bounds.append((0, max_intensity))

        def eval_primaries(intensities):
            primaries = [self.wavelengths_to_XYZ(w, val, consider_black_level=False, channels=ch)
                         for w, val, ch in zip(wavelengths, intensities, channels)]
            return np.sum(primaries, axis=0), primaries

        def constraint_fun(intensities):
            return eval_primaries(intensities)[0] - target_XYZ

        def constraint_jac(intensities):
            _, primaries = eval_primaries(intensities)
            return np.array([primary / max(intensities[i], 1e-3) for i, primary in enumerate(primaries)]).T

        def objective(intensities):
            return -np.min(intensities)

        def objective_jac(intensities):
            grad = np.zeros(len(wavelengths))
            grad[np.argmin(intensities)] = -1.0
            return grad

        intensities = np.array([3.0, 3.0, 3.0, 3.0])
        for _ in range(2):
            xyz_primaries = np.array([self.wavelengths_to_XYZ(w, val, consider_black_level=False, channels=ch)
                                      for w, val, ch in zip(wavelengths, intensities, channels)]).T
            intensities = (np.linalg.pinv(xyz_primaries) @ target_XYZ) * intensities
            intensities = np.clip(intensities, [b[0] for b in bounds], [b[1] for b in bounds])

        constraints = {"type": "eq", "fun": constraint_fun, "jac": constraint_jac}
        result = minimize(objective, intensities, jac=objective_jac, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"ftol": 1e-3, "maxiter": 500})
        best_result = result.x if result.success and np.all(result.x >= -1e-4) else None

        if best_result is None:
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            target_lab = XYZ_to_Lab(target_XYZ / ADAPTATION_BRIGHTNESS)

            def color_error(x):
                return float(delta_E_CIE2000(XYZ_to_Lab(eval_primaries(x)[0] / ADAPTATION_BRIGHTNESS), target_lab))

            def boundary_start(drop):
                """One primary off, the remaining three carrying the color."""
                keep = [i for i in range(len(wavelengths)) if i != drop]
                x = np.array(intensities, dtype=float)
                x[drop] = 0.0
                for _ in range(3):
                    M = np.array([self.wavelengths_to_XYZ(wavelengths[i], x[i], consider_black_level=False,
                                                          channels=channels[i]) for i in keep]).T
                    try:
                        scale = np.linalg.solve(M, target_XYZ)
                    except np.linalg.LinAlgError:
                        return None
                    for j, i in enumerate(keep):
                        x[i] = x[i] * scale[j]
                    x = np.clip(x, lo, hi)
                    x[drop] = 0.0
                return x

            candidates = []
            for drop in range(len(wavelengths)):
                start = boundary_start(drop)
                if start is None:
                    continue
                polished = minimize(objective, start, jac=objective_jac, method="SLSQP", bounds=bounds,
                                    constraints=constraints, options={"ftol": 1e-3, "maxiter": 500})
                for x in (polished.x, start):
                    x = np.asarray(x, dtype=float)
                    if np.any(x < -1e-4):
                        continue
                    x = np.clip(x, lo, hi)
                    candidates.append((color_error(x), x))
            feasible = [c for c in candidates if c[0] <= FOUR_PRIMARY_COLOR_TOL]
            if feasible:
                best_result = max(feasible, key=lambda c: float(np.min(c[1])))[1]

        if best_result is None:
            current_xyz = eval_primaries(intensities)[0]
            raise TableBounds(f"No feasible 4-primary solution for target XYZ {target_XYZ}",
                              max_intens=current_xyz[1], asked_intens=target_XYZ[1], wavelength=0)

        intensities_final = np.clip(best_result, 0, None)
        intensities_final[intensities_final < 1e-2] = 0
        return intensities_final

    def waves_n_xyY_to_ao_intesities(self, wavelengths, xyY, channels=None):
        """Intensities of 3 or 4 primaries that reproduce xyY for the standard
        observer (4 primaries: the max-min member of the family)."""
        target_XYZ = np.clip(xyY_to_XYZ(xyY) - self.black_level_xyz, 0, None)
        if channels is None:
            channels = [i + 1 for i in range(len(wavelengths))]
        if len(wavelengths) == 3:
            return self._solve_3_primary_for_XYZ(wavelengths, target_XYZ, channels=channels)
        if len(wavelengths) == 4:
            return self._solve_4_primary_maximin(target_XYZ, wavelengths, channels)
        raise NotImplementedError("Only 3 or 4 wavelengths are supported.")
