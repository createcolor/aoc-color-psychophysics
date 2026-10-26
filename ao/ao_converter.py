"""Device model of the colorimeter: frequency <-> wavelength, drive amplitude
<-> band intensity per channel, and the drive solve for a target color."""

import warnings

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from colour import XYZ_to_Lab
from colour.colorimetry import CCS_ILLUMINANTS
from colour.difference import delta_E_CIE2000

from ao.config import ADAPTATION_BRIGHTNESS, NUMBER_OF_CHANNELS, PRIMARY_CHANNEL_ASSIGNMENT
from ao.spectral_converter import SpectralConverter, TableBounds, XYZ_to_xyY, xyY_to_XYZ

warnings.filterwarnings("ignore", category=RuntimeWarning)

# How negative a solved intensity may be before the color counts as outside the
# primaries' gamut, relative to the largest intensity in the same solve.
NEGATIVE_INTENSITY_REL_TOL = 1e-3


def _within_gamut(intensities, tol=NEGATIVE_INTENSITY_REL_TOL):
    """True if no intensity is negative beyond ``tol`` of the solve's own scale."""
    a = np.asarray(intensities, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        return False
    scale = float(np.max(np.abs(a)))
    if scale == 0.0:
        return False
    return bool(np.all(a >= -tol * scale))


def _read_calibration_csv(path):
    """Amplitude-intensity table of one channel -> (table, frequencies, wavelengths)."""
    data = pd.read_csv(path, delimiter=",", decimal=".")
    data = data[data["frequency"] != 65].copy()
    data["channel1_intensity"] = data["channel1_intensity"].fillna(0)
    data = data.dropna(axis=1, how="all").rename(columns={"channel1_intensity": "intensity"})
    by_freq = data.groupby("frequency").max()
    return data, by_freq.index.values, by_freq["wavelength"].values


class AoConverter:
    """Calibrated model of the four-channel colorimeter.

    ``calibration_dir`` holds, per channel N, ``amplitude_intensity_calibration_chN.csv``
    (band intensity against drive amplitude at every calibrated frequency) and
    ``wv_intens_spectra_chN`` (the spectrum at every drive level).
    """

    def __init__(self, calibration_dir):
        self._channels = list(range(1, NUMBER_OF_CHANNELS + 1))
        self.illuminant = CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["D50"]
        csv = {ch: calibration_dir / f"amplitude_intensity_calibration_ch{ch}.csv" for ch in self._channels}
        # frequency grid and wavelength mapping are shared (one crystal): channel 1
        self._calibration_data, self._frequency_array, self._wavelength_array = _read_calibration_csv(csv[1])
        self.wv_fr_interp = interp1d(np.array(self._wavelength_array), self._frequency_array,
                                     kind="linear", fill_value="extrapolate")
        self._table = {}
        for channel in self._channels:
            cal = _read_calibration_csv(csv[channel])[0]
            self._table[channel] = {f: cal[cal["frequency"] == f]["intensity"].values
                                    for f in self._frequency_array}
        self._spectral_converter = SpectralConverter(
            {ch: calibration_dir / f"wv_intens_spectra_ch{ch}" for ch in self._channels})

    @staticmethod
    def _channel_assignment(n_primaries: int) -> list:
        """Physical channels (1-based) driven for an ``n_primaries`` color."""
        return list(PRIMARY_CHANNEL_ASSIGNMENT[n_primaries])

    def _get_amplitude_range_from_csv(self):
        frequency = self._calibration_data["frequency"].values[0]
        return self._calibration_data[self._calibration_data["frequency"] == frequency]["amplitude"].values

    def _intensity_curve(self, fr, channel):
        """Intensity against amplitude at frequency ``fr`` (interpolated between
        calibrated frequencies), held at its peak beyond it -> (curve, i_max)."""
        table = self._table[channel]
        freq_array = np.array(self._frequency_array)
        if fr in freq_array:
            intens_array = table[fr].copy()
        else:
            idx = int(np.searchsorted(freq_array, fr))
            if idx >= len(freq_array):
                idx = len(freq_array) - 2
            if idx <= 0:
                idx = 0
            fr_down = freq_array[idx]
            fr_up = freq_array[idx + 1] if idx + 1 < len(freq_array) else freq_array[-1]
            shift = table[fr_up] - table[fr_down]
            intens_array = table[fr_down] + (fr - fr_down) / (fr_up - fr_down) * shift
        i_max = int(np.argmax(intens_array))
        intens_array[i_max:] = intens_array[i_max]
        return intens_array, i_max

    def _get_frequency(self, wavelength):
        return self.wv_fr_interp(wavelength)

    def _snap_frequencies_to_calibration_grid(self, frequencies):
        """Nearest calibrated frequency for each of ``frequencies``."""
        frequencies = np.asarray(frequencies, dtype=float)
        freq_grid = np.asarray(self._frequency_array, dtype=float)
        snapped = freq_grid[[int(np.argmin(np.abs(freq_grid - f))) for f in frequencies.ravel()]]
        return snapped.reshape(frequencies.shape) if frequencies.ndim else float(snapped[0])

    def _get_wavelengths_from_frequencies(self, frequencies):
        f_wv = interp1d(self._frequency_array, self._wavelength_array)
        return [float(f_wv(f)) for f in frequencies]

    def _xyz_to_lab(self, xyz):
        """CIELAB against D50 (CIE 1964 10 deg) at the adaptation luminance."""
        return np.array(XYZ_to_Lab(np.array(xyz) / ADAPTATION_BRIGHTNESS, illuminant=self.illuminant))

    def _ciede2000(self, lab1, lab2):
        return float(delta_E_CIE2000(lab1, lab2))

    def _get_ellipse_points_ciede2000(self, center_xyy, delta_e_threshold, n_points=8):
        """``n_points`` colors at CIEDE2000 distance ``delta_e_threshold`` around
        ``center_xyy`` at its luminance, equally spaced in xy direction (bisection
        to within 0.1 dE00)."""
        points_xyy = []
        center_lab = self._xyz_to_lab(xyY_to_XYZ(center_xyy))
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dx_init = delta_e_threshold * np.cos(angle) * 0.005
            dy_init = delta_e_threshold * np.sin(angle) * 0.005
            low, high = 0.0, delta_e_threshold * 2.0
            best_point = center_xyy + np.array([dx_init, dy_init, 0])
            for _ in range(10):
                mid = (low + high) / 2
                scale = mid / max(np.sqrt(dx_init**2 + dy_init**2), 0.001)
                test_point = center_xyy + np.array([dx_init * scale, dy_init * scale, 0])
                current_delta_e = self._ciede2000(center_lab, self._xyz_to_lab(xyY_to_XYZ(test_point)))
                if abs(current_delta_e - delta_e_threshold) < 0.1:
                    best_point = test_point
                    break
                elif current_delta_e < delta_e_threshold:
                    low = mid
                else:
                    high = mid
                best_point = center_xyy + np.array([dx_init * scale, dy_init * scale, 0])
            points_xyy.append(best_point)
        return np.array(points_xyy)

    def _feasible_for_targets(self, frequencies, target_points, tol_dE=1.0):
        """True iff every target renders on these frequencies through the device
        path (xyY_to_P_f, then P_f_to_XYZ) within ``tol_dE`` CIEDE2000."""
        freqs = [float(f) for f in frequencies]
        for p in target_points:
            target = [float(v) for v in p]
            try:
                power, freq = self.xyY_to_P_f(target, frequencies=freqs)
                rep = XYZ_to_xyY(self.P_f_to_XYZ(power, freq))
            except (ValueError, TableBounds, LinAlgError, IndexError):
                return False
            if self._ciede2000(self._xyz_to_lab(xyY_to_XYZ(target)), self._xyz_to_lab(xyY_to_XYZ(rep))) > tol_dE:
                return False
        return True

    def _get_intensity(self, frequencies, powers, channels=None):
        """Band intensity at each (frequency, drive amplitude) on its channel."""
        if channels is None:
            channels = [i + 1 for i in range(len(frequencies))]
        out = []
        for f, p, ch in zip(frequencies, powers, channels):
            curve, _ = self._intensity_curve(f, ch)
            out.append(interp1d(self._get_amplitude_range_from_csv(), curve, fill_value="extrapolate")(p))
        return out

    def _get_power(self, frequencies, intensities, channels=None):
        """Drive amplitude for each (frequency, band intensity) on its channel."""
        if len(frequencies) != len(intensities):
            raise ValueError("Frequency and intensity must have the same length.")
        if channels is None:
            channels = [i + 1 for i in range(len(frequencies))]
        out = []
        for f, intensity, ch in zip(frequencies, intensities, channels):
            if f < self._frequency_array.min() or f > self._frequency_array.max():
                raise ValueError(f"Frequency {f} is out of bounds.")
            curve, i_max = self._intensity_curve(f, ch)
            f_linear = interp1d(curve, self._get_amplitude_range_from_csv(), fill_value="extrapolate")
            out.append(np.clip(f_linear(intensity), 0, i_max + 1))
        return out

    def _channel_power_curve(self, fr, channel=1):
        """(amplitude, intensity) curve at a frequency as plain arrays, for the
        fast np.interp lookups of the four-primary search."""
        amp_range = np.asarray(self._get_amplitude_range_from_csv(), dtype=float)
        intens_array, i_max = self._intensity_curve(fr, channel)
        intens_array = np.asarray(intens_array, dtype=float)
        inc = i_max + 1                       # monotone branch for the inverse lookup
        return {"amp": amp_range, "intens": intens_array,
                "intens_inc": intens_array[:inc], "amp_inc": amp_range[:inc]}

    def _dP_sensitivity_fast(self, intensities, wavelengths, curves, target_lab, dP=0.1, channels=None):
        """Worst dE00 of a +/-dP drive move on one channel, the color of the
        others held: the base color with that channel's contribution swapped."""
        sc = self._spectral_converter
        intensities = np.clip(np.asarray(intensities, dtype=float), 0, None)
        n = len(intensities)
        if channels is None:
            channels = [i + 1 for i in range(n)]
        contrib = [sc.wavelengths_to_XYZ(wavelengths[i], float(intensities[i]), consider_black_level=False,
                                         channels=channels[i]) for i in range(n)]
        base = np.sum(contrib, axis=0) + sc.black_level_xyz
        perturbed = np.empty((2 * n, 3))
        for i in range(n):
            c = curves[i]
            p_i = round(float(np.interp(intensities[i], c["intens_inc"], c["amp_inc"])), 1)
            for j, sign in enumerate((dP, -dP)):
                i_pert = max(float(np.interp(p_i + sign, c["amp"], c["intens"], right=c["intens"][-1])), 0.0)
                c_pert = sc.wavelengths_to_XYZ(wavelengths[i], i_pert, consider_black_level=False,
                                               channels=channels[i])
                perturbed[2 * i + j] = base - contrib[i] + c_pert
        return float(np.max(delta_E_CIE2000(target_lab, self._xyz_to_lab(perturbed))))

    def _solve_4_primary_segment(self, xyY, wavelengths, frequencies, channels):
        """Four-primary member with the smallest drive-step sensitivity.

        The exact-color solutions form a segment whose endpoints are
        one-primary-off 3-primary solutions. The segment is traced by sweeping
        one channel and solving the other three; the member minimizing the
        single-step color change is found by a 13-point scan and a bounded
        refinement.
        :raises TableBounds: if no feasible endpoint exists (caller falls back)
        """
        sc = self._spectral_converter
        wavelengths = [float(w) for w in wavelengths]
        frequencies = np.array(frequencies, dtype=float)
        n = len(wavelengths)
        target_XYZ = np.clip(xyY_to_XYZ(xyY) - sc.black_level_xyz, 0, None)
        target_lab = self._xyz_to_lab(xyY_to_XYZ(xyY))
        curves = [self._channel_power_curve(f, ch) for f, ch in zip(frequencies, channels)]

        endpoints = []
        for k in range(n):
            keep = [j for j in range(n) if j != k]
            try:
                sub_I = sc.waves_n_xyY_to_ao_intesities([wavelengths[j] for j in keep], xyY,
                                                        channels=[channels[j] for j in keep])
            except (TableBounds, LinAlgError):
                continue
            sub_I = np.asarray(sub_I, dtype=float)
            if np.any(sub_I < 0):
                continue
            full_I = np.zeros(n)
            full_I[keep] = sub_I
            endpoints.append((k, full_I))
        if not endpoints:
            raise TableBounds(f"No feasible 4-primary endpoint for xyY={xyY}", max_intens=0.0,
                              asked_intens=float(target_XYZ[1]), wavelength=0)
        if len(endpoints) == 1:
            # a point, not a segment: a 3-primary drive on four channels
            return np.clip(endpoints[0][1], 0, None)

        # the two most separated endpoints span the segment
        pairs = [(a, b) for a in range(len(endpoints)) for b in range(a + 1, len(endpoints))]
        a, b = max(pairs, key=lambda ab: np.linalg.norm(endpoints[ab[0]][1] - endpoints[ab[1]][1]))
        (anchor, v1), (_, v2) = endpoints[a], endpoints[b]
        if float(v2[anchor]) <= 1e-6:
            return np.clip(v1, 0, None)
        others = [j for j in range(n) if j != anchor]
        sub_wl = [wavelengths[j] for j in others]
        sub_channels = [channels[j] for j in others]

        def snap(s):
            """Exact-color intensities at sweep position s."""
            anchor_I = (s * v1 + (1 - s) * v2)[anchor]
            target_rem = target_XYZ - sc.wavelengths_to_XYZ(wavelengths[anchor], anchor_I,
                                                            consider_black_level=False, channels=channels[anchor])
            full_I = np.zeros(n)
            full_I[anchor] = anchor_I
            full_I[others] = np.asarray(sc._solve_3_primary_for_XYZ(sub_wl, target_rem, channels=sub_channels),
                                        dtype=float)
            return np.clip(full_I, 0, None)

        def objective(s):
            try:
                return self._dP_sensitivity_fast(snap(s), wavelengths, curves, target_lab, channels=channels)
            except (TableBounds, LinAlgError, ValueError):
                return 1e6

        grid = np.linspace(0.0, 1.0, 13)
        grid_vals = [objective(s) for s in grid]
        i_best = int(np.argmin(grid_vals))
        best_s, best_val = float(grid[i_best]), float(grid_vals[i_best])
        step = grid[1] - grid[0]
        lo, hi = max(best_s - step, 0.0), min(best_s + step, 1.0)
        if hi > lo:
            res = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"maxiter": 8})
            if float(res.fun) < best_val:
                best_s = float(res.x)
        try:
            return np.clip(snap(best_s), 0, None)
        except (TableBounds, LinAlgError, ValueError):
            return np.clip(v1, 0, None)

    def xyY_to_P_f(self, xyY, frequencies):
        """Drive for a target color on the given primary frequencies (snapped to
        the calibration grid): three primaries are solved outright, four by the
        segment solver (max-min solve as fallback). Returned in device order, one
        (power, frequency) per physical channel, unused channels at power 0."""
        frequency_array = self._snap_frequencies_to_calibration_grid(frequencies)
        wavelengths = self._get_wavelengths_from_frequencies(frequency_array)
        channels = self._channel_assignment(len(wavelengths))
        sc = self._spectral_converter
        if len(wavelengths) == 4:
            try:
                intensity_array = self._solve_4_primary_segment(xyY, wavelengths, frequency_array, channels)
            except (TableBounds, LinAlgError, ValueError):
                intensity_array = sc.waves_n_xyY_to_ao_intesities(wavelengths, xyY, channels=channels)
        else:
            intensity_array = sc.waves_n_xyY_to_ao_intesities(wavelengths, xyY, channels=channels)
        if not _within_gamut(intensity_array):
            raise ValueError(f"Color {xyY} can't be reproduced with wavelengths {wavelengths}: "
                             f"negative intensities {intensity_array}")
        intensity_array = np.clip(np.asarray(intensity_array, dtype=float), 0, None)
        power_array = self._get_power(frequency_array, intensity_array, channels=channels)
        device_freq = [float(self._frequency_array[0])] * NUMBER_OF_CHANNELS
        device_power = [0.0] * NUMBER_OF_CHANNELS
        for freq_val, power_val, channel in zip(frequency_array, power_array, channels):
            device_freq[channel - 1] = float(freq_val)
            device_power[channel - 1] = float(power_val)
        return device_power, device_freq

    @staticmethod
    def _active_channels(frequency_array, power_array, channels):
        """Keep only entries with power > 0 (drops off channels in device order)."""
        active = [(float(f), float(p), ch) for f, p, ch in zip(frequency_array, power_array, channels) if p > 0]
        if not active:
            return [], [], []
        freqs, powers, chans = zip(*active)
        return list(freqs), list(powers), list(chans)

    def P_f_to_sd(self, power_array, frequency_array, channels=None):
        """Spectrum produced by a drive (default channel i+1 for entry i)."""
        if channels is None:
            channels = [i + 1 for i in range(len(frequency_array))]
        frequency_array, power_array, channels = self._active_channels(frequency_array, power_array, channels)
        wavelengths = self._get_wavelengths_from_frequencies(frequency_array)
        intensities = self._get_intensity(frequency_array, power_array, channels=channels)
        return self._spectral_converter.wavelengths_to_sd(wavelengths, intensities, channels=channels)

    def P_f_to_XYZ(self, power_array, frequency_array, channels=None):
        """CIE 1964 XYZ produced by a drive."""
        if channels is None:
            channels = [i + 1 for i in range(len(frequency_array))]
        if not self._active_channels(frequency_array, power_array, channels)[0]:
            return np.zeros(3)
        return self._spectral_converter.integrate_sd_to_XYZ(
            self.P_f_to_sd(power_array, frequency_array, channels=channels))
