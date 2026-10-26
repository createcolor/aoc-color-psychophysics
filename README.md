# aoc-color-psychophysics

Code and data for "An acousto-optic colorimeter for color psychophysics": the
validation of the acousto-optic colorimeter (AOC) as a stimulus generator, and
the choice of its primaries for color-matching and color-difference
experiments, computed from the instrument's calibration and measurements.

```
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` runs every step and ends with `scripts/summary.py`, which prints
the results below (also saved to `output/summary.txt`). The selection runs are
independent single-threaded processes, run six at a time (`JOBS=n ./run_all.sh`
to change): about 16 minutes on 8 cores. The first run downloads the
individual-observer data (Asano et al. 2016, Zenodo record 3252742, about
7 MB) and caches it in `data/`; after that everything runs offline.

Each step can also be run on its own from this directory, with
`PYTHONPATH=.:scripts python scripts/<script> ...`; it writes to `output/`.

## Results

### 1. Validation: set color differences are reproduced

At each of the five CIE color centers the AOC alternated the center with a
test color displaced by a set step (0.5 to 2.5 ΔE₀₀, 8 directions, 40 pairs),
measured in 14 blocks over a 54-hour campaign. The commanded separations
(0.37 to 3.65 ΔE₀₀) are reproduced with a mean absolute error of 0.08 to
0.12 ΔE₀₀, and the centers within 0.41 to 0.88 ΔE₀₀ of their targets. Block
means of the pair error vary by only 0.01 to 0.07 ΔE₀₀ over the campaign.

`validation.py` — from `data/validation_campaign_2026-10-17.json`.

### 2. Primaries for color matching: the resolution criterion

Every primary set the calibration allows is solved for the target through the
device model, its drive rounded to the 0.1 % quantum, and kept if the target
and the eight colors at ΔE₀₀ = 3 around it all render. The set minimizing the
placement error plus the largest color change of a single drive quantum is
selected. Four primaries win at every center, with minimal steps of 0.08 to
0.14 ΔE₀₀, comparable to the reference display at an effective 12 bits (0.04
to 0.16).

`ring_step_selection.py --center <c> --primaries 3|4` for each center, then
`resolution_sets.py`; `display_model.py` for the display.

### 3. Primaries for color difference: the observer criterion

Each four-primary set is scored by how far the percepts of real observers (the
151 color-normal observers of Asano et al.) fall from the specified color,
each in that observer's own CIELAB. The population is split once by age into 75
observers for selection and 76 for reporting. Across the feasible sets the mean
departure varies 3- to 8-fold, so the choice matters. The selected set must
also render the ΔE₀₀ = 3 neighborhood, and within its family of drives that
match the standard observer the one with the best worst observer is adopted.
The fourth primary and the choice of drive both contribute: at yellow the
adopted drive lowers the mean from 0.92 to 0.64 ΔE₀₀ against the solver's.

`standard_anchor_selection.py --center <c>` for each center, then
`standard_anchor_gate_refine.py`; `k3_standard_anchor_best.py` for the best
three-primary sets.

### 4. The criteria compared, with the display as baseline

On the 76 held-out observers, the resolution-selected configurations depart
from the specified color by 1.9 to 4.9 ΔE₀₀ on average (up to 14.4 for one
observer), more than the reference display (1.2 to 2.5). The observer-selected
ones depart by 0.64 to 1.94, 1.5 to 6.2 times less, and below the display at
every center for the mean and for the worst observer (marginally, for the
worst, at green and red). The cost is a minimal
step larger by at most 0.40 ΔE₀₀.

From the outputs of 2 and 3; `summary.py`.

### 5. Transfer to another observer population

Configurations selected on other observers (the CIE 2015 observer, the two
Stiles & Burch observers and the Asano model over ages 20 to 70) and scored on
the 151 lose 0.6 to 1.6 ΔE₀₀ against the in-sample optimum, so the selection
should be repeated for the population an experiment serves.

`primary_selection_crossvalidate.py`.

## Data

| Path | Contents |
|---|---|
| `ao/` | The instrument's device model: calibration tables, spectral model, drive solver. Nothing here talks to hardware. |
| `ao/calibration/2026-10-14/` | The calibration: per-channel amplitude–intensity tables (`amplitude_intensity_calibration_ch*.csv`) and the X-Rite i1 Pro 2 spectrum of every band at every drive level (`wv_intens_spectra_ch*`, pickled tables). |
| `data/validation_campaign_2026-10-17.json` | The 54-hour validation campaign: for every pair of the protocol, the commanded test color and every reading of the measured center and test color (xyY), with the center error recorded by the control software. |
| `data/monitor_2026-08-28.txt` | X-Rite spectra of the reference display (EIZO CG245W) at the five color centers: the display-derived targets and the display baseline. |
| `data/eizo_primaries/` | Argyll `.sp` spectra of the display's three primaries (the display model). |
| `data/asano2015_color_normal_observers_XYZ10.npz` | Written on the first run: the 151 color-normal observers of Asano, Fairchild & Blondé (2016), 10° color-matching functions with ages, from Zenodo record 3252742 (the record links to the spreadsheets on rit-mcsl.org). |
| `data/asano_cie2006_data/` | CIE 2006 base data for the age-model observers of the transfer run. |

Conventions (`scripts/common.py`): CIE 1964 10° observer; CIELAB against the
CIE D50 spectrum at 100 cd/m², as each observer sees it; CIEDE2000; primaries
assigned to channels in descending wavelength (three on channels 2–4, four on
1–4); drive quantum 0.1; primaries at least 20 nm apart. The resolution
criterion works at the nominal CIE centers, the observer criterion at the
display-derived ones.
