# Home Run Fuzzy Regression Discontinuity Design

A causal inference project studying how home runs affect pitcher and team behavior,
using a Fuzzy RDD where the running variable is the ball's clearance (or shortfall)
over the outfield fence.

---

## Research Design

The key identification challenge: home runs are not random — they happen to good hitters
against bad pitchers in favorable counts. We cannot simply compare outcomes after HRs vs.
non-HRs because of selection.

The RDD solves this: balls that just barely clear the fence vs. balls that just miss it
are as-good-as-randomly assigned to HR / non-HR status, conditional on how close they were
to the boundary. Differences in pitcher and team behavior afterward can be attributed to
the home run itself.

**Running variable:**
```
running_var = z_at_fence − fence_height_ft
```
where `z_at_fence` is the ball's computed height when it reaches the outfield fence.
- `running_var > 0` → ball clears the fence → home run (treatment = 1)
- `running_var < 0` → ball does not clear → no home run (treatment = 0)
- `running_var ≈ 0` → the regression discontinuity boundary

**Fuzzy RDD:** Because `z_at_fence` is computed from a physics model (not directly
measured), there is measurement error in the running variable. The jump in P(HR) at
zero is large but not perfectly sharp, making this a Fuzzy RDD.

---

## Data Sources

| Source | Contents |
|---|---|
| Baseball Savant (via `pybaseball`) | Statcast pitch-level data 2015–present |
| `data/park_dimensions.csv` | Fence distances and heights by spray angle for all 30 MLB parks |
| Stadium elevation lookup | Hard-coded in `02_prepare_data.py` from USGS data |

**Statcast era:** 2015–present (full 3D tracking via Hawk-Eye/Trackman).

---

## Physics Model

The current production running variable uses a reduced-form trajectory
approximation rather than a full Alan Nathan RK45 trajectory integration. The
ball's descending angle is set to 1.1 times its launch angle, reflecting a
rough 10% steepening from drag and Magnus effects. The multiplier is exposed as
`--steepness-factor` for sensitivity analysis; `1.0` reproduces the symmetric
vacuum approximation.

The full Alan Nathan trajectory model integrates the equations of motion with
drag and Magnus (backspin) lift forces. The exploratory comparison added here
uses an RK4 implementation of a Nathan/Statcast-calibrated coefficient form; it
is not a verified reproduction of Nathan's spreadsheet calculator:

```
ma = F_drag + F_Magnus + mg
F_drag   = −½ρ A C_D(v) · v² · v̂
F_Magnus = +½ρ A C_L    · v² · (ω̂ × v̂)
```

**Key inputs for a full implementation:**
- `launch_speed` — exit velocity (mph)
- `launch_angle` — vertical launch angle (degrees)
- Spray angle — derived from `hc_x`, `hc_y` coordinates
- Air density — computed from stadium elevation
- Backspin — assumed 1800 RPM (Nathan calibration default; public Statcast does not
  include batted-ball spin)

The exploratory implementation is in `src/nathan_trajectory.py` and
`src/05_nathan_physics_comparison.py`. It assumes 1,800 RPM pure backspin, zero
wind, and approximate standard air density adjusted for park elevation. It does
not replace the production running-variable method; see
`docs/statcast_distance_and_coordinates.md` for definitions, results, and
limitations before using its score for inference.

**References:**
- Nathan (2017), *The Physics Teacher* 55:134
- Nathan 3D Trajectory Calculator: http://baseball.physics.illinois.edu/trajectory-calculator-new3D.html

---

## Project Structure

```
hr_rdd/
├── data/
│   ├── raw/                    # Per-season Statcast parquet files (01_download output)
│   ├── processed/              # Analysis-ready dataset with running variable (02_prepare output)
│   └── park_dimensions.csv     # Fence distance/height by spray angle for all 30 parks
├── src/
│   ├── 01_download_statcast.py # Download Statcast data via pybaseball
│   ├── 02_prepare_data.py      # Build running variable (spray angle → physics model)
│   ├── 03_first_stage.py       # First-stage RDD: does running_var predict HR treatment?
│   ├── 04_local_randomization.py # Balance diagnostics in a local window
│   ├── 05_nathan_physics_comparison.py # Exploratory trajectory comparison
│   └── nathan_trajectory.py    # Drag/Magnus flight integration
├── output/
│   ├── figures/                # First-stage and trajectory-comparison plots
│   └── tables/                 # Results tables (CSV)
├── docs/
│   └── statcast_distance_and_coordinates.md # Field definitions and model caveats
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Download Statcast data

```bash
python src/01_download_statcast.py
```

Downloads all regular-season fly balls 2015–2024 from Baseball Savant via `pybaseball`.
Saves per-season parquet files to `data/raw/statcast_{year}.parquet` and a combined
`data/raw/statcast_all.parquet`.

**Runtime:** ~30–60 minutes for the full Statcast era (2015–2024). `pybaseball` caches
data on disk, so subsequent runs are instant.

**Options:**
```
--start-year 2015    # First season to download (default: 2015)
--end-year 2024      # Last season to download (default: 2024)
--no-cache           # Force fresh download, bypass pybaseball disk cache
```

### Step 2 — Build the running variable

```bash
python src/02_prepare_data.py
```

Computes `running_var` for each fly ball using the reduced-form trig approximation and saves
`data/processed/rdd_data.parquet`.

The default descent angle is 1.1 times launch angle. This step does not run an
ODE physics model.

**Options:**
```
--input data/raw/statcast_all.parquet
--output data/processed/rdd_data.parquet
--park-dims data/park_dimensions_geom.csv
--steepness-factor 1.0 # Optional sensitivity; 1.1 is the current default
```

At the end of step 2, a validation table is printed:
```
RUNNING VARIABLE VALIDATION
  True Positives  (HR, model says HR):      ...%
  True Negatives  (no HR, model says no):   ...%
  False Positives (no HR, model says HR):   ...%
  False Negatives (HR, model says no):      ...%
  Overall accuracy: ...%
```

This validation is a diagnostic of the reduced-form score, not evidence that it measures
physical fence clearance without error.

### Step 3 — Estimate the first stage

```bash
python src/03_first_stage.py
```

Estimates the jump in P(home run) at `running_var = 0`.

**Output figures** (`output/figures/`):
- `first_stage_rdd.png` — Binned scatter of HR rate vs. running_var with RD fit lines
- `density_test.png` — McCrary density test (checks for bunching at cutoff)
- `first_stage_bw_sensitivity.png` — First-stage estimate across a range of bandwidths
- `covariate_balance.png` — Covariate continuity tests at the cutoff

**Output tables** (`output/tables/`):
- `first_stage_main.csv` — Main first-stage estimate (rdrobust CCT)
- `first_stage_bandwidth_sensitivity.csv` — Sensitivity across bandwidths
- `covariate_balance.csv` — Covariate balance tests

**Options:**
```
--bandwidth 15      # Override CCT bandwidth with manual value (ft)
--max-rv 50         # Drop |running_var| > this (outlier cap)
```

### Exploratory Nathan-style trajectory comparison

With the prepared dataset available, run:

```bash
python src/05_nathan_physics_comparison.py
```

This writes a separate `data/processed/rdd_data_nathan.parquet`, plots first-stage
patterns for all fly balls and the below-median / below-Q1 HR-launch-angle samples,
and event-level diagnostics under `output/`. It does not overwrite the production
running variable. This approximation assumes 1,800 RPM pure backspin, no wind, and
approximate park air density; it is exploratory and not calibrated for causal use.
See [docs/statcast_distance_and_coordinates.md](docs/statcast_distance_and_coordinates.md)
for field definitions, results, and limitations.

---

## Interpreting the First Stage

The first stage answers: **"Does the running variable predict home run status?"**

A valid Fuzzy RDD requires a large, significant jump in treatment probability at the
cutoff. We expect:

- **P(HR | running_var > 0) ≈ 85–95%** (not 100% due to physics model error)
- **P(HR | running_var ≤ 0) ≈ 5–15%** (some false negatives)
- The jump Δ = P(HR | right) − P(HR | left) should be ~70–90 percentage points
- The jump should be stable across bandwidths
- Pre-determined covariates (inning, score, runners) should NOT jump at the cutoff
- The running variable density should be smooth at zero (no bunching)

A strong first stage (large Δ, small SE) validates that the running variable is a
credible instrument for the Fuzzy RDD in later stages.

---

## Planned Next Steps

- **Stage 2:** Second-stage RDD — causal effect of HR on pitcher velocity, location,
  and pitch type in the next 1–5 pitches
- **Stage 3:** Pitching change probability — does a near-fence HR trigger an earlier
  hook?
- **Stage 4:** Win probability and run scoring downstream effects
- **Robustness:** Noack-Rothe (2021) noise-corrected estimates; sensitivity to backspin
  assumption (1500–2200 RPM); weather-adjusted air density from Open-Meteo API

---

## Key References

- Nathan, A. (2017). Statcast and the Baseball Trajectory Calculator. *The Physics Teacher*, 55, 134.
- Calonico, Cattaneo & Titiunik (2014). Robust Nonparametric Confidence Intervals for
  Regression-Discontinuity Designs. *Econometrica*, 82(6), 2295–2326.
- McCrary, J. (2008). Manipulation of the Running Variable in the Regression Discontinuity
  Design. *Journal of Econometrics*, 142(2), 698–714.
- Noack, C. & Rothe, C. (2021). Bias-Aware Inference in Fuzzy Regression Discontinuity Designs.
  arXiv:2004.09458.
