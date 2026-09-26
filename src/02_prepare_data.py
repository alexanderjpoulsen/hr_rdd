"""
02_prepare_data.py
------------------
Constructs the running variable for the HR Fuzzy RDD using trig geometry:

    running_var = (hit_distance_sc − fence_dist_ft) × tan(launch_angle) − fence_height_ft

Derivation
----------
Statcast's hit_distance_sc is the projected horizontal distance the ball would
travel if unobstructed.  On the descending leg of the trajectory, the ball has
(hit_distance_sc − fence_dist_ft) feet of horizontal travel remaining when it
passes the fence.  Assuming the descent angle equals the launch angle:

    h_at_fence = (hit_distance_sc − fence_dist_ft) × tan(launch_angle)

    running_var = h_at_fence − fence_height_ft

  running_var > 0  →  ball clears fence  →  home run
  running_var < 0  →  ball does not clear
  running_var ≈ 0  →  regression discontinuity boundary

This formula works uniformly for both cases:
  - Ball lands beyond fence (hit_distance_sc > fence_dist_ft): h_at_fence > 0
  - Ball lands short of fence (hit_distance_sc < fence_dist_ft): h_at_fence < 0,
    making running_var strongly negative.

Assumption note: The descent-equals-launch-angle assumption is exact for a
symmetric parabola (vacuum).  With drag and Magnus lift the descent angle is
~5–15% steeper than the launch angle for typical HR trajectories.  A flag
--steepness-factor (default 1.0) allows applying a multiplicative correction
to tan(launch_angle) to account for this if needed.

Pipeline
--------
  1.  Load raw Statcast fly balls (output of 01_download_statcast.py)
  2.  Compute spray angle from hc_x, hc_y
  3.  Join park_dimensions.csv → fence_dist_ft, fence_height_ft
  4.  Apply trig formula → running_var
  5.  Validate: sign(running_var) vs events == 'home_run'
  6.  Save analysis-ready parquet to data/processed/

Usage
-----
    python src/02_prepare_data.py
    python src/02_prepare_data.py --steepness-factor 1.1   # 10% steeper descent
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Spray angle ───────────────────────────────────────────────────────────────
def compute_spray_angle(hc_x: float, hc_y: float) -> float:
    """
    Spray angle in degrees from straight-away center field.
    Negative = left field, Positive = right field.

    MLBAM pixel convention: home plate ≈ (125.42, 198.27).
      x_rel = hc_x − 125.42   (positive → first-base / RF side)
      y_rel = 198.27 − hc_y   (positive → away from plate / CF direction)
    """
    x = hc_x - 125.42
    y = 198.27 - hc_y
    return np.degrees(np.arctan2(x, y))


# ── Park dimension lookup ────────────────────────────────────────────────────

def load_park_dimensions(csv_path: Path) -> pd.DataFrame:
    """
    Load park dimensions CSV.  Supports both the old hand-coded format
    (team column, coarse spray-angle segments) and the new GeomMLBStadiums
    format (team_abbr column, fine 0.5° resolution rows).
    """
    df = pd.read_csv(csv_path, comment="#")
    df = df.dropna(subset=[c for c in ["team_abbr", "team"] if c in df.columns])
    # Normalise: always call the team identifier column 'team_abbr'
    if "team_abbr" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "team_abbr"})
    return df


def lookup_fence(
    home_team: str,
    game_year: int,
    spray_angle_deg: float,
    park_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Return (fence_dist_ft, fence_height_ft) for a given team/year/spray angle.

    For fine-resolution tables (0.5° rows from GeomMLBStadiums) we find the
    two nearest angle rows and linearly interpolate.
    For coarse tables (segment rows) we pick the matching segment.
    Falls back to the nearest available angle if out of range.
    """
    mask = (
        (park_df["team_abbr"] == home_team)
        & (park_df["year_start"] <= game_year)
        & (park_df["year_end"] >= game_year)
    )
    park_rows = park_df[mask].copy()

    if park_rows.empty:
        return np.nan, np.nan

    park_rows = park_rows.sort_values("spray_angle_deg")
    angles = park_rows["spray_angle_deg"].values
    dists  = park_rows["fence_dist_ft"].values
    heights = park_rows["fence_height_ft"].values

    # Interpolate distance; take height from nearest row
    dist = float(np.interp(spray_angle_deg, angles, dists))

    nearest_idx = int(np.argmin(np.abs(angles - spray_angle_deg)))
    height = float(heights[nearest_idx])

    return dist, height


def add_park_features(df: pd.DataFrame, park_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised park dimension join."""
    results = [
        lookup_fence(row["home_team"], int(row["game_year"]), row["spray_angle_deg"], park_df)
        for _, row in df.iterrows()
    ]
    dist, height = zip(*results)
    df = df.copy()
    df["fence_dist_ft"] = dist
    df["fence_height_ft"] = height
    return df


# ── Trig running variable ─────────────────────────────────────────────────────

def compute_running_variable(
    df: pd.DataFrame,
    steepness_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Add h_at_fence_ft and running_var columns.

    h_at_fence_ft  = (hit_distance_sc − fence_dist_ft) × tan(launch_angle × steepness_factor)
    running_var    = h_at_fence_ft − fence_height_ft

    Parameters
    ----------
    steepness_factor : multiplier on the descent angle to account for the ball
        descending more steeply than it was launched (due to drag).
        1.0 = symmetric assumption (launch angle = descent angle).
        1.1 = descent 10% steeper than launch.
    """
    df = df.copy()

    # tan of the (possibly adjusted) launch angle
    # Clamp launch_angle to (0°, 85°) to avoid tan blowup and nonsensical values
    la_clamped = df["launch_angle"].clip(lower=1.0, upper=85.0)
    tan_angle = np.tan(np.radians(la_clamped * steepness_factor))

    horizontal_remaining = df["hit_distance_sc"] - df["fence_dist_ft"]
    df["h_at_fence_ft"] = horizontal_remaining * tan_angle
    df["running_var"] = df["h_at_fence_ft"] - df["fence_height_ft"]

    return df


# ── Validation ───────────────────────────────────────────────────────────────

def validate_running_variable(df: pd.DataFrame) -> None:
    """
    Compare sign(running_var) to observed events == 'home_run'.
    The sharper the discontinuity at 0, the better the running variable.
    """
    HR_EVENTS = {"home_run"}
    NON_HR_EVENTS = {
        "field_out", "single", "double", "triple",
        "sac_fly", "sac_fly_double_play",
        "force_out", "grounded_into_double_play",
        "double_play", "fielders_choice", "fielders_choice_out",
    }
    valid = df.dropna(subset=["running_var", "events"])
    valid = valid[valid["events"].isin(HR_EVENTS | NON_HR_EVENTS)].copy()

    actual_hr = valid["events"].isin(HR_EVENTS)
    predicted_hr = valid["running_var"] > 0

    n = len(valid)
    n_hr = actual_hr.sum()
    n_non = n - n_hr
    tp = (predicted_hr & actual_hr).sum()
    tn = (~predicted_hr & ~actual_hr).sum()
    fp = (predicted_hr & ~actual_hr).sum()
    fn = (~predicted_hr & actual_hr).sum()

    print("\n" + "=" * 60)
    print("RUNNING VARIABLE VALIDATION")
    print("=" * 60)
    print(f"  Total observations:               {n:>8,}")
    print(f"  Actual HRs:                       {n_hr:>8,}")
    print(f"  Actual non-HRs:                   {n_non:>8,}")
    print(f"")
    print(f"  True Positives  (HR, pred HR):    {tp:>8,}  ({100*tp/n_hr:.1f}% of HRs)")
    print(f"  True Negatives  (no HR, pred no): {tn:>8,}  ({100*tn/n_non:.1f}% of non-HRs)")
    print(f"  False Positives (no HR, pred HR): {fp:>8,}  ({100*fp/n_non:.1f}% of non-HRs)")
    print(f"  False Negatives (HR, pred no):    {fn:>8,}  ({100*fn/n_hr:.1f}% of HRs)")
    print(f"")
    print(f"  Overall accuracy:                 {100*(tp+tn)/n:.2f}%")
    print("=" * 60)

    for bw in [2, 5, 10]:
        near = valid[valid["running_var"].abs() <= bw]
        if len(near) > 0:
            acc = ((near["events"].isin(HR_EVENTS)) == (near["running_var"] > 0)).mean()
            hr_rate_left  = near[near["running_var"] <= 0]["events"].isin(HR_EVENTS).mean()
            hr_rate_right = near[near["running_var"] >  0]["events"].isin(HR_EVENTS).mean()
            print(
                f"  Within ±{bw:>2} ft: N={len(near):>6,} | acc={100*acc:.1f}% | "
                f"HR rate: left={100*hr_rate_left:.1f}%, right={100*hr_rate_right:.1f}% | "
                f"jump={100*(hr_rate_right - hr_rate_left):.1f} pp"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build RDD running variable from Statcast hit_distance_sc + trig geometry."
    )
    parser.add_argument("--input",       type=Path, default=RAW_DIR / "statcast_all.parquet")
    parser.add_argument("--output",      type=Path, default=PROCESSED_DIR / "rdd_data.parquet")
    parser.add_argument("--park-dims",   type=Path, default=ROOT / "data" / "park_dimensions_geom.csv")
    parser.add_argument(
        "--steepness-factor", type=float, default=1.0,
        help="Multiplier on launch angle for descent (1.0 = symmetric, 1.1 = 10%% steeper).",
    )
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows loaded")

    # Drop rows without the required inputs
    required = ["launch_angle", "hit_distance_sc", "hc_x", "hc_y",
                "home_team", "game_year", "events"]
    before = len(df)
    df = df.dropna(subset=required).copy()
    print(f"  {len(df):,} rows with all required fields ({before - len(df):,} dropped)")

    # ── Spray angle ──────────────────────────────────────────────────────────
    print("Computing spray angles ...")
    df["spray_angle_deg"] = df.apply(
        lambda r: compute_spray_angle(r["hc_x"], r["hc_y"]), axis=1
    )
    df = df[df["spray_angle_deg"].between(-50, 50)].copy()
    print(f"  {len(df):,} rows after ±50° spray angle filter")

    # ── Park dimensions ──────────────────────────────────────────────────────
    print(f"Loading park dimensions from {args.park_dims} ...")
    park_df = load_park_dimensions(args.park_dims)
    print(f"  {len(park_df):,} fence rows for {park_df['team_abbr'].nunique()} teams")

    print("Joining fence distances and heights ...")
    df = add_park_features(df, park_df)
    n_miss = df["fence_dist_ft"].isna().sum()
    if n_miss:
        print(f"  WARNING: {n_miss:,} rows with no park dimension match — will have NaN running_var")
    print(f"  {df['fence_dist_ft'].notna().sum():,} rows with park dimensions")

    # ── Trig running variable ─────────────────────────────────────────────────
    sf = args.steepness_factor
    label = f"steepness_factor={sf:.2f}"
    print(f"\nComputing running variable ({label}) ...")
    print("  Formula: (hit_distance_sc − fence_dist) × tan(launch_angle) − fence_height")
    df = compute_running_variable(df, steepness_factor=sf)

    n_valid = df["running_var"].notna().sum()
    print(f"  {n_valid:,} rows with valid running_var")

    # ── Treatment indicator ───────────────────────────────────────────────────
    df["home_run"]     = (df["events"] == "home_run").astype(int)
    # predicted_hr is nullable where running_var is NaN
    df["predicted_hr"] = np.where(
        df["running_var"].isna(), np.nan, (df["running_var"] > 0).astype(float)
    )

    # ── Validate ─────────────────────────────────────────────────────────────
    validate_running_variable(df)

    # ── Distribution summary ──────────────────────────────────────────────────
    print(f"\nRunning variable summary:")
    print(df["running_var"].describe().round(2))
    print(f"\nHR rate by running_var sign:")
    print(f"  running_var > 0: {100*df.loc[df.running_var > 0,'home_run'].mean():.1f}%  "
          f"(N={( df.running_var > 0).sum():,})")
    print(f"  running_var ≤ 0: {100*df.loc[df.running_var <= 0,'home_run'].mean():.1f}%  "
          f"(N={(df.running_var <= 0).sum():,})")

    # ── Binned HR rates near the cutoff (sanity check for sharpness) ──────────
    print("\nBinned HR rate near cutoff (5-ft bins):")
    near = df[df["running_var"].between(-30, 30)].copy()
    near["rv_bin"] = pd.cut(near["running_var"], bins=np.arange(-30, 31, 5))
    tbl = (
        near.groupby("rv_bin", observed=True)["home_run"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "hr_rate", "count": "n"})
    )
    tbl["hr_rate_pct"] = (tbl["hr_rate"] * 100).round(1)
    print(tbl[["hr_rate_pct", "n"]].to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_parquet(args.output, index=False)
    print(f"\nSaved → {args.output}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
