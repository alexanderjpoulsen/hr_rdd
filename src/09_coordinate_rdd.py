"""09_coordinate_rdd.py
----------------------
Builds a running variable based on the direct distance (in feet) between
the Statcast hit coordinate (hc_x, hc_y) and the outfield fence, then
produces a regression discontinuity plot.

Running variable definition
---------------------------
  coord_fence_dist_ft = hc_radius_ft - fence_dist_ft

  where hc_radius_ft  = Euclidean distance from home plate to hit coordinate,
        fence_dist_ft = fence distance (ft) at the hit's spray angle.

  > 0  →  coordinate is beyond the fence  (home run territory)
  < 0  →  coordinate is inside the fence  (in-play territory)
  = 0  →  coordinate is exactly at the fence (discontinuity cutoff)

Note: this is a purely horizontal / 2-D measure.  It captures where the ball
*landed* (or was last tracked) relative to the fence, ignoring ball height.
Compare to the trig-based running_var in 02_prepare_data.py, which models
ball height at the fence.

Inputs
------
  data/processed/rdd_data_nathan.parquet   (contains hc_x, hc_y, fence_dist_ft,
                                             events, and other Statcast fields)

Outputs
-------
  output/figures/09_coordinate_rdd.png
  output/09_coordinate_rdd_estimate.csv

Usage
-----
    python src/09_coordinate_rdd.py
    python src/09_coordinate_rdd.py --window 50 --bandwidth 25 --bin-width 1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "rdd_data_nathan.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "output"

# ── MLBAM coordinate constants ────────────────────────────────────────────────
HC_HOME_X = 125.42   # pixel x of home plate in MLBAM coordinate system
HC_HOME_Y = 198.27   # pixel y of home plate
MLBAM_TO_FT = 2.495  # pixels per foot (MLBAM → feet conversion)


# ── Running variable ──────────────────────────────────────────────────────────

def add_coord_fence_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signed coordinate-to-fence distance (feet) and home_run flag.

    coord_fence_dist_ft = hc_radius_ft - fence_dist_ft
      Positive  → hit coordinate beyond the fence   (+, HR territory)
      Negative  → hit coordinate inside the fence   (-, in-play territory)
    """
    df = df.copy()

    # Convert MLBAM pixels to feet, with home plate as origin
    x_rel_ft = (pd.to_numeric(df["hc_x"], errors="coerce") - HC_HOME_X) * MLBAM_TO_FT
    y_rel_ft = (HC_HOME_Y - pd.to_numeric(df["hc_y"], errors="coerce")) * MLBAM_TO_FT

    # Radial distance from home plate to the hit coordinate
    df["hc_radius_ft"] = np.hypot(x_rel_ft, y_rel_ft)

    # Signed distance: positive = beyond fence, negative = inside
    df["coord_fence_dist_ft"] = df["hc_radius_ft"] - pd.to_numeric(
        df["fence_dist_ft"], errors="coerce"
    )

    # Binary outcome
    df["home_run"] = df["events"].eq("home_run").astype(int)

    return df


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(successes: np.ndarray, counts: np.ndarray, z: float = 1.96
              ) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score confidence interval for a proportion."""
    p = successes / counts
    denom = 1.0 + z**2 / counts
    center = (p + z**2 / (2 * counts)) / denom
    half = z * np.sqrt(p * (1 - p) / counts + z**2 / (4 * counts**2)) / denom
    return center - half, center + half


# ── RD estimate ───────────────────────────────────────────────────────────────

def fit_local_linear_rdd(df: pd.DataFrame, bandwidth: float) -> dict:
    """Triangular-kernel local-linear WLS RD estimate at the zero cutoff."""
    local = df[df["coord_fence_dist_ft"].abs().le(bandwidth)].copy()
    running = local["coord_fence_dist_ft"].to_numpy(dtype=float)
    outcome = local["home_run"].to_numpy(dtype=float)
    treatment = (running >= 0).astype(float)
    weights = 1.0 - np.abs(running) / bandwidth  # triangular kernel

    design = np.column_stack([
        np.ones(len(local)),
        treatment,
        running,
        treatment * running,
    ])
    fit = sm.WLS(outcome, design, weights=weights).fit(cov_type="HC1")

    ci = np.asarray(fit.conf_int())  # works with both ndarray and DataFrame
    return {
        "bandwidth_ft": bandwidth,
        "n": len(local),
        "n_left": int((running < 0).sum()),
        "n_right": int((running >= 0).sum()),
        "tau": float(fit.params[1]),
        "se_hc1": float(fit.bse[1]),
        "ci_lower": float(ci[1, 0]),
        "ci_upper": float(ci[1, 1]),
        "p_value": float(fit.pvalues[1]),
    }


# ── RDD plot ──────────────────────────────────────────────────────────────────

def make_rdd_plot(
    df: pd.DataFrame,
    estimate: dict,
    save_path: Path,
    window: float = 40.0,
    bin_width: float = 2.0,
    bandwidth: float = 20.0,
) -> None:
    """Binned-scatter RD plot: coord_fence_dist_ft (X) vs P(home run) (Y).

    The plot shows:
      • Binned scatter of observed HR rates with 95% Wilson CIs
      • Local-linear fit lines on each side of the cutoff (within bandwidth)
      • Vertical dashed line at the fence boundary (x = 0)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── Binned scatter ────────────────────────────────────────────────────
    sample = df[df["coord_fence_dist_ft"].between(-window, window)].copy()
    edges = np.arange(-window, window + bin_width, bin_width)
    sample["bin"] = pd.cut(
        sample["coord_fence_dist_ft"], edges, right=False, include_lowest=True
    )
    binned = (
        sample.groupby("bin", observed=True)["home_run"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if not binned.empty:
        binned["mid"] = binned["bin"].map(lambda iv: iv.mid).astype(float)

        counts = binned["count"].to_numpy(dtype=float)
        means = binned["mean"].to_numpy(dtype=float)
        lo, hi = wilson_ci(means * counts, counts)

        # Split colours by side of the cutoff
        is_right = binned["mid"] >= 0
        for mask, color, label in [
            (~is_right, "#2980b9", "Inside fence (in-play)"),
            (is_right,  "#e74c3c", "Beyond fence (HR territory)"),
        ]:
            sel = binned[mask]
            lo_s = lo[mask.to_numpy()]
            hi_s = hi[mask.to_numpy()]
            means_s = means[mask.to_numpy()]
            ax.errorbar(
                sel["mid"], sel["mean"],
                yerr=np.vstack([
                    np.maximum(means_s - lo_s, 0),
                    np.maximum(hi_s - means_s, 0),
                ]),
                fmt="o", color=color, ecolor="#aaaaaa",
                capsize=2, markersize=5, lw=1,
                label=label,
            )

    # ── Local-linear fit lines (within bandwidth) ─────────────────────────
    local = df[df["coord_fence_dist_ft"].abs().le(bandwidth)].copy()
    for side, color, grid in [
        ("left",  "#2980b9", np.linspace(-bandwidth, 0, 200)),
        ("right", "#e74c3c", np.linspace(0, bandwidth, 200)),
    ]:
        if side == "left":
            side_mask = local["coord_fence_dist_ft"] < 0
        else:
            side_mask = local["coord_fence_dist_ft"] >= 0

        x = local.loc[side_mask, "coord_fence_dist_ft"].to_numpy(dtype=float)
        y = local.loc[side_mask, "home_run"].to_numpy(dtype=float)

        if len(x) > 3:
            w = 1.0 - np.abs(x) / bandwidth  # triangular weights
            coefs = np.polyfit(x, y, deg=1, w=np.sqrt(w))
            ax.plot(grid, np.polyval(coefs, grid), color=color, lw=2.2, zorder=4)

    # ── Fence boundary ────────────────────────────────────────────────────
    ax.axvline(0, color="#333333", ls="--", lw=1.4, label="Fence boundary (cutoff)")

    # ── Annotations ───────────────────────────────────────────────────────
    tau_pp = estimate["tau"] * 100
    se_pp  = estimate["se_hc1"] * 100
    bw_ft  = estimate["bandwidth_ft"]
    ax.set_title(
        "Regression Discontinuity: Coordinate-to-Fence Distance vs Home Run\n"
        f"Local-linear jump at fence = {tau_pp:+.1f} pp  "
        f"(HC1 SE {se_pp:.1f} pp,  BW ±{bw_ft:g} ft,  "
        f"N={estimate['n']:,})",
        fontsize=11,
    )
    ax.set_xlabel("Distance from hit coordinate to outfield fence (ft)\n"
                  "Negative = inside park   |   Positive = beyond fence",
                  fontsize=10)
    ax.set_ylabel("P(Home Run)", fontsize=10)
    ax.set_xlim(-window, window)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RDD plot: coordinate-to-fence distance (ft) vs home run."
    )
    parser.add_argument("--input",      type=Path, default=DEFAULT_INPUT,
                        help="Input parquet file (default: rdd_data_nathan.parquet)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: output/)")
    parser.add_argument("--window",     type=float, default=40.0,
                        help="Half-width of x-axis window (ft)")
    parser.add_argument("--bandwidth",  type=float, default=20.0,
                        help="Local-linear estimation bandwidth (ft)")
    parser.add_argument("--bin-width",  type=float, default=2.0,
                        help="Bin width for scatter (ft)")
    args = parser.parse_args()

    # ── Setup output dirs ─────────────────────────────────────────────────
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows loaded")

    # ── Build running variable ────────────────────────────────────────────
    print("Computing coordinate-to-fence distance ...")
    df = add_coord_fence_distance(df)
    df = df.dropna(subset=["coord_fence_dist_ft", "home_run"]).copy()
    print(f"  {df['coord_fence_dist_ft'].notna().sum():,} rows with valid running variable")

    # Quick sanity check: HR rate by sign
    for label, mask in [
        ("Beyond fence (>0)", df["coord_fence_dist_ft"] >= 0),
        ("Inside fence (< 0)", df["coord_fence_dist_ft"] < 0),
    ]:
        n = mask.sum()
        hr_rate = df.loc[mask, "home_run"].mean() * 100
        print(f"  {label}: N={n:,}, HR rate={hr_rate:.1f}%")

    # ── RD estimate ───────────────────────────────────────────────────────
    print(f"\nFitting local-linear RDD (bandwidth ±{args.bandwidth:.0f} ft) ...")
    estimate = fit_local_linear_rdd(df, args.bandwidth)
    print(f"  Jump (tau)      = {estimate['tau']*100:+.2f} pp")
    print(f"  HC1 SE          = {estimate['se_hc1']*100:.2f} pp")
    print(f"  95% CI          = [{estimate['ci_lower']*100:+.2f}, {estimate['ci_upper']*100:+.2f}] pp")
    print(f"  p-value         = {estimate['p_value']:.4f}")
    print(f"  N (in BW)       = {estimate['n']:,}  (left={estimate['n_left']:,}, right={estimate['n_right']:,})")

    # ── Save estimate ─────────────────────────────────────────────────────
    est_path = args.output_dir / "09_coordinate_rdd_estimate.csv"
    pd.DataFrame([estimate]).to_csv(est_path, index=False)
    print(f"\nSaved estimate → {est_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_path = figure_dir / "09_coordinate_rdd.png"
    make_rdd_plot(
        df, estimate, plot_path,
        window=args.window,
        bin_width=args.bin_width,
        bandwidth=args.bandwidth,
    )


if __name__ == "__main__":
    main()
