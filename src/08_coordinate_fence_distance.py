"""Build a signed HC-coordinate-to-fence distance and first-stage plot.

The signed score is the radial distance from home plate to the Statcast hit
coordinate minus the fence radius at that coordinate's spray angle. Positive
means the coordinate lies beyond the modeled fence ray; negative means it lies
inside. This is not a 3D fence-clearance measure.
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "rdd_data_nathan.parquet"
DEFAULT_OUTPUT = ROOT / "output"
HC_HOME_X = 125.42
HC_HOME_Y = 198.27
MLBAM_TO_FT = 2.495


def add_coordinate_fence_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    x_rel_ft = (pd.to_numeric(df["hc_x"], errors="coerce") - HC_HOME_X) * MLBAM_TO_FT
    y_rel_ft = (HC_HOME_Y - pd.to_numeric(df["hc_y"], errors="coerce")) * MLBAM_TO_FT
    df["hc_radius_ft"] = np.hypot(x_rel_ft, y_rel_ft)
    df["coordinate_fence_distance_ft"] = df["hc_radius_ft"] - pd.to_numeric(
        df["fence_dist_ft"], errors="coerce",
    )
    df["home_run"] = df["events"].eq("home_run").astype(int)
    return df


def wilson_interval(successes: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = 1.96
    p = successes / counts
    denom = 1.0 + z**2 / counts
    center = (p + z**2 / (2 * counts)) / denom
    half = z * np.sqrt(p * (1 - p) / counts + z**2 / (4 * counts**2)) / denom
    return center - half, center + half


def fit_local_jump(df: pd.DataFrame, bandwidth: float) -> dict:
    local = df[df.coordinate_fence_distance_ft.abs().le(bandwidth)].copy()
    running = local.coordinate_fence_distance_ft.to_numpy(dtype=float)
    outcome = local.home_run.to_numpy(dtype=float)
    treatment = (running >= 0).astype(float)
    weights = 1.0 - np.abs(running) / bandwidth
    design = np.column_stack([np.ones(len(local)), treatment, running, treatment * running])
    fit = sm.WLS(outcome, design, weights=weights).fit(cov_type="HC1")
    return {
        "bandwidth_ft": bandwidth,
        "n": len(local),
        "n_left": int((running < 0).sum()),
        "n_right": int((running >= 0).sum()),
        "tau": float(fit.params[1]),
        "se_hc1": float(fit.bse[1]),
        "ci_lower": float(fit.conf_int()[1, 0]),
        "ci_upper": float(fit.conf_int()[1, 1]),
        "p_value": float(fit.pvalues[1]),
    }


def make_plot(df: pd.DataFrame, path: Path, window: float, bin_width: float,
              bandwidth: float, estimate: dict) -> None:
    sample = df[df.coordinate_fence_distance_ft.between(-window, window)].copy()
    edges = np.arange(-window, window + bin_width, bin_width)
    sample["bin"] = pd.cut(sample.coordinate_fence_distance_ft, edges,
                           right=False, include_lowest=True)
    binned = sample.groupby("bin", observed=True).home_run.agg(["mean", "count"]).reset_index()
    if not binned.empty:
        binned["mid"] = binned["bin"].map(lambda interval: interval.mid).astype(float)
        lower, upper = wilson_interval(binned["mean"].to_numpy(), binned["count"].to_numpy(dtype=float))
        ax.errorbar(
            binned["mid"], binned["mean"],
            yerr=np.vstack([np.maximum(binned["mean"].to_numpy() - lower, 0),
                            np.maximum(upper - binned["mean"].to_numpy(), 0)]),
            fmt="o", color="#173f5f", ecolor="#7f8c8d", capsize=2,
            markersize=4, lw=1, label="Binned HR share (95% Wilson CI)",
        )

    local = df[df.coordinate_fence_distance_ft.abs().le(bandwidth)].copy()
    for side, color, grid in [
        ("left", "#00798c", np.linspace(-bandwidth, 0, 100)),
        ("right", "#d1495b", np.linspace(0, bandwidth, 100)),
    ]:
        mask = local.coordinate_fence_distance_ft.lt(0) if side == "left" else local.coordinate_fence_distance_ft.ge(0)
        x = local.loc[mask, "coordinate_fence_distance_ft"].to_numpy(dtype=float)
        y = local.loc[mask, "home_run"].to_numpy(dtype=float)
        if len(x) > 1:
            weights = np.sqrt(1.0 - np.abs(x) / bandwidth)
            slope, intercept = np.polyfit(x, y, deg=1, w=weights)
            ax.plot(grid, intercept + slope * grid, color=color, lw=2,
                    label=f"Local linear fit ({side})")

    ax.axvline(0, color="#333333", ls="--", lw=1.2, label="Fence boundary")
    ax.set_xlim(-window, window)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Signed radial distance from HC coordinate to fence (ft)")
    ax.set_ylabel("P(observed home run)")
    ax.set_title(
        "Home Run First Stage by Coordinate-to-Fence Distance\n"
        f"Local-linear jump = {estimate['tau'] * 100:.1f} pp "
        f"(HC1 SE {estimate['se_hc1'] * 100:.1f}; BW +/-{bandwidth:g} ft)"
    )
    ax.grid(alpha=0.25, ls="--")
    ax.legend(fontsize=8)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HR probability by signed coordinate-to-fence distance.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=float, default=40.0)
    parser.add_argument("--bandwidth", type=float, default=20.0)
    parser.add_argument("--bin-width", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    df = add_coordinate_fence_distance(df)
    df = df.dropna(subset=["coordinate_fence_distance_ft", "home_run"]).copy()
    estimate = fit_local_jump(df, args.bandwidth)

    output_data = ROOT / "data" / "processed" / "rdd_data_coordinate_fence.parquet"
    output_data.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_data, index=False)
    make_plot(
        df, figure_dir / "first_stage_coordinate_fence_distance.png",
        args.window, args.bin_width, args.bandwidth, estimate,
    )

    summary = []
    for label, mask in [("all", pd.Series(True, index=df.index)),
                        ("inside", df.coordinate_fence_distance_ft.lt(0)),
                        ("outside", df.coordinate_fence_distance_ft.ge(0))]:
        sample = df.loc[mask]
        summary.append({
            "group": label,
            "n": len(sample),
            "home_runs": int(sample.home_run.sum()),
            "home_run_share": sample.home_run.mean(),
            "median_signed_distance_ft": sample.coordinate_fence_distance_ft.median(),
        })
    pd.DataFrame(summary).to_csv(args.output_dir / "coordinate_fence_distance_summary.csv", index=False)
    pd.DataFrame([estimate]).to_csv(args.output_dir / "coordinate_fence_distance_rd.csv", index=False)

    print("Coordinate-to-fence first-stage estimate:")
    print(pd.Series(estimate).to_string())
    print("\nOutcome shares by signed coordinate distance:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved plot: {figure_dir / 'first_stage_coordinate_fence_distance.png'}")
    print(f"Saved data: {output_data}")


if __name__ == "__main__":
    main()
