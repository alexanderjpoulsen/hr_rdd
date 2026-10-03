"""Compare the existing clearance proxy with a Nathan-style flight model.

Reads prepared batted-ball data (including park fence features), integrates
trajectories from launch conditions, and evaluates the first stage in three
samples: all observations, launch angle at/below the HR median, and at/below
the HR first quartile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nathan_trajectory import PARK_ELEVATION_FT, trajectory_clearance


SAMPLE_NAMES = ["All fly balls", "HR launch angle <= median", "HR launch angle <= Q1"]


def binned_rates(df: pd.DataFrame, window: float, bin_width: float) -> pd.DataFrame:
    edges = np.arange(-window, window + bin_width, bin_width)
    sample = df[df["running_var_nathan"].between(-window, window)].copy()
    sample["bin"] = pd.cut(sample["running_var_nathan"], edges, right=False, include_lowest=True)
    grouped = sample.groupby("bin", observed=True)["home_run"].agg(["mean", "count"]).reset_index()
    if grouped.empty:
        return grouped.assign(midpoint=[], ci_lower=[], ci_upper=[])
    grouped["midpoint"] = grouped["bin"].map(lambda interval: interval.mid).astype(float)
    # Wilson score intervals for the binned HR proportions.
    n = grouped["count"].to_numpy(dtype=float)
    p = grouped["mean"].to_numpy(dtype=float)
    z = 1.96
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    grouped["ci_lower"] = center - half
    grouped["ci_upper"] = center + half
    return grouped


def summarize_sample(name: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for window in [1, 2, 5, 10, 20]:
        in_window = df["running_var_nathan"].abs().le(window)
        left = in_window & df["running_var_nathan"].lt(0)
        right = in_window & df["running_var_nathan"].ge(0)
        rows.append({
            "sample": name,
            "metric": f"n_within_pm_{window}ft",
            "value": int(in_window.sum()),
        })
        rows.append({
            "sample": name,
            "metric": f"hr_share_within_pm_{window}ft",
            "value": float(df.loc[in_window, "home_run"].mean()) if in_window.any() else np.nan,
        })
        rows.append({
            "sample": name,
            "metric": f"hr_share_left_within_pm_{window}ft",
            "value": float(df.loc[left, "home_run"].mean()) if left.any() else np.nan,
        })
        rows.append({
            "sample": name,
            "metric": f"hr_share_right_within_pm_{window}ft",
            "value": float(df.loc[right, "home_run"].mean()) if right.any() else np.nan,
        })
    observed = df["home_run"].astype(bool)
    predicted = df["running_var_nathan"].gt(0)
    rows.extend([
        {"sample": name, "metric": "n_observations", "value": len(df)},
        {"sample": name, "metric": "n_home_runs", "value": int(observed.sum())},
        {"sample": name, "metric": "home_run_launch_angle_cutoff_deg", "value": float(df.attrs["launch_angle_cutoff"])},
        {"sample": name, "metric": "false_negative_rate", "value": float((observed & ~predicted).sum() / observed.sum()) if observed.sum() else np.nan},
        {"sample": name, "metric": "false_positive_rate", "value": float((~observed & predicted).sum() / (~observed).sum()) if (~observed).sum() else np.nan},
        {"sample": name, "metric": "n_reached_fence_in_flight", "value": int(df["trajectory_status"].eq(1).sum())},
        {"sample": name, "metric": "n_landed_before_fence", "value": int(df["trajectory_status"].eq(2).sum())},
        {"sample": name, "metric": "n_trajectory_limit", "value": int(df["trajectory_status"].eq(3).sum())},
        {"sample": name, "metric": "median_nathan_running_var_ft", "value": float(df["running_var_nathan"].median())},
        {"sample": name, "metric": "median_prior_trig_running_var_ft", "value": float(df["running_var"].median())},
        {"sample": name, "metric": "correlation_nathan_vs_prior_trig", "value": float(df[["running_var_nathan", "running_var"]].corr().iloc[0, 1])},
    ])
    return rows


def summarize_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in ["home_run", "field_out", "double", "triple", "single"]:
        subset = df[df["events"].eq(event)]
        range_error = (
            subset["model_landing_distance_ft"] - pd.to_numeric(subset["hit_distance_sc"], errors="coerce")
        ).dropna()
        rows.append({
            "event": event,
            "n": len(subset),
            "share_model_clearance_positive": subset["running_var_nathan"].gt(0).mean(),
            "share_reached_fence_before_landing": subset["trajectory_status"].eq(1).mean(),
            "median_model_landing_distance_ft": subset["model_landing_distance_ft"].median(),
            "median_statcast_hit_distance_sc_ft": subset["hit_distance_sc"].median(),
            "median_model_minus_statcast_distance_ft": range_error.median(),
            "p90_model_minus_statcast_distance_ft": range_error.quantile(0.9),
        })
    return pd.DataFrame(rows)


def make_figures(sampled: list[tuple[str, pd.DataFrame]], output_dir: Path,
                 window: float, bin_width: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (name, frame) in zip(axes, sampled):
        bins = binned_rates(frame, window, bin_width)
        if not bins.empty:
            yerr = np.vstack([
                np.maximum(bins["mean"].to_numpy() - bins["ci_lower"].to_numpy(), 0.0),
                np.maximum(bins["ci_upper"].to_numpy() - bins["mean"].to_numpy(), 0.0),
            ])
            ax.errorbar(
                bins["midpoint"], bins["mean"], yerr=yerr,
                fmt="o", color="#173f5f", ecolor="#7f8c8d",
                capsize=2, markersize=4, lw=1,
            )
        ax.axvline(0, color="#d1495b", ls="--", lw=1.2)
        ax.set_title(f"{name}\nN={len(frame):,}; HR={int(frame.home_run.sum()):,}")
        ax.set_xlim(-window, window)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nathan-model clearance score (ft)")
        ax.grid(alpha=0.25, ls="--")
    axes[0].set_ylabel("Share of batted balls that were HRs")
    fig.suptitle("First Stage Using Nathan-Style Trajectory Clearance (95% Wilson CIs)")
    fig.tight_layout()
    fig.savefig(output_dir / "nathan_first_stage_launch_angle_samples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    edges = np.arange(-window, window + bin_width, bin_width)
    for ax, (name, frame) in zip(axes, sampled):
        near = frame[frame["running_var_nathan"].between(-window, window)]
        for indicator, color, label in [(1, "#d1495b", "HR"), (0, "#00798c", "Non-HR")]:
            vals = near.loc[near.home_run.eq(indicator), "running_var_nathan"]
            if len(vals):
                ax.hist(vals, bins=edges, density=True, histtype="step", lw=1.8,
                        color=color, label=f"{label} (n={len(vals):,})")
        ax.axvline(0, color="#333333", ls="--", lw=1.1)
        ax.set_title(name)
        ax.set_xlabel("Nathan-model clearance score (ft)")
        ax.grid(alpha=0.25, ls="--")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density within plotted window")
    fig.suptitle(f"Running-Variable Distribution Within +/-{window:g} ft")
    fig.tight_layout()
    fig.savefig(output_dir / "nathan_running_variable_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nathan-style running-variable construction.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "rdd_data.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--spin-rpm", type=float, default=1800.0)
    parser.add_argument("--launch-height-ft", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--plot-window", type=float, default=40.0)
    parser.add_argument("--bin-width", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    required = ["launch_speed", "launch_angle", "spray_angle_deg", "fence_dist_ft", "fence_height_ft", "home_team", "events"]
    df = df.dropna(subset=required).copy()
    df["home_run"] = df["events"].eq("home_run").astype(int)
    df["running_var"] = pd.to_numeric(df["running_var"], errors="coerce")
    hr_angles = pd.to_numeric(df.loc[df.home_run.eq(1), "launch_angle"], errors="coerce").dropna()
    median_cutoff = float(hr_angles.quantile(0.50))
    q1_cutoff = float(hr_angles.quantile(0.25))

    print(f"Integrating {len(df):,} trajectories (dt={args.dt}s; spin={args.spin_rpm:g} RPM)...")
    elevations = df["home_team"].map(PARK_ELEVATION_FT).fillna(0).to_numpy(dtype=float)
    model = trajectory_clearance(
        launch_speed_mph=pd.to_numeric(df.launch_speed, errors="coerce").to_numpy(dtype=float),
        launch_angle_deg=pd.to_numeric(df.launch_angle, errors="coerce").to_numpy(dtype=float),
        spray_angle_deg=pd.to_numeric(df.spray_angle_deg, errors="coerce").to_numpy(dtype=float),
        fence_dist_ft=pd.to_numeric(df.fence_dist_ft, errors="coerce").to_numpy(dtype=float),
        fence_height_ft=pd.to_numeric(df.fence_height_ft, errors="coerce").to_numpy(dtype=float),
        elevation_ft=elevations,
        spin_rpm=args.spin_rpm,
        launch_height_ft=args.launch_height_ft,
        dt=args.dt,
    )
    for key, values in model.items():
        df[key] = values
    df["running_var_nathan"] = df["running_var_ft"]

    definitions = [
        (SAMPLE_NAMES[0], pd.Series(True, index=df.index)),
        (SAMPLE_NAMES[1], df.launch_angle.le(median_cutoff)),
        (SAMPLE_NAMES[2], df.launch_angle.le(q1_cutoff)),
    ]
    samples = []
    summary_rows = []
    for name, mask in definitions:
        subset = df.loc[mask].copy()
        subset.attrs["launch_angle_cutoff"] = np.inf if name == SAMPLE_NAMES[0] else (
            median_cutoff if name == SAMPLE_NAMES[1] else q1_cutoff
        )
        samples.append((name, subset))
        summary_rows.extend(summarize_sample(name, subset))

    output_data = ROOT / "data" / "processed" / "rdd_data_nathan.parquet"
    df.to_parquet(output_data, index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "nathan_running_variable_summary.csv", index=False)
    summarize_events(df).to_csv(args.output_dir / "nathan_event_diagnostics.csv", index=False)
    make_figures(samples, args.output_dir / "figures", args.plot_window, args.bin_width)

    print(f"HR launch-angle median: {median_cutoff:.1f} deg; first quartile: {q1_cutoff:.1f} deg")
    print("\nRunning-variable summary by sample:")
    summary = pd.DataFrame(summary_rows)
    print(summary[summary.metric.isin([
        "n_observations", "n_home_runs", "false_negative_rate", "false_positive_rate",
        "n_reached_fence_in_flight", "n_landed_before_fence", "median_nathan_running_var_ft",
    ])].to_string(index=False))
    print(f"\nSaved modeled data: {output_data}")
    print(f"Saved plots/tables: {args.output_dir}")


if __name__ == "__main__":
    main()
