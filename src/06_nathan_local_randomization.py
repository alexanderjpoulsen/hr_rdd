"""Test local-randomization covariate balance using the Nathan score.

Consumes data/processed/rdd_data_nathan.parquet and reports balance for the
full sample and HR-launch-angle restricted samples at +/-1, +/-2, and +/-5 ft.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "rdd_data_nathan.parquet"
OUTPUT = ROOT / "output"
WINDOWS = [1.0, 2.0, 5.0]


def load_balance_module():
    path = Path(__file__).with_name("04_local_randomization.py")
    spec = importlib.util.spec_from_file_location("local_randomization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    balance = load_balance_module()
    df = pd.read_parquet(INPUT)
    df = balance.add_balance_features(df)
    valid_events = df["events"].isin(balance.CONTROL_EVENTS | {"home_run"})
    df = df[valid_events & df["running_var_nathan"].notna()].copy()
    df["running_var"] = pd.to_numeric(df["running_var_nathan"], errors="coerce")

    hr_launch_angles = pd.to_numeric(
        df.loc[df.home_run.eq(1), "launch_angle"], errors="coerce",
    ).dropna()
    median_la = float(hr_launch_angles.quantile(0.50))
    q1_la = float(hr_launch_angles.quantile(0.25))
    samples = [
        ("Full sample", df),
        (f"LA <= HR median ({median_la:g} deg)", df[df.launch_angle.le(median_la)]),
        (f"LA <= HR Q1 ({q1_la:g} deg)", df[df.launch_angle.le(q1_la)]),
    ]

    numeric_rows = []
    categorical_rows = []
    summary_rows = []
    for sample_name, sample in samples:
        for window in WINDOWS:
            local = sample[sample.running_var.abs().le(window)].copy()
            numeric = balance.numeric_balance(local)
            if not numeric.empty:
                numeric.insert(0, "window_ft", window)
                numeric.insert(0, "sample", sample_name)
                numeric_rows.extend(numeric.to_dict("records"))
            categorical = balance.categorical_balance(local)
            if not categorical.empty:
                categorical.insert(0, "window_ft", window)
                categorical.insert(0, "sample", sample_name)
                categorical_rows.extend(categorical.to_dict("records"))

            abs_smd = numeric["standardized_mean_difference"].abs()
            categorical_tests = (
                categorical[["covariate", "omnibus_p"]].drop_duplicates()
                if not categorical.empty else pd.DataFrame(columns=["covariate", "omnibus_p"])
            )
            summary_rows.append({
                "sample": sample_name,
                "window_ft": window,
                "n": len(local),
                "home_runs": int(local.home_run.sum()),
                "controls": int(local.home_run.eq(0).sum()),
                "home_run_share": local.home_run.mean() if len(local) else np.nan,
                "max_abs_numeric_smd": abs_smd.max() if len(abs_smd) else np.nan,
                "numeric_covariates_abs_smd_ge_0_1": int(abs_smd.ge(0.1).sum()),
                "categorical_tests_p_lt_0_05": int(categorical_tests.omnibus_p.lt(0.05).sum()),
            })

    numeric_df = pd.DataFrame(numeric_rows)
    categorical_df = pd.DataFrame(categorical_rows)
    summary_df = pd.DataFrame(summary_rows)
    numeric_df.to_csv(OUTPUT / "nathan_local_randomization_balance.csv", index=False)
    categorical_df.to_csv(OUTPUT / "nathan_local_randomization_categorical.csv", index=False)
    summary_df.to_csv(OUTPUT / "nathan_local_randomization_summary.csv", index=False)

    # Forest plots for the requested primary +/-1-ft window.
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=True)
    for ax, (sample_name, _) in zip(axes, samples):
        plot_df = numeric_df[
            numeric_df["sample"].eq(sample_name) & numeric_df["window_ft"].eq(1.0)
        ].dropna(subset=["standardized_mean_difference"]).copy()
        plot_df = plot_df.sort_values("standardized_mean_difference")
        y = np.arange(len(plot_df))
        smd = plot_df.standardized_mean_difference.to_numpy()
        se = plot_df.standardized_mean_difference_se.to_numpy()
        yerr = np.vstack([np.maximum(1.96 * se, 0.0), np.maximum(1.96 * se, 0.0)])
        ax.errorbar(smd, y, xerr=yerr, fmt="o", color="#173f5f", ecolor="#7f8c8d",
                    capsize=2, markersize=4, lw=1)
        ax.axvline(0, color="#444444", lw=1)
        ax.axvline(-0.1, color="#c6c6c6", lw=1, ls="--")
        ax.axvline(0.1, color="#c6c6c6", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df.covariate, fontsize=8)
        ax.set_title(sample_name)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Standardized mean difference (HR - control), 95% CI")
    axes[1].set_xlabel("Standardized mean difference (HR - control), 95% CI")
    axes[2].set_xlabel("Standardized mean difference (HR - control), 95% CI")
    fig.suptitle("Nathan-Score Local Randomization Balance: +/-1 ft")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures" / "nathan_local_randomization_balance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Nathan-score local-randomization summary:")
    print(summary_df.to_string(index=False))
    print("\nSaved numeric/categorical balance tables, summary, and +/-1-ft figure to output/.")


if __name__ == "__main__":
    main()
