"""
04_local_randomization.py
-------------------------
Builds a local-randomization sample around the predicted fence-clearance
threshold and tests balance between home runs and non-home runs.

The sample is restricted to observations with

    abs(running_var) <= window

where running_var is recomputed using the same fence geometry and trajectory
approximation as 02_prepare_data.py. The treatment is the observed event
being ``home_run``. Controls are all other recorded batted-ball outcomes.

Outputs
-------
  local_randomization_sample.parquet
  local_randomization_balance.csv       numeric covariates
  local_randomization_categorical.csv   categorical covariates and levels
  local_randomization_summary.csv       sample and treatment summaries
  local_randomization_balance.png       standardized mean differences

Usage
-----
    python src/04_local_randomization.py
    python src/04_local_randomization.py --window 5 --steepness-factor 1.1
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"

CONTROL_EVENTS = {
    "field_out", "single", "double", "triple", "sac_fly",
    "sac_fly_double_play", "force_out", "grounded_into_double_play",
    "double_play", "fielders_choice", "fielders_choice_out",
    "field_error", "triple_play",
}

NUMERIC_COVARIATES = [
    "lineup_sequence_proxy", "at_bat_number", "pitch_number", "inning",
    "launch_speed", "launch_angle", "release_spin_rate", "outs_when_up",
    "runners_on",
    "home_score", "away_score", "bat_score", "fld_score", "score_diff",
    "bat_win_exp", "home_win_exp", "n_thruorder_pitcher", "release_speed",
    "effective_speed", "plate_x", "plate_z", "pfx_x", "pfx_z", "zone",
]

CATEGORICAL_COVARIATES = [
    "inning_topbot", "stand", "p_throws", "pitch_type", "pitch_name",
    "if_fielding_alignment", "of_fielding_alignment",
]


def load_prepare_module():
    """Load the existing running-variable implementation without duplicating it."""
    path = Path(__file__).with_name("02_prepare_data.py")
    spec = importlib.util.spec_from_file_location("prepare_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def construct_running_variable(input_path: Path, park_dims_path: Path,
                               steepness_factor: float) -> pd.DataFrame:
    """Load raw batted balls and recompute the current running variable."""
    prepare = load_prepare_module()
    df = pd.read_parquet(input_path)
    # Reuse an analysis-ready file when supplied. This avoids repeating the
    # expensive row-wise park lookup for sensitivity analyses.
    if "running_var" in df.columns and "fence_dist_ft" in df.columns:
        return prepare.compute_running_variable(
            df, steepness_factor=steepness_factor,
        )
    required = [
        "launch_angle", "hit_distance_sc", "hc_x", "hc_y", "home_team",
        "game_year", "events",
    ]
    df = df.dropna(subset=required).copy()
    df["spray_angle_deg"] = df.apply(
        lambda row: prepare.compute_spray_angle(row["hc_x"], row["hc_y"]),
        axis=1,
    )
    df = df[df["spray_angle_deg"].between(-50, 50)].copy()
    park_df = prepare.load_park_dimensions(park_dims_path)
    df = prepare.add_park_features(df, park_df)
    df = prepare.compute_running_variable(
        df, steepness_factor=steepness_factor,
    )
    return df


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-period indicators and a transparent at-bat sequence proxy."""
    df = df.copy()
    df["home_run"] = (df["events"] == "home_run").astype(int)
    df["runners_on"] = df[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1).astype(float)
    df["score_diff"] = pd.to_numeric(df["bat_score"], errors="coerce") - pd.to_numeric(
        df["fld_score"], errors="coerce"
    )
    # This is the sequence of the PA within the game half-inning, not a true
    # lineup slot; substitutions make lineup position unavailable here.
    df["lineup_sequence_proxy"] = (
        df.groupby(["game_pk", "inning", "inning_topbot"])["at_bat_number"]
        .rank(method="dense")
    )
    return df


def numeric_balance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    treated = df["home_run"] == 1
    control = ~treated
    for covariate in NUMERIC_COVARIATES:
        if covariate not in df.columns:
            continue
        values = pd.to_numeric(df[covariate], errors="coerce")
        t = values[treated].dropna().astype(float).to_numpy()
        c = values[control].dropna().astype(float).to_numpy()
        if len(t) < 2 or len(c) < 2:
            continue
        pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        smd = (t.mean() - c.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        difference_se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        smd_se = difference_se / pooled_sd if pooled_sd > 0 else np.nan
        test = stats.ttest_ind(t, c, equal_var=False, nan_policy="omit")
        rows.append({
            "covariate": covariate,
            "treated_n": len(t),
            "control_n": len(c),
            "treated_mean": t.mean(),
            "control_mean": c.mean(),
            "difference": t.mean() - c.mean(),
            "difference_se": difference_se,
            "standardized_mean_difference": smd,
            "standardized_mean_difference_se": smd_se,
            "smd_ci_lower": smd - 1.96 * smd_se if np.isfinite(smd_se) else np.nan,
            "smd_ci_upper": smd + 1.96 * smd_se if np.isfinite(smd_se) else np.nan,
            "welch_t": test.statistic,
            "welch_p": test.pvalue,
        })
    return pd.DataFrame(rows)


def categorical_balance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    treated = df["home_run"] == 1
    for covariate in CATEGORICAL_COVARIATES:
        if covariate not in df.columns:
            continue
        values = df[covariate].fillna("<missing>").astype(str)
        levels = sorted(values.unique())
        table = pd.crosstab(values, df["home_run"]).reindex(
            index=levels, columns=[0, 1], fill_value=0,
        )
        if table.shape[0] < 2:
            continue
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        for level in levels:
            t_rate = (values[treated] == level).mean()
            c_rate = (values[~treated] == level).mean()
            pooled = np.sqrt((t_rate * (1 - t_rate) + c_rate * (1 - c_rate)) / 2)
            rows.append({
                "covariate": covariate,
                "level": level,
                "treated_rate": t_rate,
                "control_rate": c_rate,
                "difference": t_rate - c_rate,
                "standardized_difference": (t_rate - c_rate) / pooled if pooled > 0 else np.nan,
                "omnibus_chi2": chi2,
                "omnibus_dof": dof,
                "omnibus_p": p_value,
            })
    return pd.DataFrame(rows)


def write_plot(balance: pd.DataFrame, path: Path) -> None:
    plot_df = balance.dropna(subset=["standardized_mean_difference"]).copy()
    plot_df = plot_df.sort_values("standardized_mean_difference")
    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.32)))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["standardized_mean_difference"], y,
        xerr=1.96 * plot_df["standardized_mean_difference_se"],
        fmt="o", color="#2c3e50", ecolor="#7f8c8d", elinewidth=1,
        capsize=3, markersize=5,
    )
    ax.axvline(0, color="#7f8c8d", lw=1)
    ax.axvline(-0.1, color="#bdc3c7", lw=1, ls="--")
    ax.axvline(0.1, color="#bdc3c7", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["covariate"])
    ax.set_xlabel("Standardized Mean Difference: Home Run minus Control (95% CI)")
    ax.set_title("Local Randomization Balance: Numeric Covariates")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test local randomization around the fence threshold.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "statcast_all.parquet")
    parser.add_argument("--park-dims", type=Path, default=ROOT / "data" / "park_dimensions_geom.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--steepness-factor", type=float, default=1.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading and constructing running variable from {args.input} ...")
    df = construct_running_variable(args.input, args.park_dims, args.steepness_factor)
    df = add_balance_features(df)
    valid_events = df["events"].isin(CONTROL_EVENTS | {"home_run"})
    df = df[valid_events & df["running_var"].notna()].copy()
    sample = df[df["running_var"].abs() <= args.window].copy()
    sample_path = args.output_dir / "local_randomization_sample.parquet"
    sample.to_parquet(sample_path, index=False)

    numeric = numeric_balance(sample)
    categorical = categorical_balance(sample)
    numeric.to_csv(args.output_dir / "local_randomization_balance.csv", index=False)
    categorical.to_csv(args.output_dir / "local_randomization_categorical.csv", index=False)
    write_plot(numeric, args.output_dir / "local_randomization_balance.png")

    summary = pd.DataFrame([
        {"metric": "input_rows_after_geometry", "value": len(df)},
        {"metric": "local_randomization_rows", "value": len(sample)},
        {"metric": "treated_home_runs", "value": int(sample["home_run"].sum())},
        {"metric": "control_non_home_runs", "value": int((sample["home_run"] == 0).sum())},
        {"metric": "window_ft", "value": args.window},
        {"metric": "steepness_factor", "value": args.steepness_factor},
        {"metric": "missing_running_variable_rows", "value": int(df["running_var"].isna().sum())},
    ])
    summary.to_csv(args.output_dir / "local_randomization_summary.csv", index=False)

    print("\nLOCAL RANDOMIZATION SUMMARY")
    print(summary.to_string(index=False))
    print("\nNumeric balance:")
    print(numeric[["covariate", "standardized_mean_difference", "welch_p"]].to_string(index=False))
    if not categorical.empty:
        print("\nCategorical omnibus balance:")
        print(categorical[["covariate", "omnibus_p"]].drop_duplicates().to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()