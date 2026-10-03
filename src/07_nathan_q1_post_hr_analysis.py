"""Explore observed outcomes in the Q1 launch-angle, +/-1-ft Nathan sample.

The available parquet contains batted-ball plate-appearance rows, not subsequent
pitches. Consequently only same-play change metrics (delta_run_exp and
 delta_home_win_exp) can be regressed here; they are not downstream response
outcomes. The script reports that limitation and exports descriptive statistics
for every column in the selected sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "rdd_data_nathan.parquet"
OUTPUT = ROOT / "output"

CONTROL_EVENTS = {
    "field_out", "single", "double", "triple", "sac_fly",
    "sac_fly_double_play", "force_out", "grounded_into_double_play",
    "double_play", "fielders_choice", "fielders_choice_out",
    "field_error", "triple_play",
}

SAME_PLAY_OUTCOMES = ["delta_run_exp", "delta_home_win_exp"]
IDENTIFIER_COLUMNS = {
    "game_pk", "batter", "pitcher", "on_1b", "on_2b", "on_3b",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6",
    "fielder_7", "fielder_8", "fielder_9", "hit_location", "trajectory_status",
}


def build_sample(df: pd.DataFrame, window: float) -> tuple[pd.DataFrame, float]:
    valid_events = df["events"].isin(CONTROL_EVENTS | {"home_run"})
    df = df[valid_events & df["running_var_nathan"].notna()].copy()
    df["home_run"] = df["events"].eq("home_run").astype(int)
    hr_angles = pd.to_numeric(
        df.loc[df.home_run.eq(1), "launch_angle"], errors="coerce",
    ).dropna()
    q1_cutoff = float(hr_angles.quantile(0.25))
    sample = df[
        df["running_var_nathan"].abs().le(window)
        & pd.to_numeric(df["launch_angle"], errors="coerce").le(q1_cutoff)
    ].copy()
    return sample, q1_cutoff


def descriptive_rows(sample: pd.DataFrame) -> pd.DataFrame:
    """Tidy summaries for every field and each treatment group."""
    rows = []
    groups = [("all", sample), ("home_run", sample[sample.home_run.eq(1)]),
              ("control", sample[sample.home_run.eq(0)])]
    for column in sample.columns:
        series = sample[column]
        numeric = (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
            and column not in IDENTIFIER_COLUMNS
        )
        for group_name, group in groups:
            values = group[column]
            nonmissing = values.dropna()
            row = {
                "variable": column,
                "group": group_name,
                "dtype": str(series.dtype),
                "n": int(values.notna().sum()),
                "missing_n": int(values.isna().sum()),
                "missing_pct": float(values.isna().mean()) if len(values) else np.nan,
                "n_unique": int(nonmissing.nunique()),
            }
            if numeric:
                nums = pd.to_numeric(nonmissing, errors="coerce").dropna()
                for stat, val in {
                    "mean": nums.mean(), "std": nums.std(), "min": nums.min(),
                    "q25": nums.quantile(0.25), "median": nums.median(),
                    "q75": nums.quantile(0.75), "max": nums.max(),
                }.items():
                    row[stat] = val
                row["top_values"] = ""
            else:
                counts = nonmissing.astype(str).value_counts().head(10)
                row["mean"] = np.nan
                row["std"] = np.nan
                row["min"] = np.nan
                row["q25"] = np.nan
                row["median"] = np.nan
                row["q75"] = np.nan
                row["max"] = np.nan
                row["top_values"] = "; ".join(f"{key}: {value}" for key, value in counts.items())
            rows.append(row)
    return pd.DataFrame(rows)


def regression_rows(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in SAME_PLAY_OUTCOMES:
        if outcome not in sample:
            continue
        subset = sample[["home_run", outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        treated_n = int(subset.home_run.sum())
        control_n = int((subset.home_run == 0).sum())
        if treated_n < 2 or control_n < 2:
            continue
        design = sm.add_constant(subset["home_run"].astype(float))
        fit = sm.OLS(subset[outcome].astype(float), design).fit(cov_type="HC3")
        effect = float(fit.params["home_run"])
        se = float(fit.bse["home_run"])
        rows.append({
            "outcome": outcome,
            "outcome_scope": "same plate appearance; not post-HR response",
            "treated_n": treated_n,
            "control_n": control_n,
            "treated_mean": float(subset.loc[subset.home_run.eq(1), outcome].mean()),
            "control_mean": float(subset.loc[subset.home_run.eq(0), outcome].mean()),
            "hr_coefficient": effect,
            "hc3_robust_se": se,
            "ci_95_lower": effect - 1.96 * se,
            "ci_95_upper": effect + 1.96 * se,
            "p_value": float(fit.pvalues["home_run"]),
            "r_squared": float(fit.rsquared),
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if df.empty:
        return "No estimable rows."
    view = df[columns].copy()
    for col in view.columns:
        converted = pd.to_numeric(view[col], errors="coerce")
        if converted.notna().sum() != view[col].notna().sum():
            continue
        if col == "p_value":
            view[col] = converted.map(lambda x: "" if pd.isna(x) else (f"{x:.2e}" if x < 0.001 else f"{x:.3f}"))
        else:
            view[col] = converted.map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(sample: pd.DataFrame, q1_cutoff: float, window: float,
                 regressions: pd.DataFrame, descriptives: pd.DataFrame,
                 report_path: Path) -> None:
    treated = sample[sample.home_run.eq(1)]
    control = sample[sample.home_run.eq(0)]
    launch_speed_row = descriptives[
        descriptives.variable.eq("launch_speed") & descriptives.group.eq("all")
    ]
    la_row = descriptives[
        descriptives.variable.eq("launch_angle") & descriptives.group.eq("all")
    ]
    launch_hr = descriptives[
        descriptives.variable.eq("launch_speed") & descriptives.group.eq("home_run")
    ].iloc[0]
    launch_control = descriptives[
        descriptives.variable.eq("launch_speed") & descriptives.group.eq("control")
    ].iloc[0]
    pooled_sd = np.sqrt((launch_hr["std"] ** 2 + launch_control["std"] ** 2) / 2)
    launch_speed_smd = (launch_hr["mean"] - launch_control["mean"]) / pooled_sd

    all_stats = descriptives.set_index(["variable", "group"])
    full_rows = []
    for variable in sample.columns:
        overall = all_stats.loc[(variable, "all")]
        treated_stats = all_stats.loc[(variable, "home_run")]
        control_stats = all_stats.loc[(variable, "control")]
        is_numeric = (
            pd.api.types.is_numeric_dtype(sample[variable].dtype)
            and not pd.api.types.is_bool_dtype(sample[variable].dtype)
            and variable not in IDENTIFIER_COLUMNS
        )
        if is_numeric:
            full_rows.append({
                "variable": variable,
                "type": "numeric",
                "missing (all)": overall["missing_n"],
                "unique (all)": overall["n_unique"],
                "all mean": overall["mean"], "all SD": overall["std"],
                "all min": overall["min"], "all Q1": overall["q25"],
                "all median": overall["median"], "all Q3": overall["q75"], "all max": overall["max"],
                "HR mean": treated_stats["mean"], "HR SD": treated_stats["std"],
                "control mean": control_stats["mean"], "control SD": control_stats["std"],
                "most common levels by group": "",
            })
        else:
            def brief_levels(value: str) -> str:
                return "; ".join(value.split("; ")[:5])
            common_levels = (
                f"all: {brief_levels(overall['top_values'])}; "
                f"HR: {brief_levels(treated_stats['top_values'])}; "
                f"control: {brief_levels(control_stats['top_values'])}"
            )
            full_rows.append({
                "variable": variable,
                "type": "categorical/id",
                "missing (all)": overall["missing_n"],
                "unique (all)": overall["n_unique"],
                "all mean": np.nan, "all SD": np.nan, "all min": np.nan,
                "all Q1": np.nan, "all median": np.nan, "all Q3": np.nan,
                "all max": np.nan, "HR mean": np.nan, "HR SD": np.nan,
                "control mean": np.nan, "control SD": np.nan,
                "most common levels by group": common_levels,
            })
    full_stats_table = pd.DataFrame(full_rows)
    text = [
        "# Exploratory HR Outcome Regressions",
        "",
        "## Sample",
        "",
        f"The sample is the Nathan-score window |score| <= {window:g} ft, restricted to launch angles at or below the first quartile of observed HR launch angle ({q1_cutoff:g} degrees). It contains {len(sample):,} batted-ball plate appearances: {len(treated):,} HRs and {len(control):,} controls (HR share {sample.home_run.mean():.1%}).",
        "",
        "This is an exploratory unadjusted comparison: each regression is `outcome ~ home_run` with HC3 heteroskedasticity-robust standard errors. Robust SEs address heteroskedasticity, not confounding or misspecification.",
        "",
        "## Outcome Availability",
        "",
        "The stored data are one row per batted-ball plate appearance. They do not contain subsequent pitches, pitcher substitutions after the event, next-half-inning outcomes, next at-bat outcomes, post-event scores, or a complete pitch-by-pitch sequence. Therefore, the only available numeric change outcomes are `delta_run_exp` and `delta_home_win_exp`; both measure the same play/plate appearance that defines the treatment, not the response after it. Their regressions below are descriptive/mechanical and do not estimate downstream effects of a home run.",
        "",
        "The requested next-pitch, rest-of-inning, next-half-inning, and next-PA effects require a complete pitch-level game stream joined to the treated batted-ball rows. The current downloader filters to fly-ball batted events, so those follow-up observations are absent.",
        "",
        "## Regression Results",
        "",
        markdown_table(regressions, ["outcome", "treated_n", "control_n", "treated_mean", "control_mean", "hr_coefficient", "hc3_robust_se", "ci_95_lower", "ci_95_upper", "p_value", "r_squared"]),
        "",
        "The coefficient is the HR-minus-control mean difference in the same-play outcome. Interpret neither coefficient as an effect on subsequent pitching or scoring.",
        "",
        "## Sample Summary",
        "",
        "| Variable | HR mean | HR SD | Control mean | Control SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for variable in ["launch_speed", "launch_angle", "running_var_nathan", "z_at_fence_ft", "model_landing_distance_ft", "delta_run_exp", "delta_home_win_exp"]:
        block = descriptives[descriptives.variable.eq(variable)].set_index("group")
        if "home_run" not in block.index or "control" not in block.index:
            continue
        def cell(group: str, stat: str) -> str:
            val = block.loc[group, stat]
            return "NA" if pd.isna(val) else f"{val:.3f}"
        text.append(f"| {variable} | {cell('home_run','mean')} | {cell('home_run','std')} | {cell('control','mean')} | {cell('control','std')} |")
    text.extend([
        "",
        f"Launch speed remains substantially imbalanced: HR mean {launch_hr['mean']:.2f} mph versus control mean {launch_control['mean']:.2f} mph; SMD = {launch_speed_smd:.2f}. This is not ‘pretty good’ balance and the unadjusted regressions should not be interpreted causally.",
        "",
        "## Full Descriptive Statistics",
        "",
        f"The table below summarizes all {len(sample.columns)} columns in the selected sample. Numeric rows show overall and treatment/control means, SDs, quartiles, and range; categorical and identifier rows show missingness, unique counts, and the ten most frequent values in each group. A machine-readable version is in [nathan_q1_pm1_descriptive_stats.csv](nathan_q1_pm1_descriptive_stats.csv).",
        "",
        markdown_table(full_stats_table, list(full_stats_table.columns), digits=3),
        "",
        "## Reproducibility",
        "",
        "Run `python src/07_nathan_q1_post_hr_analysis.py` to regenerate this report and the CSV outputs.",
        "",
    ])
    report_path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory HR outcome regressions in Q1-angle +/-1-ft Nathan sample.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--window", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    sample, q1_cutoff = build_sample(df, args.window)
    regressions = regression_rows(sample)
    descriptives = descriptive_rows(sample)
    regressions.to_csv(args.output_dir / "nathan_q1_pm1_regressions.csv", index=False)
    descriptives.to_csv(args.output_dir / "nathan_q1_pm1_descriptive_stats.csv", index=False)
    sample.to_parquet(args.output_dir / "nathan_q1_pm1_analysis_sample.parquet", index=False)
    report_path = args.output_dir / "nathan_q1_pm1_post_hr_analysis.md"
    write_report(sample, q1_cutoff, args.window, regressions, descriptives, report_path)

    print(f"HR launch-angle Q1 cutoff: {q1_cutoff:g} degrees")
    print(f"Sample: {len(sample):,}; HRs={int(sample.home_run.sum()):,}; controls={int(sample.home_run.eq(0).sum()):,}")
    print("\nAvailable same-play robust regressions:")
    print(regressions.to_string(index=False))
    print("\nDescriptive statistics for all sample variables:")
    print(descriptives.to_string(index=False))
    print(f"\nFull descriptive stats for {len(sample.columns)} variables saved to {args.output_dir / 'nathan_q1_pm1_descriptive_stats.csv'}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
