"""
03_first_stage.py
-----------------
Estimates the first stage of the Fuzzy RDD:
    "Does the running variable predict home run treatment?"

The running variable is  running_var = z_at_fence - fence_height_ft.
The cutoff is zero.  The first stage establishes that there is a sharp
jump in P(home_run) at running_var = 0.

Steps:
  1.  Load processed data (output of 02_prepare_data.py)
  2.  McCrary (2008) density test — checks for manipulation/bunching at cutoff
  3.  Local polynomial RDD estimate of the first stage (rdrobust)
  4.  Binned scatter plot with RD fit lines
  5.  Histogram of running variable with HR rate overlay
  6.  Covariate continuity ("balance") checks at the cutoff
  7.  Save results tables and figures to output/

Usage:
    python src/03_first_stage.py [--input data/processed/rdd_data.parquet]
                                  [--bandwidth 20]   # optional manual override
                                  [--output-dir output]

Dependencies:
    rdrobust (pip install rdrobust)
    rddensity (pip install rddensity)
    matplotlib, seaborn, pandas, numpy, statsmodels
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec

# rdrobust / rddensity — pip install rdrobust rddensity
try:
    from rdrobust import rdrobust, rdbwselect
    from rddensity import rddensity
    HAS_RDROBUST = True
except ImportError:
    HAS_RDROBUST = False
    print(
        "WARNING: rdrobust / rddensity not installed.\n"
        "  Install with:  pip install rdrobust rddensity\n"
        "  Falling back to manual local-linear regression."
    )

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

CUTOFF_COLOR = "#e74c3c"
LEFT_COLOR = "#2980b9"
RIGHT_COLOR = "#27ae60"


# ── Utility: manual local-linear RDD ─────────────────────────────────────────

def local_linear_rdd(
    running_var: np.ndarray,
    outcome: np.ndarray,
    cutoff: float = 0.0,
    bandwidth: float = None,
    kernel: str = "triangular",
) -> dict:
    """
    Simple local-linear RDD estimator using WLS.
    Returns a dict with keys: tau, se, t, p, bw, n_left, n_right.
    Uses explicit weighted normal equations to avoid the O(n²) np.diag issue.
    """
    rv = np.asarray(running_var)
    y = np.asarray(outcome, dtype=float)

    if bandwidth is None:
        bandwidth = 20.0

    centered = rv - cutoff
    mask = np.abs(centered) <= bandwidth
    rv_m = centered[mask]
    y_m = y[mask]

    if len(rv_m) < 10:
        return {"error": "Too few observations in bandwidth"}

    # Kernel weights (1D array, not diagonal matrix)
    if kernel == "triangular":
        w = 1.0 - np.abs(rv_m) / bandwidth
    else:
        w = np.ones(len(rv_m))

    d = (rv_m >= 0).astype(float)

    X = np.column_stack([
        np.ones(len(rv_m)),
        d,
        rv_m,
        d * rv_m,
    ])

    # Weighted normal equations: (X'WX)^{-1} X'Wy  — O(n * k²), not O(n²)
    Xw = X * w[:, None]          # each row of X scaled by its weight
    XtWX = Xw.T @ X              # k×k
    XtWy = Xw.T @ y_m            # k×1

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix"}

    # HC1 sandwich SE
    e = y_m - X @ beta
    # Score: X_i * w_i * e_i  (k×1 per obs)
    score = X * (w * e)[:, None]   # n×k
    meat = score.T @ score          # k×k
    XtWX_inv = np.linalg.inv(XtWX)
    vcov = XtWX_inv @ meat @ XtWX_inv
    se = np.sqrt(np.diag(vcov))

    tau = beta[1]
    se_tau = se[1]
    t = tau / se_tau if se_tau > 0 else 0.0

    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(t)))

    n_left = (rv_m < 0).sum()
    n_right = (rv_m >= 0).sum()

    return {
        "tau": tau,
        "se": se_tau,
        "t": t,
        "p": p,
        "bw": bandwidth,
        "n_left": int(n_left),
        "n_right": int(n_right),
        "n_total": int(n_left + n_right),
        "beta_left_slope": beta[2],
        "beta_right_slope": beta[2] + beta[3],
        "intercept_left": beta[0],
        "intercept_right": beta[0] + beta[1],
    }


# ── Binned scatter ────────────────────────────────────────────────────────────

def binned_scatter(
    rv: pd.Series,
    outcome: pd.Series,
    n_bins: int = 40,
    cutoff: float = 0.0,
    window: float = 30.0,
) -> pd.DataFrame:
    """
    Compute mean outcome in equal-width bins of the running variable,
    within [-window, +window] of the cutoff.
    Returns DataFrame with columns: bin_mid, mean_outcome, n.
    """
    mask = rv.between(cutoff - window, cutoff + window)
    rv_w = rv[mask]
    out_w = outcome[mask]

    # Separate bins on each side
    left_bins = np.linspace(cutoff - window, cutoff, n_bins // 2 + 1)
    right_bins = np.linspace(cutoff, cutoff + window, n_bins // 2 + 1)
    bins = np.unique(np.concatenate([left_bins, right_bins]))

    labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    rv_binned = pd.cut(rv_w, bins=bins, labels=labels, include_lowest=True)

    result = (
        pd.DataFrame({"rv_bin": rv_binned, "outcome": out_w.values})
        .groupby("rv_bin", observed=True)
        .agg(mean_outcome=("outcome", "mean"), n=("outcome", "count"))
        .reset_index()
        .rename(columns={"rv_bin": "bin_mid"})
    )
    result["bin_mid"] = result["bin_mid"].astype(float)
    return result.dropna(subset=["mean_outcome"])


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_first_stage(
    df: pd.DataFrame,
    result: dict,
    bandwidth: float,
    save_path: Path,
) -> None:
    """
    Two-panel figure:
      Left:  Binned scatter of HR rate vs. running_var with local-linear fits
      Right: Histogram of running variable colored by HR status
    """
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    ax_rd = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    window = min(bandwidth * 1.5, 40.0)
    mask = df["running_var"].between(-window, window)
    rv = df.loc[mask, "running_var"]
    hr = df.loc[mask, "home_run"]

    # ── Panel 1: RD scatter ────────────────────────────────────────────────
    bins_df = binned_scatter(rv, hr, n_bins=40, cutoff=0.0, window=window)
    left = bins_df[bins_df["bin_mid"] < 0]
    right = bins_df[bins_df["bin_mid"] >= 0]

    ax_rd.scatter(
        left["bin_mid"], left["mean_outcome"],
        color=LEFT_COLOR, s=left["n"] / left["n"].max() * 80 + 10,
        alpha=0.8, zorder=3, label="Non-HR side (≤ 0)",
    )
    ax_rd.scatter(
        right["bin_mid"], right["mean_outcome"],
        color=RIGHT_COLOR, s=right["n"] / right["n"].max() * 80 + 10,
        alpha=0.8, zorder=3, label="HR side (> 0)",
    )

    # Fit lines
    rv_arr = rv.values
    hr_arr = hr.values
    for side, color, x_range in [
        ("left", LEFT_COLOR, np.linspace(-window, 0, 200)),
        ("right", RIGHT_COLOR, np.linspace(0, window, 200)),
    ]:
        side_mask = (rv_arr < 0) if side == "left" else (rv_arr >= 0)
        side_mask &= np.abs(rv_arr) <= bandwidth
        if side_mask.sum() > 10:
            rv_s = rv_arr[side_mask]
            hr_s = hr_arr[side_mask]
            wts = 1.0 - np.abs(rv_s) / bandwidth  # triangular kernel
            try:
                coeffs = np.polyfit(rv_s, hr_s, 1, w=wts)
                x_fit = x_range if side == "right" else x_range
                ax_rd.plot(x_fit, np.polyval(coeffs, x_fit), color=color, lw=2, zorder=4)
            except Exception:
                pass

    ax_rd.axvline(0, color=CUTOFF_COLOR, lw=1.5, ls="--", label="Cutoff (0 ft)", zorder=5)

    # Annotate jump
    if "tau" in result:
        tau_pct = result["tau"] * 100
        se_pct = result["se"] * 100
        bw_used = result["bw"]
        ax_rd.annotate(
            f"First-stage jump:\n"
            f"Δ = {tau_pct:.1f} pp\n"
            f"SE = {se_pct:.1f} pp\n"
            f"BW = ±{bw_used:.1f} ft",
            xy=(0.03, 0.65), xycoords="axes fraction",
            fontsize=9, color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdc3c7", alpha=0.9),
        )

    ax_rd.set_xlabel("Running Variable: z_at_fence − fence_height (ft)", fontsize=10)
    ax_rd.set_ylabel("P(Home Run)", fontsize=10)
    ax_rd.set_title("First Stage: Jump in Home Run Probability at Fence Threshold", fontsize=11)
    ax_rd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_rd.legend(fontsize=8, framealpha=0.9)

    # ── Panel 2: Histogram ─────────────────────────────────────────────────
    hr_mask = hr.astype(bool)
    nonhr_mask = ~hr_mask

    bins_hist = np.linspace(-window, window, 61)
    ax_hist.hist(
        rv[nonhr_mask], bins=bins_hist, color=LEFT_COLOR, alpha=0.6,
        label="Non-HR", density=False,
    )
    ax_hist.hist(
        rv[hr_mask], bins=bins_hist, color=RIGHT_COLOR, alpha=0.6,
        label="Home Run", density=False,
    )
    ax_hist.axvline(0, color=CUTOFF_COLOR, lw=1.5, ls="--", label="Cutoff")
    ax_hist.set_xlabel("Running Variable (ft)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("Distribution of Running Variable by Outcome", fontsize=11)
    ax_hist.legend(fontsize=9)

    plt.suptitle(
        "HR Fuzzy RDD — First Stage\n"
        f"Statcast Fly Balls 2015–2024  |  N = {len(df):,}",
        fontsize=12, y=1.01,
    )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


def plot_density_test(
    rv: pd.Series,
    save_path: Path,
    density_result=None,
) -> None:
    """
    McCrary-style density plot: histogram of the running variable.
    Annotates with the rddensity test statistic if available.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    bins = np.linspace(rv.min(), rv.max(), 80)
    ax.hist(rv, bins=bins, color="#7f8c8d", alpha=0.6, density=True, label="Observed density")
    ax.axvline(0, color=CUTOFF_COLOR, lw=2, ls="--", label="Cutoff")

    if density_result is not None:
        try:
            t = float(density_result.test["t_jk"])
            p = float(density_result.test["p_jk"])
            annotation_text = f"McCrary density test\nt = {t:.3f},  p = {p:.3f}"
            if p > 0.05:
                annotation_text += "\n→ No bunching detected"
            ax.annotate(
                annotation_text,
                xy=(0.65, 0.82), xycoords="axes fraction",
                fontsize=9, color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdc3c7", alpha=0.9),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    ax.set_xlabel("Running Variable (ft)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("McCrary (2008) Density Test: Distribution of Running Variable at Cutoff", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


def plot_covariate_balance(
    df: pd.DataFrame,
    covariates: List[str],
    bandwidth: float,
    save_path: Path,
) -> None:
    """
    For each covariate, run a local-linear RDD and plot the estimated
    jump with 95% CI. No covariate should have a significant jump.
    """
    results = []
    for cov in covariates:
        sub = df[["running_var", cov]].dropna()
        if len(sub) < 50 or sub[cov].nunique() < 2:
            continue
        res = local_linear_rdd(
            sub["running_var"].values,
            sub[cov].values,
            bandwidth=bandwidth,
        )
        if "error" not in res:
            results.append({
                "covariate": cov,
                "tau": res["tau"],
                "se": res["se"],
                "p": res["p"],
            })

    if not results:
        print("  No valid covariate balance estimates.")
        return

    res_df = pd.DataFrame(results).sort_values("tau")

    fig, ax = plt.subplots(figsize=(9, max(4, len(res_df) * 0.45)))
    y_pos = np.arange(len(res_df))

    colors = [
        "#e74c3c" if p < 0.05 else "#2c3e50"
        for p in res_df["p"]
    ]
    ax.barh(y_pos, res_df["tau"], xerr=1.96 * res_df["se"],
            align="center", color=colors, alpha=0.7, capsize=4)
    ax.axvline(0, color="#7f8c8d", lw=1.5, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df["covariate"], fontsize=9)
    ax.set_xlabel("Estimated Jump at Cutoff (with 95% CI)", fontsize=10)
    ax.set_title(
        "Covariate Continuity at Cutoff\n"
        "(Red = significant at 5% — should be rare under valid RD)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")
    return res_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Estimate first-stage Fuzzy RDD.")
    parser.add_argument(
        "--input", type=Path,
        default=PROCESSED_DIR / "rdd_data.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--bandwidth", type=float, default=None,
        help="Manual bandwidth (ft). If omitted, uses rdrobust CCT or default 20 ft.",
    )
    parser.add_argument(
        "--max-rv", type=float, default=50.0,
        help="Cap |running_var| at this value to exclude extreme outliers.",
    )
    args = parser.parse_args()

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows loaded")

    # Drop rows without running variable
    df = df.dropna(subset=["running_var", "home_run"]).copy()
    # Cast to plain Python dtypes to avoid nullable Float64/Int64 indexing issues
    df["running_var"] = df["running_var"].astype(float)
    df["home_run"]    = df["home_run"].astype(int)
    print(f"  {len(df):,} rows with valid running_var")

    # Cap extreme outliers (very long HRs or balls hit straight at center field
    # with no good fence data)
    df = df[df["running_var"].abs() <= args.max_rv].copy()
    print(f"  {len(df):,} rows after |running_var| ≤ {args.max_rv} ft cap")

    rv = df["running_var"].astype(float).values
    hr = df["home_run"].astype(float).values

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SAMPLE SUMMARY")
    print("="*60)
    print(f"  N total:           {len(df):>8,}")
    print(f"  N home runs:       {int(hr.sum()):>8,}  ({100*hr.mean():.1f}%)")
    print(f"  N non-HR fly balls:{int((1-hr).sum()):>8,}")
    print(f"\n  Running variable distribution:")
    print(f"    Min:    {rv.min():.1f} ft")
    print(f"    P10:    {np.percentile(rv, 10):.1f} ft")
    print(f"    Median: {np.median(rv):.1f} ft")
    print(f"    P90:    {np.percentile(rv, 90):.1f} ft")
    print(f"    Max:    {rv.max():.1f} ft")
    print(f"\n  HR rate by running_var sign:")
    print(f"    running_var > 0: {100*hr[rv > 0].mean():.1f}%  (should be ~100%)")
    print(f"    running_var ≤ 0: {100*hr[rv <= 0].mean():.1f}%  (should be ~0%)")

    # ── McCrary density test ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("McCRARY DENSITY TEST")
    print("="*60)
    density_result = None
    if HAS_RDROBUST:
        try:
            density_result = rddensity(rv, c=0.0)
            print(density_result)
            t_stat = density_result.test["t_jk"]
            p_val = density_result.test["p_jk"]
            print(f"\n  T-statistic: {t_stat:.3f}")
            print(f"  P-value:     {p_val:.3f}")
            if p_val > 0.05:
                print("  → No evidence of manipulation (expected for batted balls).")
            else:
                print("  → Significant density discontinuity — investigate.")
        except Exception as e:
            print(f"  rddensity failed: {e}")
    else:
        print("  (rddensity not available — skipping formal test)")

    plot_density_test(
        pd.Series(rv),
        figures_dir / "density_test.png",
        density_result=density_result,
    )

    # ── Bandwidth selection and first-stage estimate ──────────────────────────
    print("\n" + "="*60)
    print("FIRST STAGE: P(HOME RUN) JUMP AT CUTOFF")
    print("="*60)

    bw = args.bandwidth
    rdrobust_result = None

    if HAS_RDROBUST:
        try:
            print("\n  Running rdrobust (CCT optimal bandwidth) ...")
            rdrobust_result = rdrobust(y=hr, x=rv, c=0.0, kernel="triangular",
                                       bwselect="mserd")
            print(rdrobust_result)

            # rdrobust stores results in DataFrames; extract safely
            coef_df = rdrobust_result.coef
            se_df   = rdrobust_result.se
            ci_df   = rdrobust_result.ci
            pv_df   = rdrobust_result.pv
            bw_df   = rdrobust_result.bws
            N_vec   = rdrobust_result.N

            # coef/se/pv are single-column DataFrames indexed by method name
            coef_val = float(coef_df.iloc[0, 0])
            se_conv  = float(se_df.iloc[0, 0])
            se_rb    = float(se_df.iloc[2, 0])   # robust SE (row 2)
            ci_l     = float(ci_df.iloc[2, 0])
            ci_u     = float(ci_df.iloc[2, 1])
            pval     = float(pv_df.iloc[2, 0])
            bw_l     = float(bw_df.iloc[0, 0])
            bw_r     = float(bw_df.iloc[0, 1])

            print(f"\n  CCT Optimal bandwidth (left):  {bw_l:.2f} ft")
            print(f"  CCT Optimal bandwidth (right): {bw_r:.2f} ft")
            print(f"\n  FIRST STAGE ESTIMATE (rdrobust CCT):")
            print(f"  Jump in P(HR) at cutoff:  {coef_val*100:.2f} pp")
            print(f"  Robust SE:                {se_rb*100:.2f} pp")
            print(f"  95% CI:                   [{ci_l*100:.2f}, {ci_u*100:.2f}] pp")
            print(f"  Robust p-value:           {pval:.4f}")

            if bw is None:
                bw = bw_l

        except Exception as e:
            import traceback
            print(f"  rdrobust failed: {e}")
            traceback.print_exc()
            print("  Falling back to manual local-linear estimator.")
            rdrobust_result = None

    # Manual local-linear (always run as cross-check / fallback)
    if bw is None:
        bw = 20.0
        print(f"\n  Using default bandwidth: ±{bw} ft")

    manual_result = local_linear_rdd(rv, hr, bandwidth=bw)
    print(f"\n  MANUAL LOCAL-LINEAR (BW = ±{bw:.1f} ft, triangular kernel):")
    if "error" in manual_result:
        print(f"  Error: {manual_result['error']}")
    else:
        print(f"  N (left):   {manual_result['n_left']:,}")
        print(f"  N (right):  {manual_result['n_right']:,}")
        print(f"  Jump (τ):   {manual_result['tau']*100:.2f} pp")
        print(f"  SE:         {manual_result['se']*100:.2f} pp")
        print(f"  t-stat:     {manual_result['t']:.3f}")
        print(f"  p-value:    {manual_result['p']:.4f}")

    # ── Plot first stage ────────────────────────────────────────────────────
    result_for_plot = manual_result.copy()
    if rdrobust_result is not None:
        try:
            result_for_plot = {
                "tau": coef_val,
                "se": se_rb,
                "bw": bw,
            }
        except Exception:
            pass
    plot_first_stage(
        df, result_for_plot, bandwidth=bw,
        save_path=figures_dir / "first_stage_rdd.png",
    )

    # ── Multiple bandwidths sensitivity ─────────────────────────────────────
    print("\n" + "="*60)
    print("BANDWIDTH SENSITIVITY")
    print("="*60)
    bw_grid = [5, 8, 10, 12, 15, 20, 25, 30]
    sensitivity_rows = []
    for bw_i in bw_grid:
        res = local_linear_rdd(rv, hr, bandwidth=bw_i)
        if "error" not in res:
            sensitivity_rows.append({
                "bandwidth_ft": bw_i,
                "n_total": res["n_total"],
                "n_left": res["n_left"],
                "n_right": res["n_right"],
                "tau_pp": res["tau"] * 100,
                "se_pp": res["se"] * 100,
                "t_stat": res["t"],
                "p_value": res["p"],
            })
            print(
                f"  BW={bw_i:>3} ft | N={res['n_total']:>6,} | "
                f"τ={res['tau']*100:>6.2f} pp | "
                f"SE={res['se']*100:>5.2f} | "
                f"p={res['p']:.4f}"
            )

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_path = tables_dir / "first_stage_bandwidth_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"\n  Sensitivity table saved → {sens_path}")

    # Bandwidth sensitivity plot
    if len(sensitivity_rows) > 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        bws = sens_df["bandwidth_ft"]
        taus = sens_df["tau_pp"]
        ses = sens_df["se_pp"]
        ax.plot(bws, taus, "o-", color="#2c3e50", lw=2, ms=6, label="Point estimate")
        ax.fill_between(bws, taus - 1.96 * ses, taus + 1.96 * ses,
                        alpha=0.2, color="#2c3e50", label="95% CI")
        ax.axhline(0, color="#7f8c8d", lw=1, ls="--")
        ax.set_xlabel("Bandwidth (ft)", fontsize=10)
        ax.set_ylabel("First-Stage Jump (percentage points)", fontsize=10)
        ax.set_title("Bandwidth Sensitivity: First-Stage Estimate", fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / "first_stage_bw_sensitivity.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved → {figures_dir / 'first_stage_bw_sensitivity.png'}")

    # ── Covariate balance tests ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("COVARIATE CONTINUITY TESTS")
    print("="*60)

    # Predetermined situational variables that should NOT jump at the cutoff.
    # (launch_speed and launch_angle are excluded — they correlate with rv by construction.)
    balance_covariates = [
        "inning", "outs_when_up",
        "home_score", "away_score",
        "bat_win_exp",
    ]
    # Add binary runner indicators where available
    for col in ["on_1b", "on_2b", "on_3b"]:
        if col in df.columns:
            df[f"runner_{col[-2:]}"] = df[col].notna().astype(float)
            balance_covariates.append(f"runner_{col[-2:]}")

    # Only include columns that actually exist in the data
    balance_covariates = [c for c in balance_covariates if c in df.columns]
    print(f"  Testing {len(balance_covariates)} covariates: {balance_covariates}")

    if balance_covariates:
        balance_df = plot_covariate_balance(
            df, balance_covariates, bandwidth=bw,
            save_path=figures_dir / "covariate_balance.png",
        )
        if balance_df is not None:
            balance_path = tables_dir / "covariate_balance.csv"
            balance_df.to_csv(balance_path, index=False)
            print(f"  Balance table saved → {balance_path}")
            print(f"\n  Significant jumps (p<0.05): "
                  f"{(balance_df['p'] < 0.05).sum()} / {len(balance_df)}")

    # ── Save main first-stage results table ──────────────────────────────────
    if rdrobust_result is not None:
        try:
            main_result_row = {
                "method": "rdrobust (CCT)",
                "bandwidth_left_ft": bw_l,
                "bandwidth_right_ft": bw_r,
                "tau_pp": coef_val * 100,
                "se_conventional_pp": se_conv * 100,
                "se_robust_pp": se_rb * 100,
                "ci_lower_pp": ci_l * 100,
                "ci_upper_pp": ci_u * 100,
                "p_value_robust": pval,
                "n_left": int(N_vec[0]),
                "n_right": int(N_vec[1]),
            }
        except Exception:
            main_result_row = {**manual_result, "method": "local-linear (manual)"}
    else:
        main_result_row = {**manual_result, "method": "local-linear (manual)"}

    result_df = pd.DataFrame([main_result_row])
    result_path = tables_dir / "first_stage_main.csv"
    result_df.to_csv(result_path, index=False)
    print(f"\n  Main result table saved → {result_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FIRST STAGE COMPLETE")
    print("="*60)
    print(f"  Output figures:  {figures_dir}")
    print(f"  Output tables:   {tables_dir}")
    print(
        "\n  Interpretation:\n"
        "  The first stage establishes that the running variable\n"
        "  (z_at_fence - fence_height) predicts home run treatment\n"
        "  with a large jump at the zero threshold. This is the\n"
        "  necessary condition for a valid Fuzzy RDD: the instrument\n"
        "  (running_var > 0) must strongly shift the probability of\n"
        "  receiving the treatment (home_run = 1).\n"
        "\n"
        "  A strong first stage (large τ, small SE, small p-value)\n"
        "  and a stable estimate across bandwidths supports the\n"
        "  validity of the design."
    )
    print("="*60)


if __name__ == "__main__":
    main()
