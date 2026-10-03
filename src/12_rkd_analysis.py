"""12_rkd_analysis.py
--------------------
Regression Probability Kink Design (RPKD) analysis following Dong (2018),
"Jump or Kink? Identifying Education Effects by Regression Discontinuity
Design Without the Discontinuity."

Setting
-------
Running variable X = coord_fence_dist_ft:
  Negative = hit coordinate inside the fence (no HR)
  Positive = hit coordinate beyond the fence (HR territory)
  At X = 0: no jump in P(HR), but a sharp kink (slope change). This is the
  key feature exploited by the RPKD: treatment (home_run) is binary, and
  identification comes from a change in the *slope* of P(D=1|X), not a level
  jump.

RPKD Estimator (Dong 2018, Proposition 1)
------------------------------------------
The RPKD LATE is identified as:

  tau_RPKD = [dE(Y|X)/dX|_{0+}  −  dE(Y|X)/dX|_{0-}]
             ─────────────────────────────────────────
             [dP(D|X)/dX|_{0+}  −  dP(D|X)/dX|_{0-}]

i.e., the kink in the conditional mean of Y divided by the kink in the
treatment probability. This is analogous to the fuzzy RDD Wald ratio, with
slope changes replacing level jumps.

Both kinks are estimated jointly via a single local polynomial regression
that imposes continuity at the cutoff (no jump allowed) but permits a slope
change. Using a single joint model on both sides within the bandwidth
produces efficient estimates and a well-defined joint covariance matrix for
the delta-method SE.

Estimation (triangular-kernel WLS, linear polynomial)
------------------------------------------------------
  Within bandwidth h, weight w_i = 1 − |X_i| / h.
  Model: Y_i = α + β₁·X_i + β₂·(D_i·X_i) + ε_i
    where D_i = 1(X_i ≥ 0).
    β₁ = slope for X < 0.
    β₁ + β₂ = slope for X ≥ 0.
    β₂ = kink = slope change at X = 0.
  Note: no D_i main effect term → continuity at zero is imposed.

Delta-method SE for tau_RPKD
-----------------------------
Both β₂_Y (reduced form) and β₂_D (first stage) are estimated from the
same observations, so they share a joint covariance matrix. We stack the
two WLS systems into a single GMM-style sandwich estimator to get the full
(4×4) joint covariance matrix of (α_Y, β₁_Y, β₂_Y, α_D, β₁_D, β₂_D).
The gradient of tau = β₂_Y / β₂_D w.r.t. this vector is:
  ∂tau/∂β₂_Y = 1/β₂_D
  ∂tau/∂β₂_D = −β₂_Y / β₂_D²
  all others = 0.

Inputs
------
  data/processed/rkd_outcomes.parquet
  data/processed/rdd_data_nathan.parquet

Outputs
-------
  output/rkd_results.csv
  output/figures/rkd_set1.png ... rkd_set4.png

Usage
-----
  cd /home/alexander/Documents/coding_projects/baseball/hr_rdd
  .venv/bin/python src/12_rkd_analysis.py

Reference
---------
Dong, Y. (2018). Jump or kink? Identifying education effects by regression
discontinuity design without the discontinuity. Working paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OUTCOMES_PATH = ROOT / "data" / "processed" / "rkd_outcomes.parquet"
RDD_PATH      = ROOT / "data" / "processed" / "rdd_data_nathan.parquet"
OUT_DIR       = ROOT / "output"
FIG_DIR       = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Outcome sets ──────────────────────────────────────────────────────────────
OUTCOME_SETS = {
    1: {
        "label": "Pitcher Rest-of-Inning",
        "outcomes": [
            "pitcher_replaced",
            "mean_release_speed",
            "mean_spin_rate",
            "fastball_pct",
            "breaking_pct",
            "zone_pct",
            "whiff_pct",
            "swing_pct",
            "runs_allowed",
        ],
    },
    2: {
        "label": "Opp Pitcher Next Half-Inning",
        "outcomes": [
            "mean_opp_release_speed",
            "mean_opp_spin_rate",
            "opp_fastball_pct",
            "opp_breaking_pct",
            "opp_zone_pct",
            "opp_whiff_pct",
            "opp_swing_pct",
            "opp_runs_allowed",
        ],
    },
    3: {
        "label": "Offensive Response Next Half-Inning",
        "outcomes": [
            "off_runs_scored",
            "off_swing_pct",
            "off_whiff_pct",
            "off_launch_speed_mean",
            "off_launch_angle_mean",
            "off_hard_hit_pct",
            "off_woba",
            "off_bb_pct",
            "off_k_pct",
        ],
    },
    4: {
        "label": "Batter Next At-Bat",
        "outcomes": [
            "next_ab_hit",
            "next_ab_walk",
            "next_ab_k",
            "next_ab_swing_pct",
            "next_ab_whiff_pct",
            "next_ab_chase_pct",
            "next_ab_zone_swing_pct",
            "next_ab_launch_speed",
            "next_ab_woba",
        ],
    },
}

BAT_SPEED_OUTCOMES = ["next_ab_bat_speed", "next_ab_swing_length"]
BAT_SPEED_COVERAGE_THRESHOLD = 0.20

# Fixed sensitivity bandwidths always reported.
SENSITIVITY_BWS = [10, 20, 30]
PLOT_WINDOW = 40   # ft, x-axis window for plots
BIN_WIDTH   = 2.0  # ft bin width for binned scatter

# Try to import rdbwselect for per-outcome MSE-optimal bandwidth selection.
try:
    from rdrobust import rdbwselect as _rdbwselect
    HAS_RDBWSELECT = True
except ImportError:
    HAS_RDBWSELECT = False
    print("WARNING: rdrobust not installed — skipping MSE-optimal bandwidth selection.")


# ── RPKD estimator ────────────────────────────────────────────────────────────

def _design_matrix(X: np.ndarray) -> np.ndarray:
    """
    Build the 2-column design matrix [X, D*X] where D = 1(X >= 0).
    Intercept is added by sm.add_constant; no D main effect → continuity imposed.
    """
    D  = (X >= 0).astype(float)
    return np.column_stack([X, D * X])


def _wls_kink(X: np.ndarray, Y: np.ndarray, h: float) -> dict:
    """
    Fit the continuity-imposing kink model within bandwidth h.

    Model: Y = α + β₁·X + β₂·(D·X) + ε   (no D main effect → no jump at 0)
    Triangular kernel: w_i = 1 − |X_i| / h

    Returns: kink (β₂), se (HC1), full param vector, full vcov, n, mask.
    """
    mask = (np.abs(X) <= h) & np.isfinite(Y) & np.isfinite(X)
    Xm   = X[mask]
    Ym   = Y[mask]
    n    = int(mask.sum())

    if n < 10:
        return dict(kink=np.nan, se=np.nan, params=None, vcov=None, n=n, mask=mask)

    kern = 1.0 - np.abs(Xm) / h
    Xmat = sm.add_constant(_design_matrix(Xm), has_constant=False)
    # Xmat columns: [1, X, D*X]  →  indices 0=const, 1=X, 2=D*X (kink)

    try:
        res  = sm.WLS(Ym, Xmat, weights=kern).fit(cov_type="HC1")
    except Exception:
        return dict(kink=np.nan, se=np.nan, params=None, vcov=None, n=n, mask=mask)

    return dict(
        kink   = float(res.params[2]),
        se     = float(res.bse[2]),
        params = res.params.to_numpy(),
        vcov   = np.asarray(res.cov_params()),
        n      = n,
        mask   = mask,
        res    = res,
    )


def rpkd_estimate(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    h: float,
) -> dict:
    """
    Regression Probability Kink Design estimator (Dong 2018).

    tau_RPKD = kink_Y / kink_T

    where kink_Y = slope change in E(Y|X) at 0,
          kink_T = slope change in P(D=1|X) at 0.

    Both kinks are estimated on the same set of observations within the
    bandwidth, so we form the joint covariance matrix of (kink_Y, kink_T)
    via a GMM sandwich estimator to get correct delta-method SEs.

    The joint system stacks the two WLS score equations:
        score_Y_i = X_mat_i' * w_i * e_Y_i
        score_T_i = X_mat_i' * w_i * e_T_i
    The meat of the sandwich is E[score_Y * score_T'] which we estimate
    empirically, giving us Cov(β̂_Y, β̂_T) and hence the cross-term in the
    delta-method formula.

    Parameters
    ----------
    X : running variable
    Y : outcome
    T : binary treatment (home_run)
    h : bandwidth

    Returns
    -------
    dict with keys: reduced_form_kink, first_stage_kink, rpkd_estimate,
                    se, ci_lower, ci_upper, p_value, n
    """
    # Common mask: within bandwidth, both Y and X finite
    mask = (np.abs(X) <= h) & np.isfinite(Y) & np.isfinite(X) & np.isfinite(T)
    n    = int(mask.sum())

    if n < 10:
        return dict(
            reduced_form_kink=np.nan, first_stage_kink=np.nan,
            rpkd_estimate=np.nan, se=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, p_value=np.nan, n=n,
        )

    Xm   = X[mask]
    Ym   = Y[mask]
    Tm   = T[mask]
    kern = 1.0 - np.abs(Xm) / h                     # triangular weights
    Xmat = sm.add_constant(_design_matrix(Xm), has_constant=False)
    # columns: [1, X, D*X]  — kink coefficient is index 2

    # ── Fit both equations by WLS ─────────────────────────────────────────
    try:
        res_Y = sm.WLS(Ym, Xmat, weights=kern).fit()
        res_T = sm.WLS(Tm, Xmat, weights=kern).fit()
    except Exception:
        return dict(
            reduced_form_kink=np.nan, first_stage_kink=np.nan,
            rpkd_estimate=np.nan, se=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, p_value=np.nan, n=n,
        )

    kink_Y = float(res_Y.params[2])
    kink_T = float(res_T.params[2])

    if kink_T == 0.0 or not np.isfinite(kink_T):
        return dict(
            reduced_form_kink=kink_Y, first_stage_kink=kink_T,
            rpkd_estimate=np.nan, se=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, p_value=np.nan, n=n,
        )

    tau = kink_Y / kink_T

    # ── Joint sandwich SE ─────────────────────────────────────────────────
    # Weighted design: W^{1/2} X_mat
    Xw   = Xmat * kern[:, None]          # n × 3, each row scaled by w_i
    XtWX = Xw.T @ Xmat                   # 3 × 3  (= X'WX)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return dict(
            reduced_form_kink=kink_Y, first_stage_kink=kink_T,
            rpkd_estimate=tau, se=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, p_value=np.nan, n=n,
        )

    e_Y  = Ym - Xmat @ res_Y.params     # residuals
    e_T  = Tm - Xmat @ res_T.params

    # Scores: (n × 3) for each equation
    score_Y = Xmat * (kern * e_Y)[:, None]   # n × 3
    score_T = Xmat * (kern * e_T)[:, None]   # n × 3

    # HC1 correction factor
    hc1 = n / (n - Xmat.shape[1])

    # 3×3 blocks of the joint meat matrix
    meat_YY = hc1 * (score_Y.T @ score_Y)    # Cov(score_Y, score_Y)
    meat_TT = hc1 * (score_T.T @ score_T)
    meat_YT = hc1 * (score_Y.T @ score_T)    # cross-covariance

    # Sandwich covariance blocks
    # Cov(β̂_Y) = (X'WX)^{-1} meat_YY (X'WX)^{-1}
    vcov_Y  = XtWX_inv @ meat_YY @ XtWX_inv
    vcov_T  = XtWX_inv @ meat_TT @ XtWX_inv
    vcov_YT = XtWX_inv @ meat_YT @ XtWX_inv  # Cov(β̂_Y, β̂_T)

    # Variances of kink estimates (index 2)
    var_kink_Y  = vcov_Y[2, 2]
    var_kink_T  = vcov_T[2, 2]
    cov_kink_YT = vcov_YT[2, 2]

    # Delta method: Var(kink_Y / kink_T)
    # ≈ (1/kink_T)^2 * Var(kink_Y)
    #   + (kink_Y/kink_T^2)^2 * Var(kink_T)
    #   − 2 * (1/kink_T) * (kink_Y/kink_T^2) * Cov(kink_Y, kink_T)
    a  = 1.0 / kink_T
    b  = -kink_Y / kink_T ** 2
    var_tau = (a ** 2 * var_kink_Y
               + b ** 2 * var_kink_T
               + 2.0 * a * b * cov_kink_YT)

    if var_tau < 0 or not np.isfinite(var_tau):
        se_tau = np.nan
        z      = np.nan
        p_val  = np.nan
    else:
        se_tau = np.sqrt(var_tau)
        z      = tau / se_tau
        p_val  = 2.0 * (1.0 - _norm_cdf(abs(z))) if np.isfinite(z) else np.nan

    ci_lo = tau - 1.96 * se_tau if np.isfinite(se_tau) else np.nan
    ci_hi = tau + 1.96 * se_tau if np.isfinite(se_tau) else np.nan

    return dict(
        reduced_form_kink = kink_Y,
        first_stage_kink  = kink_T,
        rpkd_estimate     = tau,
        se                = se_tau,
        ci_lower          = ci_lo,
        ci_upper          = ci_hi,
        p_value           = p_val,
        n                 = n,
    )


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _local_linear_fit(X, Y, h, side):
    """
    Return (x_grid, y_hat) for local linear fit on one side within bandwidth.
    side: 'left' (X<0) or 'right' (X>=0)
    """
    if side == "left":
        mask = (X >= -h) & (X < 0) & np.isfinite(Y)
        x_grid = np.linspace(-h, 0, 100)
    else:
        mask = (X >= 0) & (X <= h) & np.isfinite(Y)
        x_grid = np.linspace(0, h, 100)

    Xm = X[mask]
    Ym = Y[mask]
    if len(Xm) < 5:
        return x_grid, np.full_like(x_grid, np.nan)

    kern = 1.0 - np.abs(Xm) / h
    Xmat = sm.add_constant(Xm)
    try:
        res = sm.WLS(Ym, Xmat, weights=kern).fit()
        y_hat = res.params[0] + res.params[1] * x_grid
    except Exception:
        y_hat = np.full_like(x_grid, np.nan)

    return x_grid, y_hat


def plot_rkd_outcome(ax, X, Y, outcome, rkd_res_opt, bin_width=BIN_WIDTH, window=PLOT_WINDOW, bw_plot=20):
    """Plot binned scatter + local linear fits for one outcome.
    bw_plot is the MSE-optimal bandwidth for this specific outcome.
    """
    # Filter to window
    mask = (np.abs(X) <= window) & np.isfinite(Y)
    Xw   = X[mask]
    Yw   = Y[mask]

    # Binned means
    bins  = np.arange(-window, window + bin_width, bin_width)
    mids  = bins[:-1] + bin_width / 2
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (Xw >= lo) & (Xw < hi)
        means.append(Yw[in_bin].mean() if in_bin.sum() > 0 else np.nan)
    means = np.array(means)

    ax.scatter(mids, means, s=16, color="steelblue", zorder=3, alpha=0.85)

    # Local linear fits within bw_plot (outcome-specific optimal BW)
    mask_full = np.isfinite(Y)
    for side, color in [("left", "steelblue"), ("right", "tomato")]:
        xg, yh = _local_linear_fit(X[mask_full], Y[mask_full], bw_plot, side)
        ax.plot(xg, yh, color=color, linewidth=1.8, zorder=4)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, zorder=2)
    ax.set_xlabel("Dist to Fence (ft)", fontsize=7)

    # Annotate RPKD estimate at optimal BW
    tau = rkd_res_opt.get("rpkd_estimate", np.nan)
    se  = rkd_res_opt.get("se", np.nan)
    pv  = rkd_res_opt.get("p_value", np.nan)
    bw_label = f"BW={bw_plot:.0f}"
    if np.isfinite(tau) and np.isfinite(se):
        stars = ""
        if np.isfinite(pv):
            if pv < 0.01:
                stars = "**"
            elif pv < 0.05:
                stars = "*"
        ax.set_title(
            f"{outcome}\nRPKD={tau:.3f} ({se:.3f}){stars} [{bw_label}]",
            fontsize=6.5, pad=2
        )
    else:
        ax.set_title(f"{outcome}\n[{bw_label}]", fontsize=6.5, pad=2)


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(results_df: pd.DataFrame, optimal_bw: dict, out_dir: Path) -> None:
    """Write a formatted markdown file with all RPKD results tables."""
    md_path = out_dir / "rkd_results.md"
    lines = []

    lines.append("# Regression Probability Kink Design Results")
    lines.append("")
    lines.append("**Design:** Dong (2018) Regression Probability Kink Design (RPKD)")
    lines.append("")
    lines.append("**Running variable:** `coord_fence_dist_ft` — signed distance from hit "
                 "coordinate to outfield fence (ft). Negative = inside park; positive = beyond fence.")
    lines.append("")
    lines.append("**Cutoff:** 0 ft. At the cutoff, P(home run | X) has a kink (slope change) "
                 "but no discontinuous jump. Identification comes from this kink.")
    lines.append("")
    lines.append("**Estimator:**")
    lines.append("")
    lines.append("```")
    lines.append("tau_RPKD = [kink in E(Y|X) at 0] / [kink in P(D=1|X) at 0]")
    lines.append("```")
    lines.append("")
    lines.append("Both kinks estimated via triangular-kernel WLS with continuity imposed at the "
                 "cutoff (no jump term). Delta-method SEs account for the joint covariance of "
                 "the numerator and denominator kinks (full GMM sandwich).")
    lines.append("")
    lines.append("**Bandwidth selection:** MSE-optimal bandwidth selected **separately for each "
                 "outcome** using `rdbwselect(deriv=1)` (Card, Lee, Pei & Weber 2015; Calonico, "
                 "Cattaneo & Titiunik 2014). `deriv=1` targets slope/kink estimation, which "
                 "converges at rate n^(-2/5), yielding wider optimal bandwidths than level "
                 "estimation. Results at fixed bandwidths of 10, 20, and 30 ft are shown as "
                 "sensitivity checks. The optimal BW row is marked with ✓.")
    lines.append("")
    lines.append("**Outs filter (Set 1 only):** Restricted to fly balls with < 2 outs. "
                 "With 2 outs, any non-HR out ends the inning, so there is no rest-of-inning "
                 "to observe.")
    lines.append("")
    lines.append("---")
    lines.append("")

    set_labels = {
        1: "Set 1: Pitcher Rest-of-Inning",
        2: "Set 2: Opposing Pitcher — Next Half-Inning",
        3: "Set 3: Offensive Response — Next Half-Inning",
        4: "Set 4: Batter's Next At-Bat (same game)",
    }

    col_headers = [
        "Outcome", "BW (ft)", "Opt", "N",
        "RF Kink", "FS Kink", "RPKD Est", "SE", "CI Lower", "CI Upper", "p-value"
    ]

    def _md_fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.4f}"

    def _md_fmt_p(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if v < 0.001:
            return "< 0.001 **"
        stars = ""
        if v < 0.01:
            stars = " **"
        elif v < 0.05:
            stars = " *"
        return f"{v:.3f}{stars}"

    for set_id, label in set_labels.items():
        subset = results_df[results_df["outcome_set"] == set_id]
        if subset.empty:
            continue

        lines.append(f"## {label}")
        lines.append("")
        lines.append("| " + " | ".join(col_headers) + " |")
        lines.append("|" + "|".join(["---"] * len(col_headers)) + "|")

        for _, row in subset.iterrows():
            is_opt = "✓" if row["is_optimal_bw"] else ""
            cells = [
                str(row["outcome"]),
                f"{row['bandwidth_ft']:.1f}",
                is_opt,
                f"{int(row['n']):,}",
                _md_fmt(row["reduced_form_kink"]),
                _md_fmt(row["first_stage_kink"]),
                _md_fmt(row["rpkd_estimate"]),
                _md_fmt(row["se"]),
                _md_fmt(row["ci_lower"]),
                _md_fmt(row["ci_upper"]),
                _md_fmt_p(row["p_value"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("\\* p < 0.05 &nbsp;&nbsp; \\*\\* p < 0.01")
    lines.append("")
    lines.append("**Opt (✓):** MSE-optimal bandwidth for that outcome's reduced-form kink "
                 "from `rdbwselect(deriv=1)`.")

    md_path.write_text("\n".join(lines))
    print(f"Saved markdown → {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load and merge data ────────────────────────────────────────────────
    print("Loading data...")
    outcomes_df = pd.read_parquet(OUTCOMES_PATH)
    rdd_df      = pd.read_parquet(RDD_PATH, columns=[
        "game_pk", "at_bat_number", "inning", "inning_topbot",
        "running_var_nathan", "home_run",
    ])

    # Merge coord_fence_dist_ft (running_var_nathan) and home_run
    df = outcomes_df.merge(
        rdd_df.rename(columns={"running_var_nathan": "coord_fence_dist_ft"}),
        left_on=["fly_game_pk", "fly_at_bat_number", "fly_inning", "fly_inning_topbot"],
        right_on=["game_pk", "at_bat_number", "inning", "inning_topbot"],
        how="left",
    )

    print(f"  Merged: {df.shape[0]:,} rows, coord_fence_dist_ft non-null: "
          f"{df['coord_fence_dist_ft'].notna().sum():,}")

    X_all = df["coord_fence_dist_ft"].to_numpy(dtype=float)
    T_all = df["home_run"].to_numpy(dtype=float)

    # ── 2. Determine which bat_speed outcomes to include ─────────────────────
    n_total = len(df)
    bat_speed_included = []
    for col in BAT_SPEED_OUTCOMES:
        if col in df.columns:
            coverage = df[col].notna().sum() / n_total
            if coverage > BAT_SPEED_COVERAGE_THRESHOLD:
                bat_speed_included.append(col)
                print(f"  Including {col} (coverage={coverage:.1%})")
            else:
                print(f"  Skipping {col} (coverage={coverage:.1%} < {BAT_SPEED_COVERAGE_THRESHOLD:.0%})")

    # Add bat_speed outcomes to set 4 if included
    if bat_speed_included:
        OUTCOME_SETS[4]["outcomes"] = OUTCOME_SETS[4]["outcomes"] + bat_speed_included

    # ── 3. Run RPKD for all outcomes × bandwidths ─────────────────────────────
    print("\nRunning RPKD estimates (Dong 2018)...")
    records = []
    # Store optimal bandwidth per outcome for plots and key-outcomes table
    optimal_bw: dict[str, float] = {}

    for set_id, set_info in OUTCOME_SETS.items():
        set_label = set_info["label"]
        for outcome in set_info["outcomes"]:
            if outcome not in df.columns:
                print(f"  WARNING: {outcome} not found in data, skipping")
                continue
            Y_all = df[outcome].to_numpy(dtype=float)

            # Per-outcome MSE-optimal bandwidth (reduced-form kink)
            h_opt = select_bandwidth(X_all, Y_all)
            if h_opt is not None:
                optimal_bw[outcome] = h_opt
                bws_to_run = sorted(set(SENSITIVITY_BWS + [h_opt]))
            else:
                optimal_bw[outcome] = 20.0   # fallback
                bws_to_run = SENSITIVITY_BWS

            for h in bws_to_run:
                res = rpkd_estimate(X_all, Y_all, T_all, h)
                records.append({
                    "outcome_set":       set_id,
                    "outcome_set_label": set_label,
                    "outcome":           outcome,
                    "bandwidth_ft":      h,
                    "optimal_bw":        optimal_bw[outcome],
                    "is_optimal_bw":     (h == optimal_bw[outcome]),
                    "n":                 res["n"],
                    "reduced_form_kink": res["reduced_form_kink"],
                    "first_stage_kink":  res["first_stage_kink"],
                    "rpkd_estimate":     res["rpkd_estimate"],
                    "se":                res["se"],
                    "ci_lower":          res["ci_lower"],
                    "ci_upper":          res["ci_upper"],
                    "p_value":           res["p_value"],
                })

    results_df = pd.DataFrame(records)

    # ── 4. Save CSV ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "rkd_results.csv"
    results_df[[
        "outcome_set", "outcome", "bandwidth_ft", "optimal_bw", "is_optimal_bw",
        "n", "reduced_form_kink", "first_stage_kink",
        "rpkd_estimate", "se", "ci_lower", "ci_upper", "p_value",
    ]].to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # ── 5. Build look-up dict of optimal-BW results for plots ─────────────────
    opt_results = (
        results_df[results_df["is_optimal_bw"]]
        .set_index("outcome")
        [["rpkd_estimate", "se", "p_value", "optimal_bw"]]
        .to_dict(orient="index")
    )

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    for set_id, set_info in OUTCOME_SETS.items():
        outcomes_in_set = [o for o in set_info["outcomes"] if o in df.columns]
        n_out = len(outcomes_in_set)
        if n_out == 0:
            continue

        ncols = 3
        nrows = int(np.ceil(n_out / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"Set {set_id}: {set_info['label']}\n"
            "RPKD (Dong 2018) — binned means (2ft) + local linear fits (MSE-optimal BW per outcome)",
            fontsize=11, y=1.01,
        )

        for idx, outcome in enumerate(outcomes_in_set):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            Y_all = df[outcome].to_numpy(dtype=float)
            opt_bw = optimal_bw.get(outcome, 20.0)
            opt_res = opt_results.get(outcome, {})
            plot_rkd_outcome(ax, X_all, Y_all, outcome, opt_res, bw_plot=opt_bw)

        # Hide unused axes
        for idx in range(n_out, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        fig_path = FIG_DIR / f"rkd_set{set_id}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_path}")

    # ── 7. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RPKD RESULTS SUMMARY (Dong 2018)")
    print("=" * 110)

    for set_id, set_info in OUTCOME_SETS.items():
        print(f"\n--- Set {set_id}: {set_info['label']} ---")
        header = f"{'Outcome':<35} {'BW':>6}  {'Opt?':>4}  {'N':>8}  {'RF Kink':>10}  {'FS Kink':>10}  "
        header += f"{'RPKD Est':>10}  {'SE':>8}  {'CI Lower':>10}  {'CI Upper':>10}  {'p-val':>7}"
        print(header)
        print("-" * len(header))

        subset = results_df[results_df["outcome_set"] == set_id]
        for _, row in subset.iterrows():
            is_opt = "  *" if row["is_optimal_bw"] else "   "
            line = (
                f"{row['outcome']:<35} "
                f"{row['bandwidth_ft']:>6.1f}  "
                f"{is_opt:>4}  "
                f"{int(row['n']):>8}  "
                f"{_fmt(row['reduced_form_kink']):>10}  "
                f"{_fmt(row['first_stage_kink']):>10}  "
                f"{_fmt(row['rpkd_estimate']):>10}  "
                f"{_fmt(row['se']):>8}  "
                f"{_fmt(row['ci_lower']):>10}  "
                f"{_fmt(row['ci_upper']):>10}  "
                f"{_fmt_p(row['p_value']):>7}"
            )
            print(line)

    # ── 8. Key outcomes at optimal BW ─────────────────────────────────────────
    main_outcomes = ["pitcher_replaced", "runs_allowed", "off_runs_scored", "next_ab_hit"]
    print("\n" + "=" * 90)
    print("KEY OUTCOMES AT MSE-OPTIMAL BANDWIDTH")
    print("=" * 90)
    print(f"{'Outcome':<30}  {'Opt BW':>7}  {'RPKD Est':>10}  {'SE':>8}  {'95% CI':>22}  {'p-val':>7}")
    print("-" * 90)
    for outcome in main_outcomes:
        if outcome in opt_results:
            r = opt_results[outcome]
            bw_str = f"{optimal_bw.get(outcome, float('nan')):.1f} ft"
            ci = f"[{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}]" if "ci_lower" in r else "  N/A"
            # Get from results_df for CI
            row_data = results_df[
                (results_df["outcome"] == outcome) & results_df["is_optimal_bw"]
            ]
            if not row_data.empty:
                rd = row_data.iloc[0]
                ci = f"[{_fmt(rd['ci_lower'])}, {_fmt(rd['ci_upper'])}]"
                print(
                    f"{outcome:<30}  {bw_str:>7}  {_fmt(rd['rpkd_estimate']):>10}  "
                    f"{_fmt(rd['se']):>8}  {ci:>22}  {_fmt_p(rd['p_value']):>7}"
                )
    print()

    # ── 9. Write markdown report ──────────────────────────────────────────────
    write_markdown(results_df, optimal_bw, OUT_DIR)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    NaN"
    return f"{v:10.4f}"


def _fmt_p(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    NaN"
    if v < 0.001:
        return " <0.001"
    return f"{v:7.3f}"


def select_bandwidth(X: np.ndarray, Y: np.ndarray) -> float | None:
    """
    Select the MSE-optimal bandwidth for the reduced-form kink of outcome Y
    using rdbwselect with deriv=1 (kink estimand).

    deriv=1 tells rdbwselect to target slope estimation rather than level
    estimation, giving the correct MSE-optimal rate for kink designs
    (n^{-2/5} rather than n^{-3/5}).  Each outcome gets its own bandwidth
    because MSE depends on the outcome's variance and second derivative.

    Falls back to None if rdbwselect is unavailable or fails.
    """
    if not HAS_RDBWSELECT:
        return None
    mask = np.isfinite(X) & np.isfinite(Y)
    if mask.sum() < 50:
        return None
    try:
        bw_res = _rdbwselect(Y[mask], X[mask], deriv=1)
        h = float(bw_res.bws[0, 0])   # left bandwidth (symmetric by default)
        # Clamp to a sensible range: not smaller than 5 ft or larger than 60 ft
        h = float(np.clip(h, 5.0, 60.0))
        return round(h, 1)
    except Exception:
        return None


if __name__ == "__main__":
    main()
