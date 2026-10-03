"""
11_build_outcomes.py

For each fly ball in the RDD sample (data/processed/rdd_data_nathan.parquet),
join to the full Statcast pitch-by-pitch data and compute 4 sets of outcome variables.

Output: data/processed/rkd_outcomes.parquet

Outcome sets:
  Set 1: Pitcher rest-of-inning performance
  Set 2: Opposing pitcher next half-inning
  Set 3: Offensive response of team that gave up the fly ball (next half-inning at bat)
  Set 4: Batter's next at-bat in the same game
"""

import gc
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RDD_PATH = "data/processed/rdd_data_nathan.parquet"
FULL_PATH = "data/raw/statcast_full_all.parquet"
OUT_PATH  = "data/processed/rkd_outcomes.parquet"

FASTBALLS    = {"FF", "SI", "FC"}
BREAKING     = {"SL", "CU", "KC", "CS", "SV"}
OFFSPEED     = {"CH", "FS", "FO", "SC"}
SWING_DESCS  = {"swinging_strike", "swinging_strike_blocked",
                "foul", "foul_tip", "hit_into_play",
                "foul_bunt", "bunt_foul_tip", "swinging_pitchout",
                "missed_bunt", "foul_pitchout"}
WHIFF_DESCS  = {"swinging_strike", "swinging_strike_blocked",
                "missed_bunt"}
CONTACT_DESCS = SWING_DESCS - WHIFF_DESCS  # swings that made contact


def is_swing(desc):
    """Return boolean Series: True if description is a swing."""
    return desc.isin(SWING_DESCS)


def is_whiff(desc):
    """Return boolean Series: True if description is a whiff (swing-and-miss)."""
    return desc.isin(WHIFF_DESCS)


def is_in_zone(zone):
    """True if zone in 1-9."""
    return zone.isin(range(1, 10))


def is_out_zone(zone):
    """True if zone in 11-14."""
    return zone.isin([11, 12, 13, 14])


# ─────────────────────────────────────────────
# Pitching metrics helper (shared by Set 1 & 2)
# ─────────────────────────────────────────────
def compute_pitching_metrics(pitches: pd.DataFrame, prefix: str = "") -> dict:
    """
    Compute pitching metrics from a DataFrame of pitches.
    prefix: string prefix for metric keys (e.g. '' for Set 1, 'opp_' for Set 2).
    Returns a dict of metric_name -> value.
    """
    n = len(pitches)
    if n == 0:
        keys = [
            "release_speed", "spin_rate", "spin_axis_mean",
            "fastball_pct", "breaking_pct", "offspeed_pct",
            "zone_pct", "chase_pct", "whiff_pct", "swing_pct",
            "runs_allowed", "n_pitches"
        ]
        if prefix == "":
            keys_mapped = [
                "mean_release_speed", "mean_spin_rate", "spin_axis_mean",
                "fastball_pct", "breaking_pct", "offspeed_pct",
                "zone_pct", "chase_pct", "whiff_pct", "swing_pct",
                "runs_allowed", "n_pitches_after"
            ]
        else:
            keys_mapped = [
                f"mean_{prefix}release_speed", f"mean_{prefix}spin_rate",
                f"{prefix}spin_axis_mean",
                f"{prefix}fastball_pct", f"{prefix}breaking_pct",
                f"{prefix}offspeed_pct", f"{prefix}zone_pct",
                f"{prefix}chase_pct", f"{prefix}whiff_pct",
                f"{prefix}swing_pct", f"{prefix}runs_allowed",
                f"{prefix}n_pitches"
            ]
        return {k: np.nan for k in keys_mapped}

    desc    = pitches["description"]
    ptype   = pitches["pitch_type"].fillna("")
    zone    = pitches["zone"]
    swings  = is_swing(desc)
    whiffs  = is_whiff(desc)
    in_zone = is_in_zone(zone)
    out_zone = is_out_zone(zone)

    # Fastball speed: only FF/SI/FC pitches
    fb_mask = ptype.isin(FASTBALLS)
    fb_speeds = pitches.loc[fb_mask, "release_speed"].dropna()
    mean_release_speed = fb_speeds.mean() if len(fb_speeds) > 0 else np.nan

    # Spin rate / axis (all pitches)
    mean_spin_rate  = pitches["release_spin_rate"].dropna().mean()
    spin_axis_mean  = pitches["spin_axis"].dropna().mean() if "spin_axis" in pitches.columns else np.nan

    # Pitch mix
    fastball_pct = fb_mask.mean()
    breaking_pct = ptype.isin(BREAKING).mean()
    offspeed_pct = ptype.isin(OFFSPEED).mean()

    # Zone/chase
    zone_pct = in_zone.mean()
    # Chase = out-of-zone AND swing
    chase_swings = out_zone & swings
    chase_pct = chase_swings.sum() / n

    # Whiff & swing
    n_swings = swings.sum()
    whiff_pct = (whiffs.sum() / n_swings) if n_swings > 0 else np.nan
    swing_pct = n_swings / n

    # Runs allowed: change in the BATTING team's score (bat_score → post_bat_score).
    # fld_score/post_fld_score are always identical in this dataset; the batting
    # team's runs appear in bat_score / post_bat_score.
    first_bat = pitches["bat_score"].iloc[0]
    last_post  = pitches["post_bat_score"].iloc[-1]
    runs_allowed = int(last_post) - int(first_bat)

    n_pitches = n

    if prefix == "":
        return {
            "mean_release_speed": mean_release_speed,
            "mean_spin_rate":     mean_spin_rate,
            "spin_axis_mean":     spin_axis_mean,
            "fastball_pct":       fastball_pct,
            "breaking_pct":       breaking_pct,
            "offspeed_pct":       offspeed_pct,
            "zone_pct":           zone_pct,
            "chase_pct":          chase_pct,
            "whiff_pct":          whiff_pct,
            "swing_pct":          swing_pct,
            "runs_allowed":       runs_allowed,
            "n_pitches_after":    n_pitches,
        }
    else:
        return {
            f"mean_{prefix}release_speed": mean_release_speed,
            f"mean_{prefix}spin_rate":     mean_spin_rate,
            f"{prefix}spin_axis_mean":     spin_axis_mean,
            f"{prefix}fastball_pct":       fastball_pct,
            f"{prefix}breaking_pct":       breaking_pct,
            f"{prefix}offspeed_pct":       offspeed_pct,
            f"{prefix}zone_pct":           zone_pct,
            f"{prefix}chase_pct":          chase_pct,
            f"{prefix}whiff_pct":          whiff_pct,
            f"{prefix}swing_pct":          swing_pct,
            f"{prefix}runs_allowed":       runs_allowed,
            f"{prefix}n_pitches":          n_pitches,
        }


# ─────────────────────────────────────────────
# Set 1: Pitcher rest-of-inning
# ─────────────────────────────────────────────
def compute_set1(fly: pd.Series, game_pitches: pd.DataFrame) -> dict:
    """
    fly: one row from RDD sample.
    game_pitches: all pitches for that game_pk.

    Restricted to fly balls with outs_when_up < 2. With 2 outs, any out ends
    the inning immediately, so there is no meaningful "rest of inning" to
    observe for the pitcher. Returns all-NaN for those observations.
    """
    inning       = fly["fly_inning"]
    topbot       = fly["fly_inning_topbot"]
    fly_ab       = fly["fly_at_bat_number"]
    fly_pitcher  = fly["fly_pitcher"]

    # Outs filter: 2-out fly balls end the inning on any out → no rest-of-inning
    if fly.get("fly_outs_when_up", 0) >= 2:
        nan_metrics = compute_pitching_metrics(pd.DataFrame(), prefix="")
        nan_metrics["pitcher_replaced"] = np.nan
        return nan_metrics

    # All pitches in same inning/half, same game
    inning_pitches = game_pitches[
        (game_pitches["inning"] == inning) &
        (game_pitches["inning_topbot"] == topbot)
    ]

    # Subsequent at-bats in that inning (at_bat_number > fly_ab)
    subsequent = inning_pitches[inning_pitches["at_bat_number"] > fly_ab]

    # pitcher_replaced logic
    if len(subsequent) == 0:
        # No subsequent at-bats: fly ball was last out → NaN
        pitcher_replaced = np.nan
    else:
        # Did the fly_pitcher pitch in any subsequent at-bat?
        flew_pitched_after = (subsequent["pitcher"] == fly_pitcher).any()
        pitcher_replaced = 0 if flew_pitched_after else 1

    # Pitches by fly_pitcher in rest of inning
    pitcher_after = subsequent[subsequent["pitcher"] == fly_pitcher]
    n_after = len(pitcher_after)

    if pd.isna(pitcher_replaced) or pitcher_replaced == 1 or n_after == 0:
        metrics = compute_pitching_metrics(pd.DataFrame(), prefix="")
        metrics["n_pitches_after"] = 0 if (not pd.isna(pitcher_replaced) and pitcher_replaced == 1) else np.nan
    else:
        # Sort by at_bat_number, pitch_number for runs_allowed calc
        pitcher_after_sorted = pitcher_after.sort_values(["at_bat_number", "pitch_number"])
        metrics = compute_pitching_metrics(pitcher_after_sorted, prefix="")

    metrics["pitcher_replaced"] = pitcher_replaced
    return metrics


# ─────────────────────────────────────────────
# Next half-inning helper
# ─────────────────────────────────────────────
def next_half_inning(inning, topbot):
    """Returns (next_inning, next_topbot)."""
    if topbot == "Top":
        return (inning, "Bot")
    else:
        return (inning + 1, "Top")


# ─────────────────────────────────────────────
# Set 2: Opposing pitcher next half-inning
# ─────────────────────────────────────────────
def compute_set2(fly: pd.Series, game_pitches: pd.DataFrame) -> dict:
    """
    The team that HIT the fly ball now pitches next half-inning.
    """
    inning  = fly["fly_inning"]
    topbot  = fly["fly_inning_topbot"]
    next_inn, next_top = next_half_inning(inning, topbot)

    half_inn_pitches = game_pitches[
        (game_pitches["inning"] == next_inn) &
        (game_pitches["inning_topbot"] == next_top)
    ]

    if len(half_inn_pitches) == 0:
        return compute_pitching_metrics(pd.DataFrame(), prefix="opp_")

    half_sorted = half_inn_pitches.sort_values(["at_bat_number", "pitch_number"])
    return compute_pitching_metrics(half_sorted, prefix="opp_")


# ─────────────────────────────────────────────
# Set 3: Offensive response next half-inning
# ─────────────────────────────────────────────
def compute_set3(fly: pd.Series, game_pitches: pd.DataFrame) -> dict:
    """
    Team that GAVE UP the fly ball bats next half-inning.
    """
    inning  = fly["fly_inning"]
    topbot  = fly["fly_inning_topbot"]
    next_inn, next_top = next_half_inning(inning, topbot)

    half_inn_pitches = game_pitches[
        (game_pitches["inning"] == next_inn) &
        (game_pitches["inning_topbot"] == next_top)
    ]

    nan_result = {
        "off_runs_scored":      np.nan,
        "off_swing_pct":        np.nan,
        "off_whiff_pct":        np.nan,
        "off_contact_pct":      np.nan,
        "off_launch_speed_mean": np.nan,
        "off_launch_angle_mean": np.nan,
        "off_hard_hit_pct":     np.nan,
        "off_woba":             np.nan,
        "off_n_pa":             np.nan,
        "off_bb_pct":           np.nan,
        "off_k_pct":            np.nan,
        "off_sb_attempt":       np.nan,
    }

    if len(half_inn_pitches) == 0:
        return nan_result

    hp = half_inn_pitches.sort_values(["at_bat_number", "pitch_number"])

    # Runs scored
    first_bat = hp["bat_score"].iloc[0]
    last_post = hp["post_bat_score"].iloc[-1]
    off_runs_scored = int(last_post) - int(first_bat)

    # Swing/whiff
    desc = hp["description"]
    swings = is_swing(desc)
    whiffs = is_whiff(desc)
    n_total = len(hp)
    n_swings = swings.sum()

    off_swing_pct   = n_swings / n_total if n_total > 0 else np.nan
    off_whiff_pct   = (whiffs.sum() / n_swings) if n_swings > 0 else np.nan
    off_contact_pct = 1.0 - off_whiff_pct if not pd.isna(off_whiff_pct) else np.nan

    # Launch
    bip = hp[hp["launch_speed"].notna() & hp["launch_angle"].notna()]
    off_launch_speed_mean = bip["launch_speed"].mean() if len(bip) > 0 else np.nan
    off_launch_angle_mean = bip["launch_angle"].mean() if len(bip) > 0 else np.nan
    off_hard_hit_pct      = (bip["launch_speed"] >= 95).mean() if len(bip) > 0 else np.nan

    # wOBA (one row per PA that has woba_value)
    woba_rows = hp[hp["woba_value"].notna()]
    off_woba  = woba_rows["woba_value"].mean() if len(woba_rows) > 0 else np.nan

    # PAs: rows with non-null events, excluding intent_walk
    pa_rows = hp[
        hp["events"].notna() &
        (hp["events"] != "intent_walk")
    ]
    off_n_pa = len(pa_rows)

    if off_n_pa > 0:
        off_bb_pct = (pa_rows["events"] == "walk").mean()
        off_k_pct  = pa_rows["events"].isin({"strikeout", "strikeout_double_play"}).mean()
    else:
        off_bb_pct = np.nan
        off_k_pct  = np.nan

    # Stolen base attempt: in pitch-by-pitch Statcast, stolen base plays appear
    # as rows where events contains stolen_base or caught_stealing.
    # These event types do NOT appear in the full statcast data as confirmed by
    # inspection — Statcast pitch-level data does not include SB mid-AB rows.
    # So we code off_sb_attempt = 0 when we have pitches but no SB events visible.
    sb_events = {"stolen_base_2b", "stolen_base_3b", "stolen_base_home",
                 "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home"}
    if hp["events"].isin(sb_events).any():
        off_sb_attempt = 1
    else:
        off_sb_attempt = 0

    return {
        "off_runs_scored":       off_runs_scored,
        "off_swing_pct":         off_swing_pct,
        "off_whiff_pct":         off_whiff_pct,
        "off_contact_pct":       off_contact_pct,
        "off_launch_speed_mean": off_launch_speed_mean,
        "off_launch_angle_mean": off_launch_angle_mean,
        "off_hard_hit_pct":      off_hard_hit_pct,
        "off_woba":              off_woba,
        "off_n_pa":              off_n_pa,
        "off_bb_pct":            off_bb_pct,
        "off_k_pct":             off_k_pct,
        "off_sb_attempt":        off_sb_attempt,
    }


# ─────────────────────────────────────────────
# Set 4: Batter's next at-bat
# ─────────────────────────────────────────────
def compute_set4(fly: pd.Series, game_pitches: pd.DataFrame) -> dict:
    """
    Find the batter's next at-bat in the same game.
    """
    fly_batter = fly["fly_batter"]
    fly_ab     = fly["fly_at_bat_number"]

    # All at-bats by this batter after the fly ball
    batter_pitches = game_pitches[
        (game_pitches["batter"] == fly_batter) &
        (game_pitches["at_bat_number"] > fly_ab)
    ]

    nan_result = {
        "next_ab_hit":            np.nan,
        "next_ab_walk":           np.nan,
        "next_ab_k":              np.nan,
        "next_ab_launch_speed":   np.nan,
        "next_ab_launch_angle":   np.nan,
        "next_ab_woba":           np.nan,
        "next_ab_swing_pct":      np.nan,
        "next_ab_whiff_pct":      np.nan,
        "next_ab_zone_swing_pct": np.nan,
        "next_ab_chase_pct":      np.nan,
        "next_ab_bat_speed":      np.nan,
        "next_ab_swing_length":   np.nan,
    }

    if len(batter_pitches) == 0:
        return nan_result

    # Find the minimum at_bat_number > fly_ab
    next_ab_num = batter_pitches["at_bat_number"].min()
    ab_pitches  = batter_pitches[batter_pitches["at_bat_number"] == next_ab_num]
    ab_sorted   = ab_pitches.sort_values("pitch_number")

    # Last pitch gives the outcome
    last_pitch  = ab_sorted.iloc[-1]
    ev = last_pitch["events"]

    hit_events  = {"single", "double", "triple", "home_run"}
    k_events    = {"strikeout", "strikeout_double_play"}

    next_ab_hit  = 1 if ev in hit_events  else (0 if pd.notna(ev) else np.nan)
    next_ab_walk = 1 if ev == "walk"       else (0 if pd.notna(ev) else np.nan)
    next_ab_k    = 1 if ev in k_events    else (0 if pd.notna(ev) else np.nan)

    # Launch (from last pitch if batted ball)
    next_ab_launch_speed = last_pitch["launch_speed"] if pd.notna(last_pitch.get("launch_speed")) else np.nan
    next_ab_launch_angle = last_pitch["launch_angle"] if pd.notna(last_pitch.get("launch_angle")) else np.nan
    next_ab_woba         = last_pitch["woba_value"]   if pd.notna(last_pitch.get("woba_value"))   else np.nan

    # Swing decisions over entire at-bat
    desc     = ab_sorted["description"]
    zone     = ab_sorted["zone"]
    swings   = is_swing(desc)
    whiffs   = is_whiff(desc)
    in_zone  = is_in_zone(zone)
    out_zone = is_out_zone(zone)

    n_total  = len(ab_sorted)
    n_swings = swings.sum()

    next_ab_swing_pct = n_swings / n_total if n_total > 0 else np.nan
    next_ab_whiff_pct = (whiffs.sum() / n_swings) if n_swings > 0 else np.nan

    # Zone swing: in-zone pitches swung at
    n_in_zone = in_zone.sum()
    next_ab_zone_swing_pct = (
        (in_zone & swings).sum() / n_in_zone
        if n_in_zone > 0 else np.nan
    )

    # Chase: out-of-zone pitches swung at
    n_out_zone = out_zone.sum()
    next_ab_chase_pct = (
        (out_zone & swings).sum() / n_out_zone
        if n_out_zone > 0 else np.nan
    )

    # Bat speed (2024+ only, mean over at-bat)
    if "bat_speed" in ab_sorted.columns:
        bs = ab_sorted["bat_speed"].dropna()
        next_ab_bat_speed   = bs.mean() if len(bs) > 0 else np.nan
    else:
        next_ab_bat_speed = np.nan

    if "swing_length" in ab_sorted.columns:
        sl = ab_sorted["swing_length"].dropna()
        next_ab_swing_length = sl.mean() if len(sl) > 0 else np.nan
    else:
        next_ab_swing_length = np.nan

    return {
        "next_ab_hit":            next_ab_hit,
        "next_ab_walk":           next_ab_walk,
        "next_ab_k":              next_ab_k,
        "next_ab_launch_speed":   next_ab_launch_speed,
        "next_ab_launch_angle":   next_ab_launch_angle,
        "next_ab_woba":           next_ab_woba,
        "next_ab_swing_pct":      next_ab_swing_pct,
        "next_ab_whiff_pct":      next_ab_whiff_pct,
        "next_ab_zone_swing_pct": next_ab_zone_swing_pct,
        "next_ab_chase_pct":      next_ab_chase_pct,
        "next_ab_bat_speed":      next_ab_bat_speed,
        "next_ab_swing_length":   next_ab_swing_length,
    }


# ─────────────────────────────────────────────
# Process one game
# ─────────────────────────────────────────────
def process_game(fly_balls: pd.DataFrame, game_pitches: pd.DataFrame) -> list:
    """
    fly_balls: rows from RDD sample for this game_pk.
    game_pitches: all pitches for this game_pk.
    Returns list of result dicts.
    """
    results = []
    for _, fly in fly_balls.iterrows():
        row = {
            "fly_game_pk":         fly["game_pk"],
            "fly_inning":          fly["fly_inning"],
            "fly_inning_topbot":   fly["fly_inning_topbot"],
            "fly_at_bat_number":   fly["fly_at_bat_number"],
            "fly_pitcher":         fly["fly_pitcher"],
            "fly_batter":          fly["fly_batter"],
        }
        row.update(compute_set1(fly, game_pitches))
        row.update(compute_set2(fly, game_pitches))
        row.update(compute_set3(fly, game_pitches))
        row.update(compute_set4(fly, game_pitches))
        results.append(row)
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Loading RDD sample...")
    rdd = pd.read_parquet(RDD_PATH)
    print(f"  RDD sample: {len(rdd):,} fly balls, {rdd['game_pk'].nunique():,} games")

    # Rename RDD columns to fly_ prefix so they don't collide with game_pitches
    rdd = rdd.rename(columns={
        "inning":        "fly_inning",
        "inning_topbot": "fly_inning_topbot",
        "at_bat_number": "fly_at_bat_number",
        "pitcher":       "fly_pitcher",
        "batter":        "fly_batter",
        "outs_when_up":  "fly_outs_when_up",
    })

    # Get unique game_pks in RDD sample
    rdd_games = set(rdd["game_pk"].unique())
    print(f"  Unique games in RDD sample: {len(rdd_games):,}")

    # Columns we need from full statcast
    NEEDED_COLS = [
        "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "pitcher", "batter",
        "pitch_type", "release_speed", "release_spin_rate", "spin_axis",
        "zone", "description", "events",
        "fld_score", "post_fld_score", "bat_score", "post_bat_score",
        "launch_speed", "launch_angle", "woba_value",
        "bat_speed", "swing_length",
    ]

    print("\nStreaming full statcast data by game...")
    pf = pq.ParquetFile(FULL_PATH)

    # We'll accumulate results in chunks and write parquet at the end
    all_results = []

    # Buffer: accumulate pitches per game_pk across batches
    # Since the parquet may not be sorted by game_pk, we collect all games' pitches
    # in a dict, then process when a game_pk won't appear again.
    # Strategy: load the entire file filtered to rdd_games only.
    # At 744 MB file / ~6.9M rows, with ~80 cols, filtering to relevant games
    # and ~25 cols should fit in memory (~1-2 GB).

    print("  Reading full statcast (filtering to RDD games only)...")
    chunks = []
    total_rows_read = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=NEEDED_COLS):
        df = batch.to_pandas()
        total_rows_read += len(df)
        # Filter to games in RDD sample
        filtered = df[df["game_pk"].isin(rdd_games)]
        if len(filtered) > 0:
            chunks.append(filtered)
        del df
        gc.collect()

    print(f"  Total rows read: {total_rows_read:,}")
    full_filtered = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"  Filtered to RDD games: {len(full_filtered):,} pitches, {full_filtered['game_pk'].nunique():,} games")

    # Group by game_pk and process
    game_groups_full = full_filtered.groupby("game_pk")
    game_groups_rdd  = rdd.groupby("game_pk")

    n_games = len(game_groups_rdd)
    print(f"\nProcessing {n_games:,} games...")

    processed = 0
    for game_pk, fly_balls in game_groups_rdd:
        if game_pk in game_groups_full.groups:
            game_pitches = game_groups_full.get_group(game_pk)
        else:
            # Game not in full data — return all NaN rows
            game_pitches = pd.DataFrame(columns=full_filtered.columns)

        results = process_game(fly_balls, game_pitches)
        all_results.extend(results)

        processed += 1
        if processed % 2000 == 0:
            print(f"  {processed:,}/{n_games:,} games processed ({100*processed/n_games:.1f}%)...")

    print(f"\nDone processing. Total result rows: {len(all_results):,}")

    # Build final DataFrame
    out_df = pd.DataFrame(all_results)

    # ─────────────────────────────────────────
    # Coverage report
    # ─────────────────────────────────────────
    print("\n" + "="*60)
    print("COVERAGE REPORT (N non-null out of", len(out_df), "fly balls)")
    print("="*60)

    outcome_cols = [c for c in out_df.columns if c not in
                    ["fly_game_pk","fly_inning","fly_inning_topbot",
                     "fly_at_bat_number","fly_pitcher","fly_batter"]]

    sets = {
        "Set 1 – Pitcher rest-of-inning": [
            "pitcher_replaced",
            "mean_release_speed", "mean_spin_rate", "spin_axis_mean",
            "fastball_pct", "breaking_pct", "offspeed_pct",
            "zone_pct", "chase_pct", "whiff_pct", "swing_pct",
            "runs_allowed", "n_pitches_after",
        ],
        "Set 2 – Opp pitcher next half-inning": [
            "mean_opp_release_speed", "mean_opp_spin_rate", "opp_spin_axis_mean",
            "opp_fastball_pct", "opp_breaking_pct", "opp_offspeed_pct",
            "opp_zone_pct", "opp_chase_pct", "opp_whiff_pct", "opp_swing_pct",
            "opp_runs_allowed", "opp_n_pitches",
        ],
        "Set 3 – Offensive response next half-inning": [
            "off_runs_scored", "off_swing_pct", "off_whiff_pct", "off_contact_pct",
            "off_launch_speed_mean", "off_launch_angle_mean", "off_hard_hit_pct",
            "off_woba", "off_n_pa", "off_bb_pct", "off_k_pct", "off_sb_attempt",
        ],
        "Set 4 – Batter's next at-bat": [
            "next_ab_hit", "next_ab_walk", "next_ab_k",
            "next_ab_launch_speed", "next_ab_launch_angle", "next_ab_woba",
            "next_ab_swing_pct", "next_ab_whiff_pct",
            "next_ab_zone_swing_pct", "next_ab_chase_pct",
            "next_ab_bat_speed", "next_ab_swing_length",
        ],
    }

    for set_name, cols in sets.items():
        print(f"\n{set_name}:")
        for c in cols:
            if c in out_df.columns:
                n_nn = out_df[c].notna().sum()
                pct  = 100 * n_nn / len(out_df)
                print(f"  {c:<35s}: {n_nn:>7,}  ({pct:5.1f}%)")
            else:
                print(f"  {c:<35s}: MISSING COLUMN")

    # Save
    print(f"\nSaving to {OUT_PATH}...")
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {len(out_df):,} rows x {len(out_df.columns)} cols")
    print("\nDone!")


if __name__ == "__main__":
    main()
