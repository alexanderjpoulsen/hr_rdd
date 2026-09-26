"""
01_download_statcast.py
-----------------------
Downloads Statcast pitch-level data for all regular seasons 2015-present
using the pybaseball package. Filters to fly balls with sufficient
launch speed and angle data. Saves annual parquet files to data/raw/.

Usage:
    python src/01_download_statcast.py [--start-year 2015] [--end-year 2024]

The download is chunked by two-week intervals to stay within Baseball
Savant's request limits. pybaseball does this automatically via
statcast(start_dt, end_dt) with the `parallel=True` flag.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pybaseball

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
# Full Statcast era
DEFAULT_START = 2015
DEFAULT_END = 2024   # update each season

# Columns we actually need downstream — pulling the full CSV and subsetting
# keeps raw files small while preserving the full provenance.
KEEP_COLS = [
    # Identifiers
    "game_pk", "game_date", "game_year", "game_type",
    "home_team", "away_team",
    "batter", "pitcher",
    "at_bat_number", "pitch_number",
    "inning", "inning_topbot",

    # Pitch / PA outcome
    "events", "description", "type", "bb_type",

    # Batted ball physics inputs
    "launch_speed", "launch_angle",
    "hc_x", "hc_y",
    "hit_distance_sc",

    # Spin (pitcher spin only; batted ball spin not in public CSV)
    "release_spin_rate",

    # Game situation covariates
    "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score",
    "bat_score", "fld_score",   # batting-team and fielding-team score at pitch time
    "bat_win_exp", "home_win_exp",
    "delta_home_win_exp", "delta_run_exp",
    "stand", "p_throws",
    "n_thruorder_pitcher",

    # Pitch features (for next-pitch outcome analysis)
    "pitch_type", "pitch_name",
    "release_speed", "effective_speed",
    "plate_x", "plate_z",
    "pfx_x", "pfx_z",
    "release_spin_rate",
    "zone",

    # Defensive alignment
    "if_fielding_alignment", "of_fielding_alignment",

    # Location / fielder
    "hit_location",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
]

# Deduplicate in case any column name appears twice above
KEEP_COLS = list(dict.fromkeys(KEEP_COLS))

# Minimum data quality thresholds for a usable batted-ball row
MIN_EXIT_VELO = 60.0   # mph — removes bunt-like contacts and data errors
MIN_LAUNCH_ANGLE = 10.0  # degrees — focus on fly-ball-range trajectories;
                          # true fly balls are typically > 25° but keep a wide
                          # margin here; final filter happens in 02_prepare_data


def season_date_range(year: int):
    """Return (start_dt, end_dt) strings for the MLB regular season."""
    # Opening Day is late March / early April; regular season ends early October
    return f"{year}-03-20", f"{year}-10-05"


def download_season(year: int) -> pd.DataFrame:
    """
    Download one season of Statcast data using monthly chunks to avoid
    Baseball Savant query timeouts. Retries each chunk up to 3 times.
    """
    import datetime

    print(f"\n{'='*60}")
    print(f"Downloading Statcast {year} (monthly chunks)")
    print(f"{'='*60}")

    pybaseball.cache.enable()

    # Build list of ~28-day chunks covering the regular season
    season_start = datetime.date(year, 3, 20)
    season_end   = datetime.date(year, 10, 5)
    chunks = []
    cursor = season_start
    while cursor < season_end:
        chunk_end = min(cursor + datetime.timedelta(days=27), season_end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + datetime.timedelta(days=1)

    all_chunks = []
    for chunk_start, chunk_end in chunks:
        print(f"  Chunk: {chunk_start} → {chunk_end}", end=" ... ", flush=True)
        for attempt in range(1, 4):
            try:
                chunk_df = pybaseball.statcast(start_dt=chunk_start, end_dt=chunk_end)
                print(f"{len(chunk_df):,} rows")
                all_chunks.append(chunk_df)
                time.sleep(2)
                break
            except Exception as e:
                if attempt < 3:
                    wait = 20 * attempt
                    print(f"\n    Timeout/error (attempt {attempt}): retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"\n    FAILED after 3 attempts — skipping chunk {chunk_start}→{chunk_end}: {e}")

    if not all_chunks:
        raise RuntimeError(f"No data retrieved for {year}")

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates()
    print(f"  Raw rows: {len(df):,}")

    # ── Subset to columns we need ────────────────────────────────────────────
    present_cols = [c for c in KEEP_COLS if c in df.columns]
    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: columns not found in {year} data: {missing_cols}")
    df = df[present_cols].copy()

    # ── Filter: batted balls only ────────────────────────────────────────────
    # 'type == X' marks pitches put in play; 'events' non-null confirms PA ended
    if "type" in df.columns:
        df = df[df["type"] == "X"].copy()
    elif "description" in df.columns:
        df = df[df["description"] == "hit_into_play"].copy()

    print(f"  Batted-ball rows: {len(df):,}")

    # ── Filter: fly balls with sufficient physics inputs ─────────────────────
    if "bb_type" in df.columns:
        df = df[df["bb_type"] == "fly_ball"].copy()
        print(f"  Fly-ball rows: {len(df):,}")

    # Drop rows missing critical inputs
    df = df.dropna(subset=["launch_speed", "launch_angle", "hc_x", "hc_y"])

    # Loose quality thresholds
    df = df[df["launch_speed"] >= MIN_EXIT_VELO]
    df = df[df["launch_angle"] >= MIN_LAUNCH_ANGLE]

    print(f"  Rows after quality filter: {len(df):,}")
    print(f"  Home runs: {(df['events'] == 'home_run').sum():,}")
    print(f"  Non-HR fly balls: {(df['events'] != 'home_run').sum():,}")

    # ── Type coercion ────────────────────────────────────────────────────────
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_year"].astype("Int64")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download Statcast data by season.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable pybaseball disk cache and force fresh download."
    )
    args = parser.parse_args()

    if args.no_cache:
        pybaseball.cache.disable()

    all_seasons = []

    for year in range(args.start_year, args.end_year + 1):
        out_path = RAW_DIR / f"statcast_{year}.parquet"

        if out_path.exists():
            print(f"\nSeason {year} already downloaded: {out_path}")
            season_df = pd.read_parquet(out_path)
            print(f"  Loaded {len(season_df):,} rows from cache.")
        else:
            season_df = download_season(year)
            season_df.to_parquet(out_path, index=False)
            print(f"  Saved → {out_path}")
            # Be polite to the Baseball Savant servers between seasons
            time.sleep(5)

        all_seasons.append(season_df)

    # ── Combine all seasons ──────────────────────────────────────────────────
    combined = pd.concat(all_seasons, ignore_index=True)
    combined_path = RAW_DIR / "statcast_all.parquet"
    combined.to_parquet(combined_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Home runs:  {(combined['events'] == 'home_run').sum():,}")
    print(f"  Combined file: {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
