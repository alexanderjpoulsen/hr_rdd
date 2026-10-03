"""
10_download_full.py
-------------------
Downloads FULL pitch-by-pitch Statcast data (all pitches, no fly-ball filter)
for regular seasons 2015-2024. This expanded dataset is needed to build
outcome variables that depend on subsequent pitches after each fly ball.

Key differences from 01_download_statcast.py:
  - No fly-ball filter: all pitches / all events are kept
  - Expanded column set: adds pitch physics, swing metrics, post-score cols, etc.
  - Output: data/raw/statcast_full_YEAR.parquet and statcast_full_all.parquet

Usage:
    python src/10_download_full.py [--start-year 2015] [--end-year 2024]
    python src/10_download_full.py --no-cache    # force fresh downloads
"""

import argparse
import datetime
import time
from pathlib import Path

import pandas as pd
import pybaseball

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_START = 2015
DEFAULT_END = 2024

# Full desired column set — columns not present in a given year are skipped
# gracefully and reported in the summary.
KEEP_COLS = [
    # ── IDENTIFIERS ──────────────────────────────────────────────────────────
    "game_pk", "game_date", "game_year", "game_type",
    "home_team", "away_team",
    "batter", "pitcher",
    "at_bat_number", "pitch_number",
    "inning", "inning_topbot",
    "player_name",

    # ── PITCH / PA OUTCOME ────────────────────────────────────────────────────
    "events", "description", "type", "bb_type",

    # ── COUNT ────────────────────────────────────────────────────────────────
    "balls", "strikes",

    # ── BATTED BALL ──────────────────────────────────────────────────────────
    "launch_speed", "launch_angle",
    "hc_x", "hc_y",
    "hit_distance_sc",
    # bb_type already listed above, included again for grouping clarity — deduped
    "woba_value",
    "estimated_woba_using_speedangle",
    "babip_value",
    "launch_speed_angle",

    # ── PITCH PHYSICS ─────────────────────────────────────────────────────────
    "pitch_type", "pitch_name",
    "release_speed", "effective_speed",
    "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "zone",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension",
    "ax", "ay", "az",
    "vx0", "vy0", "vz0",
    "api_break_x_arm", "api_break_x_batter_in", "api_break_z_with_gravity",
    "sz_top", "sz_bot",

    # ── SWING METRICS (2024+ only, ~46% coverage) ─────────────────────────────
    "bat_speed", "swing_length", "swing_path_tilt",
    "attack_angle", "attack_direction",

    # ── GAME STATE ────────────────────────────────────────────────────────────
    "outs_when_up",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score",
    "bat_score", "fld_score",
    "post_bat_score", "post_fld_score",
    "post_home_score", "post_away_score",
    "bat_win_exp", "home_win_exp",
    "delta_home_win_exp", "delta_run_exp", "delta_pitcher_run_exp",
    "stand", "p_throws",
    "n_thruorder_pitcher",

    # ── SITUATION ────────────────────────────────────────────────────────────
    "if_fielding_alignment", "of_fielding_alignment",
    "hit_location",
    "fielder_2", "fielder_3", "fielder_4",
    "fielder_5", "fielder_6", "fielder_7",
    "fielder_8", "fielder_9",
]

# Deduplicate while preserving order
KEEP_COLS = list(dict.fromkeys(KEEP_COLS))

# Swing metric columns to track separately
SWING_COLS = ["bat_speed", "swing_length", "swing_path_tilt", "attack_angle", "attack_direction"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_monthly_chunks(year: int):
    """
    Return list of (start, end) date-string pairs covering the MLB regular
    season in ~monthly chunks (avoids Baseball Savant timeouts).
    Season window: March 20 → October 5.
    """
    season_start = datetime.date(year, 3, 20)
    season_end   = datetime.date(year, 10, 5)

    chunks = []
    cursor = season_start
    while cursor < season_end:
        # ~28-day chunks; cap at season end
        chunk_end = min(cursor + datetime.timedelta(days=27), season_end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + datetime.timedelta(days=1)
    return chunks


def download_season(year: int) -> pd.DataFrame:
    """
    Download one full season of Statcast pitch-level data (ALL pitches).
    Data is fetched in monthly chunks with up to 3 retries each.
    Returns a DataFrame with the desired column subset.
    """
    print(f"\n{'='*60}")
    print(f"Downloading FULL Statcast {year} (monthly chunks, all pitches)")
    print(f"{'='*60}")

    pybaseball.cache.enable()
    chunks = build_monthly_chunks(year)
    all_chunks = []

    for chunk_start, chunk_end in chunks:
        print(f"  Chunk: {chunk_start} → {chunk_end}", end=" ... ", flush=True)
        success = False
        for attempt in range(1, 4):
            try:
                chunk_df = pybaseball.statcast(start_dt=chunk_start, end_dt=chunk_end)
                print(f"{len(chunk_df):,} rows")
                all_chunks.append(chunk_df)
                time.sleep(2)   # polite delay
                success = True
                break
            except Exception as exc:
                if attempt < 3:
                    wait = 20 * attempt
                    print(f"\n    Error (attempt {attempt}): retrying in {wait}s … ({exc})")
                    time.sleep(wait)
                else:
                    print(f"\n    FAILED after 3 attempts — skipping chunk "
                          f"{chunk_start}→{chunk_end}: {exc}")
        if not success:
            print(f"  WARNING: chunk {chunk_start}→{chunk_end} was skipped entirely.")

    if not all_chunks:
        raise RuntimeError(f"No data retrieved for {year}")

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates()
    print(f"  Raw rows (all pitches): {len(df):,}")

    # ── Subset columns gracefully ─────────────────────────────────────────────
    present_cols = [c for c in KEEP_COLS if c in df.columns]
    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    if missing_cols:
        print(f"  Columns not found in API for {year}: {missing_cols}")
    df = df[present_cols].copy()

    # ── Type coercion ────────────────────────────────────────────────────────
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    if "game_year" in df.columns:
        df["game_year"] = df["game_year"].astype("Int64")

    print(f"  Kept columns: {len(present_cols)}")
    print(f"  Final rows for {year}: {len(df):,}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def combine_and_report(start_year: int, end_year: int) -> None:
    """
    Read all per-year parquet files from disk, concatenate into a combined
    file, and print a summary report.  Operates year-by-year to cap peak RAM.

    Schema unification: pybaseball returns all-null swing-metric columns as
    int64 in years before 2024, but double in 2024 where real values exist.
    We cast all numeric columns to float64 to ensure a consistent schema.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    combined_path = RAW_DIR / "statcast_full_all.parquet"
    print(f"\nBuilding combined file from per-year parquets …")

    # ── Pass 1: determine unified schema ──────────────────────────────────────
    # Read just the first available year to get baseline column list, then
    # scan all years to detect any double-typed columns that should override int.
    unified_schema: pa.Schema | None = None
    for year in range(start_year, end_year + 1):
        path = RAW_DIR / f"statcast_full_{year}.parquet"
        if not path.exists():
            continue
        schema = pq.read_schema(path)
        if unified_schema is None:
            unified_schema = schema
        else:
            # Promote int64 → double where any year uses double
            new_fields = []
            for i, field in enumerate(unified_schema):
                try:
                    other_field = schema.field(field.name)
                    if other_field.type == pa.float64() and field.type == pa.int64():
                        new_fields.append(pa.field(field.name, pa.float64()))
                    else:
                        new_fields.append(field)
                except KeyError:
                    new_fields.append(field)
            unified_schema = pa.schema(new_fields)

    if unified_schema is None:
        print("ERROR: No year files found — nothing combined.")
        return

    # Strip pandas metadata from unified schema to avoid conflicts
    unified_schema = unified_schema.remove_metadata()

    # ── Pass 2: stream-write each year using unified schema ───────────────────
    first = True
    writer = None
    total_rows = 0
    all_cols: set = set()
    col_nonnull: dict = {}

    for year in range(start_year, end_year + 1):
        path = RAW_DIR / f"statcast_full_{year}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping year {year}")
            continue
        df = pd.read_parquet(path)
        total_rows += len(df)
        all_cols.update(df.columns)
        for col in SWING_COLS:
            if col in df.columns:
                col_nonnull[col] = col_nonnull.get(col, 0) + df[col].notna().sum()

        # Cast to unified schema: convert int64 → float64 where needed
        table = pa.Table.from_pandas(df, preserve_index=False).cast(unified_schema)
        del df

        if first:
            writer = pq.ParquetWriter(combined_path, unified_schema)
            first = False
        writer.write_table(table)
        del table
        print(f"  Appended {year}: {path.stat().st_size / 1e6:.0f} MB")

    if writer is not None:
        writer.close()
        print(f"Combined file saved → {combined_path}  ({combined_path.stat().st_size / 1e6:.0f} MB)")
    else:
        print("ERROR: No year files found — nothing combined.")
        return

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows       : {total_rows:,}")
    print(f"  Total columns    : {len(all_cols)}")

    # Swing metric coverage
    print(f"\n  Swing metric coverage (non-null counts / % of total rows):")
    for col in SWING_COLS:
        if col in all_cols:
            n   = col_nonnull.get(col, 0)
            pct = n / total_rows * 100 if total_rows else 0.0
            print(f"    {col:<30} {pct:5.1f}%  ({n:,} rows)")
        else:
            print(f"    {col:<30} MISSING from API entirely")

    # Desired columns not present in any year
    all_missing = [c for c in KEEP_COLS if c not in all_cols]
    if all_missing:
        print(f"\n  Columns requested but MISSING from API (all years):")
        for c in sorted(all_missing):
            print(f"    - {c}")
    else:
        print(f"\n  All requested columns were present in the API.")

    print(f"\n  Combined parquet : {combined_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download full Statcast pitch-by-pitch data (all pitches)."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START)
    parser.add_argument("--end-year",   type=int, default=DEFAULT_END)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable pybaseball disk cache and force fresh download.",
    )
    parser.add_argument(
        "--combine-only", action="store_true",
        help="Skip downloading; only rebuild the combined parquet from existing year files.",
    )
    args = parser.parse_args()

    if args.no_cache:
        pybaseball.cache.disable()

    downloaded_years = []
    cached_years = []
    all_missing_cols: set = set()

    if not args.combine_only:
        for year in range(args.start_year, args.end_year + 1):
            out_path = RAW_DIR / f"statcast_full_{year}.parquet"

            if out_path.exists():
                print(f"\nSeason {year} already on disk — skipping download: {out_path}")
                print(f"  (use --no-cache + delete the file to re-download)")
                cached_years.append(year)
            else:
                season_df = download_season(year)

                # Track which desired columns were missing
                missing = [c for c in KEEP_COLS if c not in season_df.columns]
                all_missing_cols.update(missing)

                season_df.to_parquet(out_path, index=False)
                print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
                downloaded_years.append(year)

                # Free memory before the next year
                del season_df

                # Polite delay between seasons
                time.sleep(5)

        print(f"\nDownloaded : {downloaded_years if downloaded_years else '(none new)'}")
        print(f"Cached     : {cached_years if cached_years else '(none)'}")

    # Combine all years into one file (memory-efficient, streaming write)
    combine_and_report(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
