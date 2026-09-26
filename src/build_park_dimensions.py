"""
build_park_dimensions.py
------------------------
Converts the GeomMLBStadiums outfield fence polygon (MLBAM coordinates)
into a fine-grained (spray_angle_deg, fence_dist_ft, fence_height_ft) lookup
table for every MLB park, replacing the hand-coded park_dimensions.csv.

Coordinate system
-----------------
The mlb_stadia_paths.csv file uses transformed MLBAM coordinates where home
plate is at the origin (0, 0) and y increases toward the outfield.  The
transformation from raw MLBAM (hc_x, hc_y) is:
    x_t = hc_x - 125.42
    y_t = 198.27 - hc_y
These are in MLBAM pixel units.  The standard conversion is:
    1 MLBAM unit ≈ 2.495 feet
(Calibrated so that the 90-ft base paths = ~36 MLBAM units.)

Spray angle convention (matches 02_prepare_data.py)
----------------------------------------------------
    angle = arctan2(x_t, y_t)   degrees
    0°  = straight center field
    negative = left field side
    positive = right field side

Output
------
data/park_dimensions_geom.csv  — one row per degree of spray angle per park/era
Columns: team_abbr, team_name, year_start, year_end,
         spray_angle_deg, fence_dist_ft, fence_height_ft

Fence heights
-------------
GeomMLBStadiums does not include wall heights.  They are sourced here from:
  - Official MLB / team publications
  - Baseball Savant park factor documentation
  - Andrew Clem's stadium diagrams (andrewclem.com)
  - The MLB article on park-adjusted HRs (mlb.com, May 2023)
Heights are specified as piecewise segments: for each park, a list of
(angle_left, angle_right, height_ft) tuples.  The script assigns the
correct height to each polygon point by matching its spray angle to a segment.

Usage
-----
    python src/build_park_dimensions.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ── MLBAM unit → feet conversion ─────────────────────────────────────────────
# The 90-ft basepaths span ~36.07 MLBAM units in the transformed coordinate
# system, giving 90/36.07 ≈ 2.495 ft per unit.
MLBAM_TO_FT = 2.495

# ── Team name → abbreviation map ─────────────────────────────────────────────
TEAM_MAP = {
    "angels":       "LAA",
    "astros":       "HOU",
    "athletics":    "OAK",
    "blue_jays":    "TOR",
    "braves":       "ATL",
    "brewers":      "MIL",
    "cardinals":    "STL",
    "cubs":         "CHC",
    "diamondbacks": "ARI",
    "dodgers":      "LAD",
    "giants":       "SF",
    "guardians":    "CLE",
    "mariners":     "SEA",
    "marlins":      "MIA",
    "mets":         "NYM",
    "nationals":    "WSH",
    "orioles":      "BAL",
    "padres":       "SD",
    "phillies":     "PHI",
    "pirates":      "PIT",
    "rangers":      "TEX",
    "rays":         "TB",
    "red_sox":      "BOS",
    "reds":         "CIN",
    "rockies":      "COL",
    "royals":       "KC",
    "tigers":       "DET",
    "twins":        "MIN",
    "white_sox":    "CWS",
    "yankees":      "NYY",
}

TEAM_FULL = {
    "LAA": "Angel Stadium",        "HOU": "Minute Maid Park",
    "OAK": "Oakland Coliseum",     "TOR": "Rogers Centre",
    "ATL": "Truist Park",          "MIL": "American Family Field",
    "STL": "Busch Stadium",        "CHC": "Wrigley Field",
    "ARI": "Chase Field",          "LAD": "Dodger Stadium",
    "SF":  "Oracle Park",          "CLE": "Progressive Field",
    "SEA": "T-Mobile Park",        "MIA": "loanDepot Park",
    "NYM": "Citi Field",           "WSH": "Nationals Park",
    "BAL": "Camden Yards",         "SD":  "Petco Park",
    "PHI": "Citizens Bank Park",   "PIT": "PNC Park",
    "TEX": "Globe Life Field",     "TB":  "Tropicana Field",
    "BOS": "Fenway Park",          "CIN": "Great American Ball Park",
    "COL": "Coors Field",          "KC":  "Kauffman Stadium",
    "DET": "Comerica Park",        "MIN": "Target Field",
    "CWS": "Guaranteed Rate Field","NYY": "Yankee Stadium",
}

# ── Year validity per team snapshot ──────────────────────────────────────────
# GeomMLBStadiums represents each park as a single snapshot.  We apply each
# snapshot across the Statcast era with manual overrides for known park changes.
# Format: abbr → (year_start, year_end)
YEAR_RANGE = {abbr: (2015, 9999) for abbr in TEAM_MAP.values()}
# Known park changes within Statcast era:
YEAR_RANGE["ATL"] = (2017, 9999)    # Truist Park opened 2017
YEAR_RANGE["TEX"] = (2020, 9999)    # Globe Life Field opened 2020
# We handle Turner Field (ATL 2015-2016) and Globe Life Park (TEX 2015-2019)
# via the hand-coded CSV for those seasons — or simply as approximate fallback.

# Camden Yards LF renovation: wall moved back and raised in 2022.
# The GeomMLBStadiums data reflects the current dimensions (post-2022).
# We'll use it for 2022-present; prior seasons used the old shorter wall.
# For simplicity we apply a height correction in the height table.
YEAR_RANGE["BAL"] = (2022, 9999)

# ── Fence height table ────────────────────────────────────────────────────────
# Format: abbr → list of (angle_left, angle_right, height_ft)
# Angles in same convention as spray_angle (0=CF, neg=LF, pos=RF).
# Segments listed left-to-right (most negative to most positive angle).
# Sources: team publications, Baseball Savant, Andrew Clem, MLB.com park article.
FENCE_HEIGHTS = {
    "ARI": [(-45,  45,  7.5)],
    "ATL": [(-45,  45,  8.0)],
    "BAL": [  # Camden Yards post-2022: LF wall raised to 13 ft, pushed back
        (-45, -28, 13.0),   # LF raised wall (post-2022 renovation)
        (-28,  -5,  7.33),  # LCF
        ( -5,  45,  7.33),  # RF side
    ],
    "BOS": [  # Fenway: Green Monster 37 ft in LF, high wall in CF/RF area
        (-45, -26, 37.0),   # Green Monster
        (-26, -18, 17.0),   # Triangle / Monster seats end
        (-18,  10, 17.0),   # Deep CF (triangle area varies 5-17)
        ( 10,  30,  5.0),   # RF bullpen side
        ( 30,  45,  3.0),   # Pesky Pole corner
    ],
    "CHC": [(-45,  45, 11.5)],   # Wrigley brick wall + basket (~11.5 effective)
    "CWS": [(-45,  45,  8.0)],
    "CIN": [
        (-45, -35, 12.0),   # LF corner raised
        (-35,  35,  9.0),
        ( 35,  45,  9.0),
    ],
    "CLE": [
        (-45, -38, 19.0),   # LF corner high wall
        (-38,  45,  8.0),
    ],
    "COL": [(-45,  45,  8.0)],
    "DET": [(-45,  45,  8.0)],
    "HOU": [
        (-45, -30, 21.0),   # Crawford Boxes
        (-30,  45,  7.0),
    ],
    "KC":  [(-45,  45,  8.0)],
    "LAA": [(-45,  45,  8.0)],
    "LAD": [(-45,  45,  4.0)],   # Short (4 ft) padded wall all around
    "MIA": [(-45,  45,  8.0)],
    "MIL": [(-45,  45,  8.0)],
    "MIN": [(-45,  45,  8.0)],
    "NYM": [(-45,  45,  8.0)],
    "NYY": [(-45,  45,  8.0)],
    "OAK": [(-45,  45,  8.0)],
    "PHI": [(-45,  45,  8.0)],
    "PIT": [
        (-45,  -5,  6.0),   # LF/CF side: 6-ft wall
        ( -5,  10, 10.0),   # CF cutout / varying heights
        ( 10,  45,  6.0),   # RF side
    ],
    "SD":  [(-45,  45,  8.0)],
    "SF":  [
        (-45,  20,  8.0),
        ( 20,  35, 25.0),   # RF Triples Alley high wall
        ( 35,  45, 25.0),   # RF corner / McCovey Cove
    ],
    "SEA": [(-45,  45,  8.0)],
    "STL": [(-45,  45,  8.0)],
    "TB":  [
        (-45, -38, 11.17),  # Tropicana LF corner (canted wall)
        (-38,  45,  6.0),   # Rest of Trop: 6-ft padded wall
    ],
    "TEX": [(-45,  45,  8.0)],   # Globe Life Field
    "TOR": [(-45,  45, 10.0)],   # Rogers Centre: 10-ft padded wall
    "WSH": [(-45,  45,  8.0)],
}

# Hand-coded earlier-era parks not in GeomMLBStadiums snapshot
# (used to fill year gaps)
EXTRA_PARKS = [
    # Turner Field (ATL 2015-2016) — approximate, same shape as Truist
    {"team_abbr": "ATL", "park_name": "Turner Field",
     "year_start": 2015, "year_end": 2016},
    # Globe Life Park (TEX 2015-2019)
    {"team_abbr": "TEX", "park_name": "Globe Life Park",
     "year_start": 2015, "year_end": 2019},
    # Camden Yards pre-2022 renovation
    {"team_abbr": "BAL", "park_name": "Camden Yards (pre-2022)",
     "year_start": 2015, "year_end": 2021},
]
# For extra parks we'll copy the GeomMLBStadiums geometry but override heights
EXTRA_HEIGHTS = {
    ("ATL", 2015): [(-45, 45, 8.0)],
    ("TEX", 2015): [(-45, 45, 8.0)],
    ("BAL", 2015): [(-45, -5, 7.33), (-5, 45, 7.33)],  # old LF wall 7'4"
}


def get_height(abbr: str, spray_deg: float,
               heights: list) -> float:
    """Return fence height for a given spray angle using the height table."""
    for (al, ar, h) in heights:
        if al <= spray_deg < ar:
            return h
    # Fallback: return last segment's height
    return heights[-1][2]


def polygon_to_polar(team_name: str, df_paths: pd.DataFrame
                     ) -> pd.DataFrame:
    """
    Convert outfield_outer polygon points for one team to
    (spray_angle_deg, fence_dist_ft) pairs.

    The mlb_stadia_paths.csv uses raw MLBAM pixel coordinates.
    Home plate is at approximately (125.42, 198.27).
    Transformation:
        x_rel = hc_x - 125.42   (positive = RF side)
        y_rel = 198.27 - hc_y   (positive = away from plate / toward CF)
    1 MLBAM unit ≈ 2.495 feet.
    """
    seg = df_paths[
        (df_paths["team"] == team_name) &
        (df_paths["segment"] == "outfield_outer")
    ].copy()

    if seg.empty:
        return pd.DataFrame()

    # Apply the MLBAM origin shift
    x = seg["x"].values.astype(float) - 125.42   # positive = RF
    y = 198.27 - seg["y"].values.astype(float)     # positive = toward CF

    # Distance from home plate in feet
    dist_ft = np.sqrt(x**2 + y**2) * MLBAM_TO_FT

    # Spray angle
    spray_deg = np.degrees(np.arctan2(x, y))

    return pd.DataFrame({
        "spray_angle_deg": spray_deg,
        "fence_dist_ft": dist_ft,
        "x_mlbam": x,
        "y_mlbam": y,
    })


def interpolate_at_angles(polar_df: pd.DataFrame,
                          angle_resolution: float = 0.5
                          ) -> pd.DataFrame:
    """
    Given raw polygon (spray_angle, fence_dist) points, interpolate to a
    uniform angular grid at `angle_resolution` degree spacing.

    The polygon may not cover the full ±45° range smoothly, so we:
    1. Sort by spray angle
    2. Interpolate fence_dist at each grid angle
    3. Clip to ±45° (fair territory)
    """
    df = polar_df.dropna().copy()
    df = df.sort_values("spray_angle_deg").drop_duplicates("spray_angle_deg")

    # Only keep fair-territory angles (-45° to +45°)
    df = df[df["spray_angle_deg"].between(-50, 50)]

    if len(df) < 3:
        return pd.DataFrame()

    grid = np.arange(-45.0, 45.1, angle_resolution)
    dist_interp = np.interp(
        grid,
        df["spray_angle_deg"].values,
        df["fence_dist_ft"].values,
    )

    return pd.DataFrame({
        "spray_angle_deg": grid,
        "fence_dist_ft": dist_interp,
    })


def build_park_table(paths_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full park dimensions table from GeomMLBStadiums data."""
    rows = []

    for team_name, abbr in TEAM_MAP.items():
        polar = polygon_to_polar(team_name, paths_df)
        if polar.empty:
            print(f"  WARNING: no outfield_outer data for {team_name}")
            continue

        grid = interpolate_at_angles(polar, angle_resolution=0.5)
        if grid.empty:
            print(f"  WARNING: interpolation failed for {team_name}")
            continue

        yr_start, yr_end = YEAR_RANGE[abbr]
        heights = FENCE_HEIGHTS.get(abbr, [(-45, 45, 8.0)])
        park_name = TEAM_FULL.get(abbr, team_name)

        for _, row in grid.iterrows():
            h = get_height(abbr, row["spray_angle_deg"], heights)
            rows.append({
                "team_abbr":      abbr,
                "team_geom_name": team_name,
                "park_name":      park_name,
                "year_start":     yr_start,
                "year_end":       yr_end,
                "spray_angle_deg": round(row["spray_angle_deg"], 2),
                "fence_dist_ft":  round(row["fence_dist_ft"], 2),
                "fence_height_ft": h,
            })

    # ── Extra parks (historical) using same geometry but different era/height ─
    for ep in EXTRA_PARKS:
        abbr = ep["team_abbr"]
        team_name = {v: k for k, v in TEAM_MAP.items()}[abbr]
        polar = polygon_to_polar(team_name, paths_df)
        if polar.empty:
            continue
        grid = interpolate_at_angles(polar, angle_resolution=0.5)
        era_key = (abbr, ep["year_start"])
        heights = EXTRA_HEIGHTS.get(era_key, [(-45, 45, 8.0)])
        for _, row in grid.iterrows():
            h = get_height(abbr, row["spray_angle_deg"], heights)
            rows.append({
                "team_abbr":      abbr,
                "team_geom_name": team_name,
                "park_name":      ep["park_name"],
                "year_start":     ep["year_start"],
                "year_end":       ep["year_end"],
                "spray_angle_deg": round(row["spray_angle_deg"], 2),
                "fence_dist_ft":  round(row["fence_dist_ft"], 2),
                "fence_height_ft": h,
            })

    return pd.DataFrame(rows)


def main():
    paths_csv = DATA_DIR / "mlb_stadia_paths.csv"
    out_csv   = DATA_DIR / "park_dimensions_geom.csv"

    print(f"Loading {paths_csv} ...")
    paths_df = pd.read_csv(paths_csv)
    print(f"  {len(paths_df):,} path points, "
          f"{paths_df['team'].nunique()} teams, "
          f"segments: {list(paths_df['segment'].unique())}")

    print("\nBuilding park dimensions table ...")
    park_df = build_park_table(paths_df)

    print(f"\nResult: {len(park_df):,} rows, {park_df['team_abbr'].nunique()} teams")
    print(f"Spray angle range: {park_df['spray_angle_deg'].min():.1f}° to "
          f"{park_df['spray_angle_deg'].max():.1f}°")
    print(f"Fence distance range: {park_df['fence_dist_ft'].min():.0f} – "
          f"{park_df['fence_dist_ft'].max():.0f} ft")

    # Quick sanity check — spot-check a few known dimensions
    print("\nSpot checks (should be close to known dimensions):")
    checks = [
        ("BOS", 0.0,  "Fenway CF ~390"),
        ("BOS", -42.0, "Fenway LF ~315 (Green Monster)"),
        ("NYY", 45.0, "Yankee RF ~314 (short porch)"),
        ("NYY", -45.0, "Yankee LF ~318"),
        ("COL", 0.0,  "Coors CF ~415"),
        ("PIT", 0.0,  "PNC CF ~399"),
        ("PIT", 45.0, "PNC RF ~320"),
        ("TB",  -45.0, "Trop LF ~315"),
        ("BAL", -30.0, "Camden LF ~364"),
        ("HOU", -43.0, "Minute Maid Crawford Boxes ~315"),
    ]
    for abbr, angle, label in checks:
        row = park_df[
            (park_df["team_abbr"] == abbr) &
            (park_df["spray_angle_deg"].between(angle-1, angle+1))
        ]
        if not row.empty:
            d = row.iloc[0]["fence_dist_ft"]
            h = row.iloc[0]["fence_height_ft"]
            print(f"  {abbr} angle={angle:+.0f}°: {d:.0f} ft, {h:.1f} ft wall  ← {label}")

    park_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # Also print per-park summary
    print("\nPer-park summary (min/CF/max fence distance):")
    summary = (
        park_df[park_df["year_start"] >= 2017]
        .groupby("team_abbr")["fence_dist_ft"]
        .agg(min_dist="min", cf_dist=lambda x: x.iloc[len(x)//2], max_dist="max")
        .round(0)
        .sort_values("team_abbr")
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
