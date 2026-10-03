# Statcast Distance and Coordinate Notes

## Field definitions

Baseball Savant's [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs) labels the batted-ball field `hit_distance` as projected hit distance. MLB's [Hit Distance glossary](https://www.mlb.com/glossary/statcast/hit-distance) describes ordinary Hit Distance as the ground distance to the ball's actual endpoint: where it hits the ground, seats, wall, or fielder's glove. MLB separately defines **Projected HR Distance** as the distance of the projected ground-level landing point for an over-the-fence HR, if unobstructed. Public documentation does not disclose the full tracking, projection, or fallback algorithm for the historical CSV field `hit_distance_sc`, and the current CSV and glossary wording is not fully consistent. Treat `hit_distance_sc` as a Statcast-produced distance metric, not as a transparent measurement of the ball's 3D location at the fence or a guaranteed common endpoint across outcomes.

The current data show the distinction is material. For fly-ball HRs, median transformed HC-coordinate radius is about 413 ft and median `hit_distance_sc` is 401 ft. For fly-ball `field_out`s, the corresponding medians are about 304 ft and 304 ft. Those aggregates do not identify the exact endpoint algorithm, but they are consistent with outcome-dependent endpoint locations and/or different distance treatments for HRs and balls fielded in play.

Baseball Savant defines `hc_x` and `hc_y` as X/Y hit coordinates, but its public CSV documentation does not specify the exact endpoint for every outcome. GeomMLBStadiums documents that its stadium paths are in the same MLBAM HC coordinate frame, where image Y increases downward; its transformation moves home plate to the origin and reverses Y. A spray-chart tutorial describes HC coordinates as where a ball was fielded or landed and shows HRs plotted beyond the wall. These are location coordinates, not documented fence-crossing coordinates. Accordingly, use them cautiously as a proxy for initial spray direction: terminal direction can differ from launch direction because of wind and spin, and the endpoint may differ by event type.

## Fence geometry

The project's `mlb_stadia_paths.csv` comes from the GeomMLBStadiums stadium paths. The `outfield_outer` segment is the exterior stadium outline used for plotting; the source package does not describe it as survey-grade fence-crossing data. This project converts HC pixel units to feet with 2.495 ft/unit, based on approximate base-path length, interpolates the outline radially, assigns wall heights from a separate hand-coded table, and applies current snapshots across years with only selected historical exceptions. These distances, heights, and era matches therefore have measurement and coverage uncertainty.

## Consequences for the old running variable

The trig construction used

```text
(hit_distance_sc - fence_distance) * tan(1.1 * launch_angle) - fence_height
```

as if `hit_distance_sc` were a common ground-level landing range for every batted ball and the ball followed a straight descending line at a fixed angle. Public definitions do not support those assumptions for every event. In particular, if a non-HR distance terminates at a fielder or wall, this formula treats that endpoint as ground level; for an HR it may instead use projected landing distance. The inferred height can therefore have treatment-dependent measurement error. HC coordinates also describe an endpoint/location, not necessarily the initial direction or the wall crossing.

Empirically, the ±1 ft window under the prior trig model held 3,105 observations, 2,683 HRs (86.4%). The HR share fell to 52.8% at ±20 ft. This is not evidence of local random assignment: the running score is strongly outcome-selected and its cutoff is not calibrated to a common physical clearance scale. In that sample, HR/control imbalances included launch speed SMD about 0.68, launch angle about -0.40, and plate-x about -0.27.

## Nathan-style model implemented for comparison

The separate `src/nathan_trajectory.py` implementation integrates Newton's equations with gravity, quadratic drag, and Magnus lift using RK4. It uses launch speed, launch angle, HC-derived spray angle, park fence ray and height, a 3 ft initial contact height, a fixed 1,800 RPM pure-backspin assumption, zero wind, standard 70 F air density adjusted approximately for park elevation, and a published Nathan/Statcast-calibrated drag/lift coefficient form. It is an explicit, reproducible approximation, not a claim to reproduce MLB's proprietary flight tracking or the exact 2026 Excel calculator. Batted-ball spin axis, individual spin, weather, and wind are missing, and park geometry remains approximate.

For model trajectories that cross the fence ray before ground contact, the running variable is modeled ball height at that ray minus wall height. If the simulated ball lands before reaching the fence, the script assigns a negative miss score equal to wall height plus remaining horizontal distance and flags it; that score is **not** a literal height at the wall. Such cases are far from a physically plausible barely-cleared threshold, but the flag must be retained in any later analysis.

### Exploratory comparison results

The comparison run (`src/05_nathan_physics_comparison.py`, using the prepared 2015–2024 fly-ball sample) produced:

| Sample | N | HRs | HR share within +/-1 ft | HR share within +/-5 ft | HRs predicted short | Non-HRs predicted clear |
|---|---:|---:|---:|---:|---:|---:|
| All fly balls | 247,640 | 40,149 | 6.9% | 6.9% | 1.5% | 25.9% |
| Launch angle <= HR median (29 deg) | 61,783 | 21,565 | 12.0% | 12.8% | 2.0% | 37.5% |
| Launch angle <= HR Q1 (26 deg) | 31,212 | 11,881 | 13.7% | 14.0% | 2.3% | 41.0% |

There is no substantial discontinuity at zero in these plots. Within +/-1 ft, the HR shares on the left and right are 6.0% and 7.7% in the full sample, 11.8% and 12.2% below the median-angle cutoff, and 13.7% on each side below Q1.

The model predicts that 29.2% of recorded fly-ball outs and 74.7% of doubles reach the fence ray before their simulated landing point. That is an **unimpeded** trajectory calculation: actual fielders can catch the ball before it reaches the wall, so a positive modeled score for a recorded out is not automatically an aerodynamic false positive. For HRs, median modeled landing distance is about 433 ft versus median Statcast `hit_distance_sc` of 401 ft (median difference +32 ft). This is a calibration warning, not a clean model-error estimate: the field's endpoint definitions are not fully transparent, and the model's spin, weather, wind, and geometry inputs are approximate.

Generated outputs:

- `data/processed/rdd_data_nathan.parquet`
- `output/nathan_running_variable_summary.csv`
- `output/nathan_event_diagnostics.csv`
- `output/figures/nathan_first_stage_launch_angle_samples.png`
- `output/figures/nathan_running_variable_distributions.png`

Do not treat the current physics score as a calibrated clearance measure. Before causal use, validate the simulator's range against a genuinely comparable set of Statcast trajectories, verify the outcome-dependent endpoints, and sensitivity-test spin, wind, atmosphere, and park geometry. The public extract does not include individual batted-ball spin axes or observed wind vectors, so exact individual trajectories cannot be reconstructed from these inputs alone.

## References

- Baseball Savant, [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs).
- MLB, [Hit Distance glossary](https://www.mlb.com/glossary/statcast/hit-distance) and [Projected Home Run Distance glossary](https://www.mlb.com/glossary/statcast/projected-home-run-distance).
- Ben Dilday, [GeomMLBStadiums README](https://github.com/bdilday/GeomMLBStadiums#coordinates).
- Alan Nathan, [3D trajectory calculator](https://baseball.physics.illinois.edu/trajectory-calculator-new3D.html) and [Analysis of Baseball Trajectories](https://baseball.physics.illinois.edu/TrajectoryAnalysis.pdf).
- Jeff Quattrociocchi, [Using Statcast Data to Estimate Minor League Home Run Distance](https://community.fangraphs.com/using-statcast-data-to-estimate-minor-league-home-run-distance/) (HC-coordinate endpoint interpretation; independent analysis, not MLB documentation).
