"""Vectorized Nathan-style batted-ball flight model.

The aerodynamic coefficient form follows the Nathan/Statcast-calibrated model
used by the University of Illinois trajectory calculator. The public Statcast
extract lacks batted-ball spin and weather, so defaults are explicit
approximations: 1,800 RPM pure backspin, no wind, 70 F, and park elevation.
"""

from __future__ import annotations

import numpy as np


# Elevations above sea level in feet; values are approximate park elevations.
PARK_ELEVATION_FT = {
    "ARI": 1086, "AZ": 1086, "ATL": 1050, "BAL": 33, "BOS": 20,
    "CHC": 595, "CWS": 595, "CIN": 550, "CLE": 653, "COL": 5280,
    "DET": 600, "HOU": 50, "KC": 750, "LAA": 157, "LAD": 515,
    "MIA": 10, "MIL": 640, "MIN": 815, "NYM": 20, "NYY": 55,
    "OAK": 42, "ATH": 42, "PHI": 39, "PIT": 730, "SD": 20,
    "SEA": 10, "SF": 52, "STL": 466, "TB": 15, "TEX": 551,
    "TOR": 380, "WSH": 25,
}


# Nathan/Statcast-calibrated aerodynamic parameterization (dimensionless).
CD0 = 0.3008
CD_SPIN = 0.0292
CL0 = 0.583
CL1 = 2.333
CL2 = 1.120

BALL_MASS_SLUG = 5.125 / 16.0 / 32.174
BALL_RADIUS_FT = 9.125 / (2.0 * np.pi * 12.0)
BALL_AREA_FT2 = np.pi * BALL_RADIUS_FT**2
GRAVITY_FTPS2 = 32.174
SEA_LEVEL_AIR_DENSITY = 0.0023769  # slugs / ft^3, standard atmosphere


def _air_density(elevation_ft: np.ndarray) -> np.ndarray:
    """Standard-atmosphere density approximation by park elevation."""
    return SEA_LEVEL_AIR_DENSITY * np.exp(-elevation_ft / 27800.0)


def trajectory_clearance(
    launch_speed_mph: np.ndarray,
    launch_angle_deg: np.ndarray,
    spray_angle_deg: np.ndarray,
    fence_dist_ft: np.ndarray,
    fence_height_ft: np.ndarray,
    elevation_ft: np.ndarray,
    spin_rpm: float = 1800.0,
    launch_height_ft: float = 3.0,
    dt: float = 0.01,
    max_time_s: float = 8.0,
) -> dict[str, np.ndarray]:
    """Integrate each launch and estimate vertical clearance at its fence ray.

    Returns running_var_ft, z_at_fence_ft, model_landing_distance_ft, and a
    status code: 1 crossed the fence plane in flight, 2 landed before it, 3
    integration limit reached without crossing. For status 2, the running
    variable is a negative miss score (wall height plus remaining range), not
    a literal ball height at the fence.
    """
    speed = np.asarray(launch_speed_mph, dtype=float) * 1.4666666667
    launch = np.radians(np.asarray(launch_angle_deg, dtype=float))
    spray = np.radians(np.asarray(spray_angle_deg, dtype=float))
    fence_dist = np.asarray(fence_dist_ft, dtype=float)
    wall_height = np.asarray(fence_height_ft, dtype=float)
    elevation = np.asarray(elevation_ft, dtype=float)

    size = len(speed)
    clearance = np.full(size, np.nan)
    z_at_fence = np.full(size, np.nan)
    landing_distance = np.full(size, np.nan)
    status = np.zeros(size, dtype=np.int8)

    valid = (
        np.isfinite(speed) & np.isfinite(launch) & np.isfinite(spray)
        & np.isfinite(fence_dist) & np.isfinite(wall_height)
        & np.isfinite(elevation) & (speed > 0) & (fence_dist > 0)
    )
    if not valid.any():
        return {
            "running_var_ft": clearance,
            "z_at_fence_ft": z_at_fence,
            "model_landing_distance_ft": landing_distance,
            "trajectory_status": status,
        }

    idx = np.flatnonzero(valid)
    v0 = speed[idx]
    theta = launch[idx]
    phi = spray[idx]
    target = fence_dist[idx]
    wall = wall_height[idx]
    rho = _air_density(elevation[idx])

    # Coordinates: x points to right field, y toward straightaway center,
    # z upward. The spin axis is selected perpendicular to horizontal flight
    # so the assumed backspin produces upward Magnus force.
    x = np.zeros(len(idx))
    y = np.zeros(len(idx))
    z = np.full(len(idx), launch_height_ft)
    vx = v0 * np.cos(theta) * np.sin(phi)
    vy = v0 * np.cos(theta) * np.cos(phi)
    vz = v0 * np.sin(theta)

    omega = spin_rpm * (2.0 * np.pi / 60.0)
    omega_x = np.cos(phi)
    omega_y = -np.sin(phi)
    aerodynamic_scale = 0.5 * rho * BALL_AREA_FT2 / BALL_MASS_SLUG
    cd = np.full(len(idx), CD0 + CD_SPIN * (spin_rpm / 1000.0))

    def acceleration(vxv, vyv, vzv, row_idx):
        velocity = np.sqrt(vxv * vxv + vyv * vyv + vzv * vzv)
        safe_velocity = np.maximum(velocity, 1.0e-6)
        spin_parameter = BALL_RADIUS_FT * omega / safe_velocity
        cl = CL2 * spin_parameter / (CL0 + CL1 * spin_parameter)

        scale = aerodynamic_scale[row_idx]
        drag_coefficient = cd[row_idx]
        omega_x_active = omega_x[row_idx]
        omega_y_active = omega_y[row_idx]
        ax_drag = -scale * drag_coefficient * velocity * vxv
        ay_drag = -scale * drag_coefficient * velocity * vyv
        az_drag = -scale * drag_coefficient * velocity * vzv

        # omega_hat cross velocity, with omega_hat=(cos(phi),-sin(phi),0).
        cross_x = -omega_y_active * vzv
        cross_y = -omega_x_active * vzv
        cross_z = omega_x_active * vyv - omega_y_active * vxv
        magnus_scale = scale * cl * velocity
        ax = ax_drag + magnus_scale * cross_x
        ay = ay_drag + magnus_scale * cross_y
        az = az_drag + magnus_scale * cross_z - GRAVITY_FTPS2
        return ax, ay, az

    active = np.ones(len(idx), dtype=bool)
    crossed_fence = np.zeros(len(idx), dtype=bool)
    t = 0.0
    steps = int(np.ceil(max_time_s / dt))

    for _ in range(steps):
        if not active.any():
            break
        current = np.flatnonzero(active)
        x0, y0, z0 = x[current], y[current], z[current]
        vx0, vy0, vz0 = vx[current], vy[current], vz[current]

        a1x, a1y, a1z = acceleration(vx0, vy0, vz0, current)
        a2x, a2y, a2z = acceleration(
            vx0 + 0.5 * dt * a1x,
            vy0 + 0.5 * dt * a1y,
            vz0 + 0.5 * dt * a1z,
            current,
        )
        a3x, a3y, a3z = acceleration(
            vx0 + 0.5 * dt * a2x,
            vy0 + 0.5 * dt * a2y,
            vz0 + 0.5 * dt * a2z,
            current,
        )
        a4x, a4y, a4z = acceleration(
            vx0 + dt * a3x,
            vy0 + dt * a3y,
            vz0 + dt * a3z,
            current,
        )

        x1 = x0 + dt * (vx0 + 2 * (vx0 + 0.5 * dt * a1x) + 2 * (vx0 + 0.5 * dt * a2x) + (vx0 + dt * a3x)) / 6
        y1 = y0 + dt * (vy0 + 2 * (vy0 + 0.5 * dt * a1y) + 2 * (vy0 + 0.5 * dt * a2y) + (vy0 + dt * a3y)) / 6
        z1 = z0 + dt * (vz0 + 2 * (vz0 + 0.5 * dt * a1z) + 2 * (vz0 + 0.5 * dt * a2z) + (vz0 + dt * a3z)) / 6
        vx1 = vx0 + dt * (a1x + 2 * a2x + 2 * a3x + a4x) / 6
        vy1 = vy0 + dt * (a1y + 2 * a2y + 2 * a3y + a4y) / 6
        vz1 = vz0 + dt * (a1z + 2 * a2z + 2 * a3z + a4z) / 6

        r0 = np.hypot(x0, y0)
        r1 = np.hypot(x1, y1)
        crossed = (r1 >= target[current]) & ~crossed_fence[current]
        crossed_idx = current[crossed]
        if crossed.any():
            fraction = np.clip(
                (target[crossed_idx] - r0[crossed])
                / np.maximum(r1[crossed] - r0[crossed], 1.0e-9),
                0.0, 1.0,
            )
            z_cross = z0[crossed] + fraction * (z1[crossed] - z0[crossed])
            clearance[idx[crossed_idx]] = z_cross - wall[crossed_idx]
            z_at_fence[idx[crossed_idx]] = z_cross
            crossed_fence[crossed_idx] = True

        landed = z1 <= 0.0
        landed_idx = current[landed]
        if landed.any():
            fraction_ground = np.clip(
                z0[landed] / np.maximum(z0[landed] - z1[landed], 1.0e-9),
                0.0, 1.0,
            )
            x_ground = x0[landed] + fraction_ground * (x1[landed] - x0[landed])
            y_ground = y0[landed] + fraction_ground * (y1[landed] - y0[landed])
            r_ground = np.hypot(x_ground, y_ground)
            landing_distance[idx[landed_idx]] = r_ground
            landed_short = ~crossed_fence[landed_idx]
            if landed_short.any():
                short_idx = landed_idx[landed_short]
                clearance[idx[short_idx]] = -(
                    wall[short_idx] + np.maximum(
                        target[short_idx] - r_ground[landed_short], 0.0,
                    )
                )
            status[idx[landed_idx]] = np.where(landed_short, 2, 1)
            active[landed_idx] = False

        still_active = current[~landed]
        x[still_active], y[still_active], z[still_active] = x1[~landed], y1[~landed], z1[~landed]
        vx[still_active], vy[still_active], vz[still_active] = vx1[~landed], vy1[~landed], vz1[~landed]
        t += dt

    if active.any():
        unfinished = np.flatnonzero(active)
        unfinished_short = ~crossed_fence[unfinished]
        if unfinished_short.any():
            short_idx = unfinished[unfinished_short]
            clearance[idx[short_idx]] = -(
                wall[short_idx] + np.maximum(
                    target[short_idx] - np.hypot(x[short_idx], y[short_idx]), 0.0,
                )
            )
        landing_distance[idx[unfinished]] = np.hypot(x[unfinished], y[unfinished])
        status[idx[unfinished]] = 3

    return {
        "running_var_ft": clearance,
        "z_at_fence_ft": z_at_fence,
        "model_landing_distance_ft": landing_distance,
        "trajectory_status": status,
    }
