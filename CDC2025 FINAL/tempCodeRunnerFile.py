"""
SDE Monte Carlo trajectories with longer, accuracy-focused phases.

Features:
- Split integrator (symplectic leapfrog for deterministic gravity + stochastic velocity kicks)
  to preserve deterministic orbital structure while modeling continuous noise.
- Longer Phase durations (configurable) with conservative default time-steps for accuracy.
- Monte Carlo ensemble (many runs), mean + percentile corridors, improved spatial plots
  (percentile axis clipping, density hexbin, inset zoom).
- Deterministic Phase B coast (stitchable) for mission-scale visualization (0.01c default).
- Exports CSVs and PNGs.

Instructions:
- Tweak parameters in the PARAMETERS section as desired.
- Run with Python 3 and standard scientific libs: numpy, pandas, matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------
# PARAMETERS (tweak these)
# -----------------------------
G = 6.67430e-11                      # gravitational constant
SOLAR_MASS = 1.98847e+30
AU = 1.495978707e+11
YEAR_SECONDS = 3.155815e+07         # approximate year in seconds (your earlier constant)

SUN_MASS = 1.0 * SOLAR_MASS
TEEGARDEN_MASS = 0.089 * SOLAR_MASS

# Ensemble & solver quality
NUM_RUNS = 200                       # Monte Carlo runs (adjustable)
# Phase duration choices (longer trajectories)
PHASE_A_YEARS = 10.0                 # Phase A extended to 10 years
PHASE_C_YEARS = 2.0                  # Phase C extended to 2 years (local capture region)
# Time steps chosen for accuracy (smaller dt for long runs)
DT_DAYS_PHASE_A = 0.25               # 6-hour steps for Phase A (accurate but heavier)
DT_DAYS_PHASE_C = 0.125              # 3-hour steps for Phase C
# Diffusion strengths (m/s noise applied to vx,vy)
NOISE_ESCAPE = 500.0
NOISE_ARRIVAL = 250.0
RANDOM_SEED = 42

# Phase B (deterministic coast) for mission-scale stitching (optional)
USE_PHASE_B = True
CRUISE_SPEED = 3.0e6                 # m/s (~0.01 c)
COAST_SAMPLE_STEP_DAYS = 30          # sampling cadence for coast (monthly)
# Visualization / percentiles
PCT_LOW = 10
PCT_HIGH = 90
CLIP_LOW = 5
CLIP_HIGH = 95
N_SAMPLE_PLOTS = 20
SHOW_DENSITY = True

OUTDIR = Path("sde_long_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)

# -----------------------------
# Physics helpers
# -----------------------------
def accel_from_pos(x, y, star_mass):
    """Return acceleration (ax, ay) due to gravity at position (x,y) for a point mass star_mass."""
    r2 = x*x + y*y
    r = np.sqrt(r2)
    if r < 1e6:
        return 0.0, 0.0
    inv_r3 = 1.0 / (r2 * r)
    ax = -G * star_mass * x * inv_r3
    ay = -G * star_mass * y * inv_r3
    return ax, ay

# -----------------------------
# Split integrator: leapfrog + stochastic kick
# -----------------------------
def simulate_phase_split(initial_state, t_end_years, star_mass, noise,
                         num_runs=NUM_RUNS, dt_days=DT_DAYS_PHASE_A):
    """
    Simulate an ensemble using a split integrator:
      - deterministic gravity integrated with leapfrog (symplectic)
      - stochastic velocity kicks (additive gaussian noise to vx,vy each dt)
    initial_state: [x, y, vx, vy]
    returns: times (s), all_runs ndarray (num_runs, n_steps, 4), df_mean, df_pct (radial)
    """
    dt = dt_days * 24 * 3600.0
    n_steps = int(np.floor(t_end_years * YEAR_SECONDS / dt)) + 1
    times = np.linspace(0.0, (n_steps-1)*dt, n_steps)
    all_runs = np.zeros((num_runs, n_steps, 4), dtype=float)

    diffusion_vector = np.array([0.0, 0.0, noise, noise], dtype=float)

    for run in range(num_runs):
        # unpack initial
        x, y, vx, vy = map(float, initial_state)
        # store
        all_runs[run, 0, :] = np.array([x, y, vx, vy])
        # precompute random normals as shape (n_steps-1, 2) for vx,vy kicks to reduce overhead
        # but we will use scalar normals multiplied by sqrt(dt)
        normals = np.random.randn(n_steps-1, 2)
        sqrt_dt = sqrt(dt)

        # Leapfrog requires initial half-step velocity update. We compute v_half = v + 0.5*dt*a(x)
        ax, ay = accel_from_pos(x, y, star_mass)
        vx_half = vx + 0.5 * dt * ax
        vy_half = vy + 0.5 * dt * ay

        for j in range(1, n_steps):
            # full position update using half-step velocity
            x = x + dt * vx_half
            y = y + dt * vy_half

            # compute new acceleration at updated position
            ax_new, ay_new = accel_from_pos(x, y, star_mass)

            # complete velocity step
            vx_new = vx_half + 0.5 * dt * ax_new
            vy_new = vy_half + 0.5 * dt * ay_new

            # stochastic kick (additive in velocity)
            dvx = noise * sqrt_dt * normals[j-1, 0]
            dvy = noise * sqrt_dt * normals[j-1, 1]
            vx_new += dvx
            vy_new += dvy

            # prepare next half-step velocity
            vx_half = vx_new + 0.5 * dt * ax_new
            vy_half = vy_new + 0.5 * dt * ay_new

            # store
            all_runs[run, j, :] = np.array([x, y, vx_new, vy_new])

        # end run loop
    # compute mean path
    mean_path = np.mean(all_runs, axis=0)  # (n_steps,4)
    df_mean = pd.DataFrame({
        "time_s": times,
        "x": mean_path[:, 0],
        "y": mean_path[:, 1],
        "vx": mean_path[:, 2],
        "vy": mean_path[:, 3]
    })
    # radial statistics
    radial = np.sqrt(all_runs[:, :, 0]**2 + all_runs[:, :, 1]**2)  # (num_runs, n_steps)
    df_pct = pd.DataFrame({
        "time_s": times,
        "r_mean": np.mean(radial, axis=0),
        "r_low": np.percentile(radial, PCT_LOW, axis=0),
        "r_high": np.percentile(radial, PCT_HIGH, axis=0)
    })
    return times, all_runs, df_mean, df_pct

# -----------------------------
# Deterministic coast Phase B (simple linear motion)
# -----------------------------
def deterministic_coast(start_pos, start_time_s, cruise_speed, coast_years, step_days=COAST_SAMPLE_STEP_DAYS):
    """
    Build coarse deterministic coast sampled at 'step_days' (good for mission-scale stitching).
    start_pos: (x, y) starting coordinates (m)
    start_time_s: mission time at coast start
    returns times_coast (s relative to mission start), coast_x, coast_y arrays
    """
    dt_coast = step_days * 24 * 3600.0
    n_steps = int(np.floor(coast_years * YEAR_SECONDS / dt_coast)) + 1
    times = start_time_s + np.linspace(0.0, (n_steps-1) * dt_coast, n_steps)
    # for simplicity assume motion along +x direction from start_pos
    x0, y0 = start_pos
    coast_x = x0 + cruise_speed * (times - start_time_s)
    coast_y = np.full_like(coast_x, y0)
    return times, coast_x, coast_y

# -----------------------------
# Plotting utilities (improved spatial + radial envelope)
# -----------------------------
def plot_spatial_improved(all_runs, df_mean, title, outpath=None,
                          n_samples=N_SAMPLE_PLOTS, clip_low=CLIP_LOW, clip_high=CLIP_HIGH,
                          show_density=SHOW_DENSITY):
    all_x = all_runs[:, :, 0].ravel()
    all_y = all_runs[:, :, 1].ravel()
    xmin, xmax = np.percentile(all_x, [clip_low, clip_high])
    ymin, ymax = np.percentile(all_y, [clip_low, clip_high])
    x_margin = (xmax - xmin) * 0.08
    y_margin = (ymax - ymin) * 0.08
    xlim = (xmin - x_margin, xmax + x_margin)
    ylim = (ymin - y_margin, ymax + y_margin)

    fig, ax = plt.subplots(figsize=(9, 6))

    if show_density:
        ax.hexbin(all_x, all_y, gridsize=160, bins='log', alpha=0.5)

    n_runs = all_runs.shape[0]
    n_plot = min(n_samples, n_runs)
    idxs = np.linspace(0, n_runs-1, n_plot, dtype=int)
    for i in idxs:
        ax.plot(all_runs[i, :, 0], all_runs[i, :, 1], linewidth=0.7, alpha=0.25, color='gray')

    ax.plot(df_mean['x'], df_mean['y'], linewidth=3.0, color='tab:red', label='Ensemble mean')

    finals = all_runs[:, -1, :2]
    final_r = np.sqrt(finals[:, 0]**2 + finals[:, 1]**2)
    sc = ax.scatter(finals[:, 0], finals[:, 1], c=final_r/AU, cmap='viridis', s=20, edgecolor='k', linewidth=0.2, zorder=6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Final radius (AU)')

    sx, sy = all_runs[0, 0, 0], all_runs[0, 0, 1]
    ax.scatter([sx], [sy], marker='*', color='white', edgecolor='black', s=90, zorder=7, label='Start')

    # show ticks in AU
    xticks = np.linspace(xlim[0], xlim[1], 5)
    yticks = np.linspace(ylim[0], ylim[1], 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{t/AU:.3f}" for t in xticks])
    ax.set_yticklabels([f"{t/AU:.3f}" for t in yticks])
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_title(title)
    ax.legend(loc='upper left')

    # inset zoom
    axins = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=1)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    inset_half_x = max(0.02 * x_span, 0.02 * AU)
    inset_half_y = max(0.02 * y_span, 0.02 * AU)
    ixlim = (sx - inset_half_x, sx + inset_half_x)
    iylim = (sy - inset_half_y, sy + inset_half_y)
    if show_density:
        axins.hexbin(all_x, all_y, gridsize=80, bins='log', alpha=0.5)
    for i in idxs:
        axins.plot(all_runs[i, :, 0], all_runs[i, :, 1], linewidth=0.7, alpha=0.25, color='gray')
    axins.plot(df_mean['x'], df_mean['y'], linewidth=2.0, color='tab:red')
    axins.scatter([sx], [sy], marker='*', color='white', edgecolor='black', s=60)
    axins.set_xlim(ixlim)
    axins.set_ylim(iylim)
    xti = np.linspace(ixlim[0], ixlim[1], 3)
    yti = np.linspace(iylim[0], iylim[1], 3)
    axins.set_xticks(xti)
    axins.set_yticks(yti)
    axins.set_xticklabels([f"{t/AU:.4f}" for t in xti], fontsize=6)
    axins.set_yticklabels([f"{t/AU:.4f}" for t in yti], fontsize=6)
    axins.set_title("Local zoom", fontsize=8)
    axins.grid(True)

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_radial(df_pct, title, outpath=None):
    plt.figure(figsize=(9,5))
    times_days = df_pct['time_s'] / (24*3600)
    plt.plot(times_days, df_pct['r_mean'] / AU, linewidth=2, label='Mean radial (AU)')
    plt.fill_between(times_days, df_pct['r_low']/AU, df_pct['r_high']/AU, alpha=0.25, label=f'{PCT_LOW}-{PCT_HIGH}% corridor')
    plt.xlabel('Time (days)')
    plt.ylabel('Radial distance (AU)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_mission(df_mean_A, df_mean_C, coast_times, coast_x, coast_y):
    # combine radial distances on a mission time axis and show log scale (good for full-scale)
    radial_A = np.sqrt(df_mean_A['x']**2 + df_mean_A['y']**2) / AU
    times_A_yr = df_mean_A['time_s'] / YEAR_SECONDS
    # coast times: provided relative to mission start (s)
    times_coast_yr = coast_times / YEAR_SECONDS
    radial_coast = np.sqrt(coast_x**2 + coast_y**2) / AU
    # Phase C times should be offset after coast
    times_C_yr = times_coast_yr[-1] + df_mean_C['time_s'] / YEAR_SECONDS
    radial_C = np.sqrt(df_mean_C['x']**2 + df_mean_C['y']**2) / AU

    plt.figure(figsize=(11,5))
    plt.plot(times_A_yr, radial_A, label='Phase A mean (AU)')
    plt.plot(times_coast_yr, radial_coast, label=f'Phase B coast @ {CRUISE_SPEED:.2e} m/s')
    plt.plot(times_C_yr, radial_C, label='Phase C mean (AU)')
    plt.yscale('log')
    plt.xlabel('Mission time (years)')
    plt.ylabel('Radial distance (AU) [log scale]')
    plt.title('Combined mission-scale radial distance (log scale)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "combined_mission_radial_log.png", dpi=300)
    plt.show()

# -----------------------------
# Utility: capture probability
# -----------------------------
def compute_capture_probability(all_runs, capture_radius_m):
    final_r = np.sqrt(all_runs[:, -1, 0]**2 + all_runs[:, -1, 1]**2)
    captured = np.sum(final_r <= capture_radius_m)
    return captured / all_runs.shape[0], final_r

# -----------------------------
# MAIN: run everything with longer phases
# -----------------------------
if __name__ == "__main__":
    # Phase A: Solar System Escape (longer)
    init_A = [1.0*AU, 0.0, 0.0, 42100.0]   # start at 1 AU, vy = escape as earlier
    print(f"Running Phase A (split integrator) for {PHASE_A_YEARS} years with dt={DT_DAYS_PHASE_A} days...")
    times_A, all_A, df_mean_A, df_pct_A = simulate_phase_split(
        init_A, PHASE_A_YEARS, SUN_MASS, NOISE_ESCAPE, num_runs=NUM_RUNS, dt_days=DT_DAYS_PHASE_A)
    print("Phase A done. Saving outputs...")
    df_mean_A.to_csv(OUTDIR / "phaseA_mean_long.csv", index=False)
    df_pct_A.to_csv(OUTDIR / "phaseA_percentiles_long.csv", index=False)

    plot_spatial_improved(all_A, df_mean_A, "Phase A — Solar System Escape (long, split integrator)",
                          outpath=OUTDIR / "phaseA_spatial_long.png")
    plot_radial(df_pct_A, "Phase A — Radial envelope (long)", outpath=OUTDIR / "phaseA_radial_long.png")

    # Compute a quick capture probability example for Phase A final positions within 0.1 AU
    capture_probA, finalsA = compute_capture_probability(all_A, 0.1*AU)
    print(f"Phase A example capture probability within 0.1 AU: {capture_probA:.4f}")

    # Phase B: deterministic coast (stitch in mission-scale)
    if USE_PHASE_B:
        coast_years = 1250.0 * (0.01 / (CRUISE_SPEED / 3.0e6))  # rough scaling: default maps 0.01c -> 1250 years
        # For clarity, derive coast_years directly from distance to Teegarden's star (approx 12.5 ly).
        # distance ~ 12.5 ly in meters:
        DISTANCE_LY = 12.5
        LY_M = 9.4607e15
        distance_m = DISTANCE_LY * LY_M
        coast_years = distance_m / CRUISE_SPEED / YEAR_SECONDS
        print(f"Using Phase B coast approximated travel time {coast_years:.1f} years at cruise speed {CRUISE_SPEED:.2e} m/s.")
        # start coast at Phase A mean endpoint
        start_pos_x = df_mean_A['x'].iloc[-1]
        start_pos_y = df_mean_A['y'].iloc[-1]
        start_time_s = df_mean_A['time_s'].iloc[-1]
        coast_times, coast_x, coast_y = deterministic_coast((start_pos_x, start_pos_y), start_time_s, CRUISE_SPEED, coast_years, step_days=COAST_SAMPLE_STEP_DAYS)
        # Save a small coast CSV
        pd.DataFrame({"time_s": coast_times, "x": coast_x, "y": coast_y}).to_csv(OUTDIR / "phaseB_coast.csv", index=False)
    else:
        # if not using coast, create empty arrays to avoid errors later
        coast_times, coast_x, coast_y = np.array([]), np.array([]), np.array([])

    # Phase C: Arrival & Capture (longer)
    init_C = [0.1*AU, 0.0, 0.0, 8000.0]    # as before but run longer
    print(f"Running Phase C (split integrator) for {PHASE_C_YEARS} years with dt={DT_DAYS_PHASE_C} days...")
    times_C, all_C, df_mean_C, df_pct_C = simulate_phase_split(
        init_C, PHASE_C_YEARS, TEEGARDEN_MASS, NOISE_ARRIVAL, num_runs=NUM_RUNS, dt_days=DT_DAYS_PHASE_C)
    df_mean_C.to_csv(OUTDIR / "phaseC_mean_long.csv", index=False)
    df_pct_C.to_csv(OUTDIR / "phaseC_percentiles_long.csv", index=False)

    plot_spatial_improved(all_C, df_mean_C, "Phase C — Arrival & Capture (long, split integrator)",
                          outpath=OUTDIR / "phaseC_spatial_long.png")
    plot_radial(df_pct_C, "Phase C — Radial envelope (long)", outpath=OUTDIR / "phaseC_radial_long.png")

    capture_probC, finalsC = compute_capture_probability(all_C, 0.02*AU)
    print(f"Phase C example capture probability within 0.02 AU: {capture_probC:.4f}")

    # Combined mission-scale radial plot
    if USE_PHASE_B and coast_times.size > 0:
        plot_combined_mission(df_mean_A, df_mean_C, coast_times, coast_x, coast_y)
    else:
        print("Phase B coast disabled or empty; combined mission plot skipped.")

    # Save flattened ensemble CSVs (careful: large files)
    print("Saving flattened ensemble CSVs (can be large)...")
    flatA = all_A.reshape((all_A.shape[0] * all_A.shape[1], 4))
    pd.DataFrame(flatA, columns=['x','y','vx','vy']).to_csv(OUTDIR / "phaseA_all_runs_flat_long.csv", index=False)
    flatC = all_C.reshape((all_C.shape[0] * all_C.shape[1], 4))
    pd.DataFrame(flatC, columns=['x','y','vx','vy']).to_csv(OUTDIR / "phaseC_all_runs_flat_long.csv", index=False)

    # Summary
    summary = {
        "phase": ["A", "C"],
        "num_runs": [NUM_RUNS, NUM_RUNS],
        "t_end_years": [PHASE_A_YEARS, PHASE_C_YEARS],
        "capture_prob_example": [capture_probA, capture_probC]
    }
    pd.DataFrame(summary).to_csv(OUTDIR / "summary_long_runs.csv", index=False)

    print(f"All done. Outputs saved to {OUTDIR.resolve()}")
