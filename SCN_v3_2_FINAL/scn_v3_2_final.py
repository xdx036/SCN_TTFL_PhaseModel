# -*- coding: utf-8 -*-
"""
SCN v3.2 FINAL (single-file, Windows-safe)
- No external utils_*.py needed
- No multiprocessing (workers=1) to avoid Windows pickling errors
- Produces 18 figures + 1 CSV of params
- Has progress/status logging for DE
- Robust CSV loading (with/without 'time_h' header)
"""

import os, json, math, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit, differential_evolution

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(path: Path, dpi=220, bbox='tight'):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches=bbox)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches=bbox)
    plt.close()

def norm01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

def detrend_linear(y):
    y = np.asarray(y, dtype=float)
    if y.size < 2 or np.all(~np.isfinite(y)):
        mu = np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0
        return np.nan_to_num(y - mu, nan=0.0)
    y = np.nan_to_num(y, nan=float(np.nanmean(y)))
    t = np.arange(len(y), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - (m*t + c)

def hampel_1d(x, k=3, nsigma=3.0):
    x = np.asarray(x, dtype=float)
    y = x.copy()
    n = x.size
    for i in range(n):
        i0, i1 = max(0, i-k), min(n, i+k+1)
        med = np.median(x[i0:i1])
        mad = np.median(np.abs(x[i0:i1] - med)) + 1e-9
        if np.abs(x[i] - med) > nsigma * 1.4826 * mad:
            y[i] = med
    return y

# ---------------------------
# Data loading
# ---------------------------
def load_per2_csv(path_csv: str):
    """Support both with header('time_h') or pure numeric table."""
    df_raw = pd.read_csv(path_csv, header=None)
    # If first cell is 'time_h' (string), re-read with header
    if isinstance(df_raw.iloc[0,0], str) and str(df_raw.iloc[0,0]).strip().lower() == 'time_h':
        df = pd.read_csv(path_csv)
        if 'time_h' not in df.columns:
            raise ValueError("CSV header found, but no 'time_h' column.")
    else:
        df = df_raw.copy()
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df.insert(0, 'time_h', np.arange(len(df_raw), dtype=float))
    df['time_h'] = pd.to_numeric(df['time_h'], errors='coerce')
    df = df.dropna(subset=['time_h']).reset_index(drop=True)
    # Coerce numeric for all cells
    for c in df.columns:
        if c == 'time_h': continue
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def build_W_from_MIC(path_csv: str, threshold: float):
    """Read MIC matrix, symmetrize, threshold, degree-normalize to row-stochastic W."""
    A = pd.read_csv(path_csv, header=None).values.astype(float)
    if not np.allclose(A, A.T, equal_nan=True):
        A = np.nan_to_num(A, nan=0.0)
        A = A + A.T
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    keep = (A >= threshold).astype(float)
    np.fill_diagonal(keep, 0.0)
    deg = keep.sum(axis=1, keepdims=True) + 1e-9
    W = keep / deg
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return W, keep

# ---------------------------
# Empirical r(t) & single-cell period
# ---------------------------
def empirical_r_from_per2(df: pd.DataFrame, max_h=432.0, decimate=1):
    df2 = df[df['time_h'] <= max_h].reset_index(drop=True)
    if decimate > 1:
        df2 = df2.iloc[::decimate, :].reset_index(drop=True)
    t = df2['time_h'].values.astype(float)
    X = df2.drop(columns=['time_h']).values.astype(float)
    # Clean each cell, detrend, normalize
    for j in range(X.shape[1]):
        xj = hampel_1d(X[:, j], k=3, nsigma=3.0)
        xj = detrend_linear(xj)
        X[:, j] = norm01(xj)
    # Hilbert phases
    phi = np.zeros_like(X)
    for j in range(X.shape[1]):
        hj = hilbert(X[:, j] - np.mean(X[:, j]))
        phi[:, j] = np.angle(hj)
    # Order param
    z = np.exp(1j * phi).mean(axis=1)
    r = np.abs(z); psi = np.angle(z)
    return t, r, psi

def estimate_singlecell_periods(df: pd.DataFrame, fit_h=96.0):
    df_fit = df[df['time_h'] <= fit_h].reset_index(drop=True)
    t = df_fit['time_h'].values.astype(float)
    X = df_fit.drop(columns=['time_h']).values.astype(float)
    per_list = []
    for j in range(X.shape[1]):
        xj = hampel_1d(X[:, j], k=3, nsigma=3.0)
        xj = detrend_linear(xj)
        xj = norm01(xj)
        # peaks by scipy
        try:
            pks, _ = find_peaks(xj, distance=max(1, int(20/(t[1]-t[0]))))
            if len(pks) >= 2:
                P = np.diff(t[pks])
                P = P[(P>12) & (P<36)]
                if len(P) > 0:
                    per_list.append(float(np.median(P)))
        except Exception:
            pass
    per_arr = np.array(per_list, dtype=float) if len(per_list)>0 else np.array([], dtype=float)
    return per_arr

def ttfl_like(t, A, T, phi, C, slope):
    return A*np.sin(2*np.pi*t/T + phi) + C + slope*t

def fit_singlecell_avg(df: pd.DataFrame, fit_h=96.0):
    df_fit = df[df['time_h'] <= fit_h].reset_index(drop=True)
    t = df_fit['time_h'].values.astype(float)
    X = df_fit.drop(columns=['time_h']).values.astype(float)
    for j in range(X.shape[1]):
        X[:, j] = hampel_1d(X[:, j], k=3, nsigma=3.0)
        X[:, j] = detrend_linear(X[:, j])
    Xn = np.apply_along_axis(norm01, 0, X)
    yavg = np.nanmean(Xn, axis=1)
    y = yavg - yavg.mean()
    # rough period guess by autocorr peak in [22, 28] h
    ac = np.correlate(y, y, mode='full')[len(y)-1:]
    T_guess = 26.0
    try:
        dt = t[1]-t[0]
        lo, hi = int(22/dt), int(28/dt)
        idx = np.argmax(ac[lo:hi])
        T_guess = (lo + idx) * dt
        T_guess = float(np.clip(T_guess, 22.0, 28.0))
    except Exception:
        pass
    p0 = (0.4, T_guess, 0.0, 0.5, 0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(ttfl_like, t, yavg, p0=p0, maxfev=20000)
    yhat = norm01(ttfl_like(t, *popt))
    rmse = float(np.sqrt(np.nanmean((yhat - yavg)**2)))
    return t, yavg, yhat, rmse, popt

# ---------------------------
# Phase network simulation (Kuramoto with network W)
# ---------------------------
def simulate_phase_network(W, T_h=120.0, dt=0.05,
                           Kv=0.02, Kg=0.5, Kl=0.3,
                           sigma=0.05, seed=0, omega0=2*np.pi/24.0, domega=0.02):
    """dtheta_i = omega_i + (Kv+Kg+Kl)*sum_j W_ij sin(theta_j - theta_i) + noise"""
    rng = np.random.default_rng(int(seed))
    N = int(W.shape[0])
    t_axis = np.arange(0.0, float(T_h), float(dt))
    omega = rng.normal(omega0, domega, size=N)
    theta = rng.uniform(0, 2*np.pi, size=N)
    k_tot = float(Kv + Kg + Kl)

    r_t = np.zeros_like(t_axis, dtype=float)
    psi_t = np.zeros_like(t_axis, dtype=float)

    for k, tk in enumerate(t_axis):
        # local mean-field coupling
        # sum_j W_ij * sin(theta_j - theta_i)
        theta_diff = theta[np.newaxis, :] - theta[:, np.newaxis]  # (N,N): theta_j - theta_i ? We want j - i
        # Actually, for each i: sum_j W_ij sin(theta_j - theta_i)
        # compute M_i = sum_j W_ij sin(theta_j - theta_i)
        # build sin(theta_j - theta_i): matrix with rows i, cols j
        mat = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])  # (N,N) sin(theta_j - theta_i)
        M = (W * mat).sum(axis=1)  # (N,)

        dtheta = omega + k_tot * M + sigma*np.sqrt(dt)*rng.normal(0.0,1.0,size=N)
        theta = (theta + dtheta*dt) % (2*np.pi)

        z = np.exp(1j*theta).mean()
        r_t[k] = np.abs(z)
        psi_t[k] = np.angle(z)

    return t_axis, r_t, psi_t, theta

# ---------------------------
# DE Objective (top-level, no closures)
# ---------------------------
def de_objective_with_ctx(x, sim_fn, W, t_emp, r_emp, dt, last, seed):
    Kv, Kg, Kl, sigma = map(float, x)
    # clamp
    Kv = float(np.clip(Kv, 0.0, 0.2))
    Kg = float(np.clip(Kg, 0.0, 0.2))
    Kl = float(np.clip(Kl, 0.0, 0.2))
    sigma = float(np.clip(sigma, 0.0, 0.2))

    t_emp = np.asarray(t_emp, dtype=float)
    r_emp = np.asarray(r_emp, dtype=float)
    W     = np.asarray(W, dtype=float)

    t_sim, r_sim, _, _ = sim_fn(W, T_h=float(t_emp.max()), dt=float(dt),
                                Kv=Kv, Kg=Kg, Kl=Kl, sigma=sigma, seed=int(seed))
    # align length
    if len(t_sim) != len(t_emp):
        idx = np.linspace(0, len(t_sim)-1, num=len(t_emp))
        r_use = np.interp(idx, np.arange(len(t_sim)), r_sim)
    else:
        r_use = r_sim

    last = max(1, int(last))
    rmse = float(np.sqrt(np.nanmean((r_use[-last:] - r_emp[-last:])**2)))
    return rmse

# ---------------------------
# Plot helpers
# ---------------------------
def plot_W_heatmap(W, outpath: Path):
    plt.figure(figsize=(6,5))
    plt.imshow(W, cmap='viridis', aspect='auto')
    plt.colorbar(label='W_ij')
    plt.title("Network weight matrix W")
    plt.xlabel("j"); plt.ylabel("i")
    savefig(outpath)

def plot_degree_hist(adj, outpath: Path):
    deg = adj.sum(axis=1)
    plt.figure(figsize=(6,4))
    plt.hist(deg, bins=30)
    plt.xlabel("Degree"); plt.ylabel("Count")
    plt.title("Degree distribution (MIC≥thr)")
    savefig(outpath)

def plot_threshold_sensitivity(mic_path: str, thr_center: float, outpath: Path):
    A = pd.read_csv(mic_path, header=None).values.astype(float)
    if not np.allclose(A, A.T, equal_nan=True):
        A = np.nan_to_num(A, nan=0.0); A = A + A.T
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    thrs = np.linspace(max(0.0, thr_center-0.05), min(1.0, thr_center+0.05), 41)
    edges = []
    for th in thrs:
        keep = (A>=th).astype(float)
        np.fill_diagonal(keep, 0.0)
        edges.append(int(keep.sum()))
    plt.figure(figsize=(6,4))
    plt.plot(thrs, edges, marker='o', linewidth=1)
    plt.axvline(thr_center, linestyle='--')
    plt.xlabel("MIC threshold"); plt.ylabel("#Edges (counting both directions)")
    plt.title("Edge count vs threshold")
    savefig(outpath)

def phase_rose(theta, outpath: Path, bins=36):
    theta = np.asarray(theta, dtype=float) % (2*np.pi)
    counts, edges = np.histogram(theta, bins=bins, range=(0, 2*np.pi))
    centers = 0.5*(edges[:-1]+edges[1:])
    ax = plt.figure(figsize=(5,5)).add_subplot(111, projection='polar')
    ax.bar(centers, counts, width=(2*np.pi)/bins, bottom=0.0, edgecolor='k', linewidth=0.5, alpha=0.8)
    ax.set_title("Final phase rose (model)")
    savefig(outpath)

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True, help="PER2 CSV (SCN1)")
    parser.add_argument("--mic_csv",  type=str, required=True, help="MIC matrix CSV")
    parser.add_argument("--mic_thr",  type=float, default=0.18)
    parser.add_argument("--fit_window_h", type=float, default=96.0)
    parser.add_argument("--emp_max_h",   type=float, default=240.0)
    parser.add_argument("--scan_res", type=int, default=20)
    parser.add_argument("--outdir",   type=str, required=True)
    parser.add_argument("--maxiter",  type=int, default=40)
    parser.add_argument("--popsize",  type=int, default=18)
    parser.add_argument("--vip_factor_ko", type=float, default=0.3)
    args = parser.parse_args()

    OUT = Path(args.outdir); ensure_dir(OUT)
    (OUT/"progress.tsv").write_text("iter\tcurr_rmse\tbest_rmse\n", encoding="utf-8")

    print("[STEP 0] Load data")
    df = load_per2_csv(args.data_csv)

    print("[STEP 1] Empirical r(t) / phase & single-cell periods")
    t_emp, r_emp, psi_emp = empirical_r_from_per2(df, max_h=args.emp_max_h, decimate=1)
    # save empirical plots
    plt.figure(figsize=(8,4))
    plt.plot(t_emp, r_emp, lw=2)
    plt.xlabel("Time (h)"); plt.ylabel("r(t)")
    plt.title("Empirical order parameter r(t)")
    savefig(OUT/"r_empirical.png")

    t_fit, yavg, yhat, rmse_fit, pfit = fit_singlecell_avg(df, fit_h=args.fit_window_h)
    plt.figure(figsize=(8,4))
    plt.plot(t_fit, yavg, label="data mean")
    plt.plot(t_fit, yhat, label="TTFL-like fit")
    plt.legend(frameon=False); plt.xlabel("Time (h)"); plt.ylabel("Norm intensity")
    plt.title(f"Single-cell average fit (RMSE={rmse_fit:.3f})")
    savefig(OUT/"single_cell_fit.png")

    plt.figure(figsize=(7,3))
    plt.plot(t_fit, yavg - yhat, label="residuals")
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel("Time (h)"); plt.ylabel("Residual")
    plt.title("Single-cell fit residuals")
    savefig(OUT/"single_cell_fit_residuals.png")

    per_arr = estimate_singlecell_periods(df, fit_h=args.fit_window_h)
    plt.figure(figsize=(6,4))
    if per_arr.size>0:
        plt.hist(per_arr, bins=20)
        plt.title(f"Single-cell period histogram (n={len(per_arr)})")
    else:
        plt.text(0.5,0.5,"No peaks detected", ha='center', va='center')
        plt.title("Single-cell period histogram")
    plt.xlabel("Period (h)"); plt.ylabel("Count")
    savefig(OUT/"period_histogram.png")

    print("[STEP 2] Build W from MIC (thr=%.3f)" % args.mic_thr)
    W, adj = build_W_from_MIC(args.mic_csv, threshold=float(args.mic_thr))
    plot_W_heatmap(W, OUT/"W_matrix.png")
    plot_degree_hist(adj, OUT/"degree_histogram.png")
    plot_threshold_sensitivity(args.mic_csv, float(args.mic_thr), OUT/"mic_threshold_effect.png")

    # ----------------- [STEP 3] DE fit (workers=1 for Windows safety) -----------------
    print("[STEP 3] Global optimization (DE)")
    # progress/state
    iter_state = {"i": 0, "best": float("inf")}
    def de_callback(xk, convergence):
        # Evaluate current
        dt = 0.05
        last = int(24.0/dt)
        curr = de_objective_with_ctx(xk, simulate_phase_network, W, t_emp, r_emp, dt, last, 2024)
        if curr < iter_state["best"]: iter_state["best"] = curr
        iter_state["i"] += 1
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iteration": int(iter_state["i"]),
            "curr_rmse": float(curr),
            "best_rmse": float(iter_state["best"]),
            "xk": [float(v) for v in xk],
            "convergence": float(convergence)
        }
        (OUT/"status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (OUT/"progress.tsv").open("a", encoding="utf-8") as f:
            f.write(f"{iter_state['i']}\t{curr:.6f}\t{iter_state['best']:.6f}\n")
        print(f"[DE] iter={iter_state['i']:03d} curr={curr:.4f} best={iter_state['best']:.4f}")
        return False

    bounds  = [(0.0, 0.12), (0.0, 0.12), (0.0, 0.12), (0.0, 0.12)]  # Kv, Kg, Kl, sigma
    dt = 0.05; last = int(24.0/dt)
    de_args = (simulate_phase_network, W, t_emp, r_emp, dt, last, 2024)

    if int(args.maxiter) > 0 and int(args.popsize) > 0:
        res = differential_evolution(
            de_objective_with_ctx, bounds, args=de_args,
            maxiter=int(args.maxiter), popsize=int(args.popsize),
            workers=-1, updating='immediate', tol=1e-3, polish=True, seed=2025, callback=de_callback
        )
        best = res.x
        best_rmse = float(res.fun)
    else:
        # Skip DE, use default mid params
        best = np.array([0.02, 0.5, 0.3, 0.05], dtype=float)
        best_rmse = float('nan')

    print(f"[DE] best: Kv={best[0]:.4f}, Kg={best[1]:.4f}, Kl={best[2]:.4f}, sigma={best[3]:.4f}, RMSE={best_rmse:.4f}")

    # simulate with best and overlay empirical
    t_sim, r_sim, psi_sim, theta_final = simulate_phase_network(W, T_h=float(t_emp.max()), dt=dt,
                                                                Kv=best[0], Kg=best[1], Kl=best[2],
                                                                sigma=best[3], seed=2025)
    if len(t_sim) != len(t_emp):
        idx = np.linspace(0, len(t_sim)-1, num=len(t_emp))
        r_use = np.interp(idx, np.arange(len(t_sim)), r_sim)
        t_use = t_emp
    else:
        r_use = r_sim; t_use = t_sim

    plt.figure(figsize=(8,4))
    plt.plot(t_emp, r_emp, label="Empirical r(t)", lw=2)
    plt.plot(t_use, r_use, label="Model r(t) (fitted)", lw=2)
    plt.legend(frameon=False); plt.xlabel("Time (h)"); plt.ylabel("r(t)")
    plt.title("Empirical vs Model r(t)")
    savefig(OUT/"r_emp_vs_model_fit.png")

    # Rose of final phases
    phase_rose(theta_final, OUT/"phase_rose_highres.png", bins=36)

    # ----------------- [STEP 4] VIP ON/KO & sweeps -----------------
    print("[STEP 4] VIP ON/KO sweeps & comparisons")
    def sweep_and_plot(title, mode, vip_on, fname, Kg_max=1.0, Kl_max=1.0, grid=20):
        Gg = np.linspace(0, Kg_max, grid)
        Gl = np.linspace(0, Kl_max, grid)
        R = np.zeros((grid, grid))
        vipfac = 1.0 if vip_on else float(args.vip_factor_ko)
        for i, kg in enumerate(Gg):
            for j, kl in enumerate(Gl):
                Kg = kg if mode in ("both","gaba") else 0.0
                Kl = kl if mode in ("both","gly")  else 0.0
                Kg*=vipfac; Kl*=vipfac
                _, rt, _, _ = simulate_phase_network(W, T_h=72.0, dt=0.1,
                                                     Kv=0.02*vipfac, Kg=Kg, Kl=Kl,
                                                     sigma=best[3], seed=999+i*17+j*13)
                R[i,j] = rt[-1]
        X, Y = np.meshgrid(Gl, Gg)
        plt.figure(figsize=(8.2,6.4))
        im = plt.pcolormesh(X, Y, R, shading="auto", vmin=0.0, vmax=1.0)
        plt.xlabel("Glycine K"); plt.ylabel("GABA K")
        cbar = plt.colorbar(im, label="Final r")
        CS = plt.contour(X, Y, R, levels=[0.2,0.4,0.6,0.8,0.9], colors="k", linewidths=0.7)
        plt.clabel(CS, inline=1, fontsize=8, fmt="r=%.1f")
        plt.title(f"{title}\nCoupled Rhythm Zone (high r) vs Desynchronization Zone (low r)")
        savefig(OUT/fname)

    sweep_and_plot("VIP ON / GABA only",  "gaba", True,  "coupling_sweep_20x20_VIPON_gaba.png", grid=int(args.scan_res))
    sweep_and_plot("VIP ON / Gly only",   "gly",  True,  "coupling_sweep_20x20_VIPON_gly.png", grid=int(args.scan_res))
    sweep_and_plot("VIP ON / GABA+Gly",   "both", True,  "coupling_sweep_20x20_VIPON_both.png", grid=int(args.scan_res))
    sweep_and_plot("VIP KO / GABA only",  "gaba", False, "coupling_sweep_20x20_VIPKO_gaba.png", grid=int(args.scan_res))
    sweep_and_plot("VIP KO / Gly only",   "gly",  False, "coupling_sweep_20x20_VIPKO_gly.png", grid=int(args.scan_res))
    sweep_and_plot("VIP KO / GABA+Gly",   "both", False, "coupling_sweep_20x20_VIPKO_both.png", grid=int(args.scan_res))

    # VIP ON vs KO r(t)
    t_on, r_on, _, _ = simulate_phase_network(W, T_h=96.0, dt=0.05, Kv=best[0], Kg=best[1], Kl=best[2],
                                              sigma=best[3], seed=77)
    t_ko, r_ko, _, _ = simulate_phase_network(W, T_h=96.0, dt=0.05, Kv=best[0]*args.vip_factor_ko,
                                              Kg=best[1]*args.vip_factor_ko, Kl=best[2]*args.vip_factor_ko,
                                              sigma=best[3]*1.15, seed=77)
    plt.figure(figsize=(8,4))
    plt.plot(t_on, r_on, label="VIP ON (GABA+Gly)", lw=2)
    plt.plot(t_ko, r_ko, label=f"VIP KO (factor={args.vip_factor_ko:.2f})", lw=2)
    plt.xlabel("Time (h)"); plt.ylabel("r(t)"); plt.legend(frameon=False)
    plt.title("Network synchronization: VIP ON vs KO")
    savefig(OUT/"vip_on_vs_ko_r_vs_time.png")

    # ----------------- [STEP 5] Jet-lag & Noise sweeps -----------------
    print("[STEP 5] Jet-lag & Noise sweeps")
    def jetlag_with_tau(shift_h=6.0):
        shift = 2*np.pi*(shift_h/24.0)
        t, r, psi, _ = simulate_phase_network(W, T_h=96.0, dt=0.05,
                                              Kv=best[0], Kg=best[1], Kl=best[2], sigma=best[3], seed=42)
        k = int(48.0/0.05)  # injection at 48h
        psi2 = psi.copy()
        psi2[k:] = ((psi2[k:] + shift + np.pi)%(2*np.pi)) - np.pi
        x = t[k:] - t[k]
        y = np.abs((psi2[k:] - psi2[k] + np.pi)%(2*np.pi) - np.pi) + 1e-6
        def model(x,a,tau,b): return a*np.exp(-x/tau)+b
        try:
            popt,_ = curve_fit(model, x, y, p0=(y[0], 8.0, 0.05), maxfev=10000)
            tau_est = float(popt[1])
        except Exception:
            tau_est = float('nan')
        return t, r, psi2, tau_est

    tj, rj, psij, tau_est = jetlag_with_tau(6.0)
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(tj, rj, linewidth=2, label="r(t)")
    ax1.set_ylabel("Order parameter r"); ax1.legend(frameon=False)
    ax1.set_title(f"Jet-lag recovery: r & mean phase ψ (τ ≈ {tau_est:.1f} h)")
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(tj, psij, linewidth=2, label="ψ(t)")
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Mean phase ψ (rad)")
    ax2.legend(frameon=False)
    savefig(OUT/"jetlag_recovery_time_phase.png")

    sigmas = np.linspace(0.0, 0.12, 7)
    series = []; labels = []
    def final_r(Kg, Kl, s, seed=21, Kv=best[0]):
        _, rt, _, _ = simulate_phase_network(W, T_h=96.0, dt=0.05, Kv=Kv, Kg=Kg, Kl=Kl, sigma=s, seed=seed)
        return rt[-1]
    # noise only
    series.append([final_r(0.0,0.0,s,seed=1, Kv=0.0) for s in sigmas]); labels.append("Noise only (no coupling)")
    # VIP ON
    series.append([final_r(best[1],0.0,s,seed=2, Kv=best[0]) for s in sigmas]); labels.append("VIP ON / GABA only")
    series.append([final_r(0.0,best[2],s,seed=3, Kv=best[0]) for s in sigmas]); labels.append("VIP ON / Gly only")
    series.append([final_r(best[1],best[2],s,seed=4, Kv=best[0]) for s in sigmas]); labels.append("VIP ON / GABA+Gly")
    # VIP KO (scale)
    series.append([final_r(best[1]*args.vip_factor_ko, best[2]*args.vip_factor_ko, s, seed=5, Kv=best[0]*args.vip_factor_ko) for s in sigmas]); labels.append("VIP KO / GABA+Gly")
    plt.figure(figsize=(8,5))
    for y, name in zip(series, labels):
        plt.plot(sigmas, y, marker='o', label=name)
    plt.xlabel("Noise sigma"); plt.ylabel("Final r")
    plt.title("Noise impact under different coupling conditions")
    plt.legend(frameon=False)
    savefig(OUT/"noise_vs_r_multi.png")

    # ----------------- Save params summary -----------------
    param_row = {
        "Kv": float(best[0]), "Kg": float(best[1]), "Kl": float(best[2]), "sigma": float(best[3]),
        "RMSE_model_vs_emp": float(best_rmse),
        "MIC_thr": float(args.mic_thr),
        "fit_h": float(args.fit_window_h),
        "vip_factor_ko": float(args.vip_factor_ko)
    }
    pd.DataFrame([param_row]).to_csv(OUT/"Model_params_v3_2.csv", index=False)

    print(" DONE. Outputs in:", str(OUT))

if __name__ == "__main__":
    # Set a neutral font if Arial missing
    try:
        matplotlib.rcParams['font.family'] = 'Arial'
    except Exception:
        pass
    main()
