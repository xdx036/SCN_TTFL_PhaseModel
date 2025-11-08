
# -*- coding: utf-8 -*-
"""
SCN v3 (fixed) — 数据约束 + 全局拟合 (DE, workers=-1) + 并行扫图 + 进度显示 + Windows spawn 兼容
"""
import os, json, time, pickle, warnings, pathlib, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Arial", "figure.dpi":120})

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, curve_fit, differential_evolution
from scipy.stats import gaussian_kde
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

import multiprocessing as mp

HERE = pathlib.Path(__file__).parent.resolve()
DATA_TS  = HERE/"scn1_full_data_with_time.csv"
DATA_MIC = HERE/"mic_indiv_scn1.csv"
MIC_TH   = 0.949
OUT = HERE/"outputs_v3"; OUT.mkdir(parents=True, exist_ok=True)

def log(s): print(s, flush=True)

def savefig(path_png, tight=True):
    if tight: plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    try:
        plt.savefig(path_png.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()

def hampel_1d(x, k=5, nsigma=3.0):
    x = x.astype(float).copy()
    n = len(x)
    for i in range(n):
        lo=max(0,i-k); hi=min(n,i+k+1)
        med = np.median(x[lo:hi])
        mad = np.median(np.abs(x[lo:hi]-med)) + 1e-9
        if abs(x[i]-med) > nsigma*1.4826*mad:
            x[i]=med
    return x

def detrend_linear(x):
    t=np.arange(len(x)); A=np.vstack([t, np.ones_like(t)]).T
    m,c=np.linalg.lstsq(A, x, rcond=None)[0]
    return x-(m*t+c)

def norm01(x):
    return (x - np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)+1e-9)

def estimate_period_autocorr(t, y):
    y = (y - np.nanmean(y))/(np.nanstd(y)+1e-9)
    ac = np.correlate(y, y, mode="full")[len(y)-1:]
    ac = ac / max(ac[0],1e-9)
    dt = t[1]-t[0]
    lo = max(2, int(16.0/dt)); hi = min(len(ac)-1, int(32.0/dt))
    if hi>lo:
        k = lo + int(np.argmax(ac[lo:hi]))
        return k*dt
    return 24.0

def load_timeseries_strict(csv_path: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, header=0, dtype=str)
    except Exception:
        df = pd.read_csv(csv_path, header=None, dtype=str)
    first_col = str(df.columns[0]).strip().lower()
    if first_col not in {"time_h","time","t","hour","hours"}:
        df0 = pd.read_csv(csv_path, header=None, dtype=str)
        header = df0.iloc[0].tolist()
        df = df0.iloc[1:].copy()
        df.columns = header
    tcol=None
    for c in df.columns:
        if str(c).strip().lower() in {"time_h","time","t","hour","hours"}:
            tcol=c; break
    if tcol is None:
        df.insert(0, "time_h", np.arange(len(df)))
        tcol="time_h"
    bad_strings = {"time_h","time","t","hour","hours"}
    df = df[~df[tcol].str.strip().str.lower().isin(bad_strings)].copy()
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    data_cols = [c for c in df.columns if c != tcol]
    for c in data_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[tcol])
    if data_cols:
        df = df.dropna(how="all", subset=data_cols)
    df = df.reset_index(drop=True).rename(columns={tcol:"time_h"})
    return df

def run_single_cell_block():
    df = load_timeseries_strict(DATA_TS)
    df_fit = df[df["time_h"] <= 96].iloc[::2, :].reset_index(drop=True)
    t_fit = pd.to_numeric(df_fit["time_h"], errors="coerce").to_numpy(dtype=float)
    X_df = df_fit.drop(columns=["time_h"]).apply(pd.to_numeric, errors="coerce")
    bad_rows = X_df.isna().all(axis=1) | pd.isna(t_fit)
    if bad_rows.any():
        X_df  = X_df.loc[~bad_rows].reset_index(drop=True)
        t_fit = t_fit[~bad_rows]
    X_df = X_df.interpolate(method="linear", axis=0, limit_direction="both")
    X_df = X_df.fillna(X_df.mean(axis=0))
    X = X_df.to_numpy(dtype=float)
    for j in range(X.shape[1]):
        X[:,j] = hampel_1d(X[:,j], k=3, nsigma=3.0)
        X[:,j] = detrend_linear(X[:,j])
    Xn = np.apply_along_axis(norm01, 0, X)
    yavg = np.nanmean(Xn, axis=1)
    T_est = estimate_period_autocorr(t_fit, yavg)
    def model_rhs(t, y, p):
        m,P,Ca,F = y
        b_m, alpha, d_m, b_p, d_p, k_in, k_out, tauF, K_i, n_i, k_FP, F0 = (
            p["b_m"], p["alpha"], p["d_m"], p["b_p"], p["d_p"],
            p["k_in"], p["k_out"], p["tauF"], p["K_i"], p["n_i"], p["k_FP"], p["F0"]
        )
        inhib = 1.0/(1.0 + (P/K_i)**n_i)
        dm  = b_m + alpha*Ca*inhib - d_m*m
        dP  = b_p*m - d_p*P
        dCa = k_in*F - k_out*Ca
        Finf= F0 + k_FP*inhib
        dF  = (Finf - F)/tauF
        return [dm,dP,dCa,dF]
    def simulate_single_cell(t_eval, y0, p, max_step_h=2.0, rtol=1e-6, atol=1e-8):
        sol = solve_ivp(lambda t,y:model_rhs(t,y,p), (float(t_eval[0]), float(t_eval[-1])),
                        y0=y0, t_eval=t_eval, method="RK45",
                        rtol=rtol, atol=atol, max_step=max_step_h)
        if not sol.success:
            Y = np.full((4, len(t_eval)), np.nan)
            return t_eval, Y
        return sol.t, sol.y
    p0 = dict(b_m=0.05, alpha=0.06, d_m=0.05, b_p=0.06, d_p=0.02,
              k_in=0.4, k_out=0.2, tauF=min(12.0, 0.5*T_est),
              K_i=1.0, n_i=3.0, k_FP=0.8, F0=0.5)
    y0 = [0.2, 0.2, 0.1, 0.6]
    lo = np.array([0,0,0,0,0,0,0,1,0.1,1,0,0], float)
    hi = np.array([1,2,1,2,1,2,1, max(3.0,0.5*T_est), 5,6,5,5], float)
    theta0 = np.array([p0[k] for k in p0.keys()], float)
    theta0 = np.minimum(np.maximum(theta0, lo+1e-6), hi-1e-6)
    def resid(theta):
        p = dict(zip(list(p0.keys()), theta))
        _, y = simulate_single_cell(t_fit, y0, p, max_step_h=2.0)
        P = y[1,:]
        Pn = norm01(P)
        r = Pn - yavg
        r = np.where(np.isfinite(r), r, 1e3)
        return r
    log("[SingleCell] fitting ...")
    res = least_squares(resid, theta0, bounds=(lo,hi), max_nfev=400, xtol=1e-6, ftol=1e-6)
    p_opt = dict(zip(list(p0.keys()), res.x))
    tt, ysim = simulate_single_cell(t_fit, y0, p_opt, max_step_h=2.0)
    P = ysim[1,:]; Pn = norm01(P)
    rmse = float(np.sqrt(np.nanmean((Pn-yavg)**2)))
    plt.figure(figsize=(8,4))
    plt.plot(t_fit, yavg,  label="PER2 data (avg)", linewidth=2)
    plt.plot(t_fit, Pn, label="Model PER2 (norm)", linestyle="--", linewidth=1.8)
    plt.xlabel("Time (h)"); plt.ylabel("Normalized PER2")
    plt.title(f"Single-cell fit  (RMSE={rmse:.3f}, period≈{T_est:.1f} h)")
    plt.legend(frameon=False)
    savefig(OUT/"single_cell_fit.png")
    plt.figure(figsize=(10,5))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t_fit, yavg, label="Data", linewidth=2)
    ax1.plot(t_fit, Pn, label="Model", linestyle="--", linewidth=1.8)
    ax1.set_ylabel("Normalized PER2"); ax1.set_title("Single-cell fit")
    ax1.legend(frameon=False)
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    resid_vec = Pn - yavg
    ax2.plot(t_fit, resid_vec, linewidth=1.6)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Residual")
    ax2.text(0.01, 0.9, f"RMSE={rmse:.3f}", transform=ax2.transAxes)
    savefig(OUT/"single_cell_fit_residuals.png")
    t_all = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
    Xall_df = df.drop(columns=["time_h"]).apply(pd.to_numeric, errors="coerce")
    bad_rows = Xall_df.isna().all(axis=1) | pd.isna(t_all)
    if bad_rows.any():
        Xall_df = Xall_df.loc[~bad_rows].reset_index(drop=True)
        t_all   = t_all[~bad_rows]
    Xall_df = Xall_df.interpolate(method="linear", axis=0, limit_direction="both")
    Xall_df = Xall_df.fillna(Xall_df.mean(axis=0))
    Xall = Xall_df.to_numpy(dtype=float)
    Xall = np.apply_along_axis(norm01, 0, Xall)
    def est_period_vec(t, X):
        out=[]
        for j in range(X.shape[1]):
            x = (X[:,j]-np.nanmean(X[:,j]))/(np.nanstd(X[:,j])+1e-9)
            ac = np.correlate(x,x,mode="full")[len(x)-1:]
            ac = ac / max(ac[0],1e-9)
            dt = t[1]-t[0]
            lo = max(2, int(16.0/dt)); hi = min(len(ac)-1, int(32.0/dt))
            if hi>lo:
                k = lo + int(np.argmax(ac[lo:hi]))
                out.append(k*dt)
            else:
                out.append(np.nan)
        return np.array(out,float)
    periods = est_period_vec(t_all, Xall)
    p = periods[np.isfinite(periods)]
    if p.size>0:
        bins = np.arange(np.nanmin(p), np.nanmax(p)+0.5, 0.5)
        mu, sd = float(np.nanmean(p)), float(np.nanstd(p))
        plt.figure(figsize=(8,5))
        plt.hist(p, bins=bins, edgecolor="white", alpha=0.7, density=True, label="Histogram")
        try:
            kde = gaussian_kde(p); xs = np.linspace(np.nanmin(p), np.nanmax(p), 300)
            plt.plot(xs, kde(xs), linewidth=2, label="KDE")
        except Exception:
            pass
        plt.axvline(mu, color="black", linestyle="--", linewidth=1, label=f"mean={mu:.2f} h")
        plt.axvspan(mu-sd, mu+sd, color="grey", alpha=0.2, label=f"±1 SD={sd:.2f} h")
        plt.xlabel("Estimated period (h)"); plt.ylabel("Density")
        plt.legend(frameon=False)
        savefig(OUT/"period_histogram.png")
    return T_est

def build_W_from_MIC():
    mic = pd.read_csv(DATA_MIC, header=None).values.astype(float)
    mic = 0.5*(mic + mic.T)
    np.fill_diagonal(mic, 0.0)
    A = (mic >= MIC_TH).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1, keepdims=True) + 1e-9
    W = A/deg
    stats = {"N": int(W.shape[0]), "edges": int(A.sum()),
             "density": float(A.sum()/(W.shape[0]*(W.shape[0]-1))),
             "threshold": float(MIC_TH)}
    with open(OUT/"W_stats.json","w") as f: json.dump(stats, f, indent=2)
    np.save(OUT/"W_scn1_mic_thresh.npy", W)
    log(f"[W] built: N={stats['N']} edges={stats['edges']} dens={stats['density']:.4f}")
    return W

def empirical_r_from_TS(smooth_sigma_h=1.0, detrend=True):
    df = load_timeseries_strict(DATA_TS)
    t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
    X_df = df.drop(columns=["time_h"]).apply(pd.to_numeric, errors="coerce")
    bad_rows = X_df.isna().all(axis=1) | pd.isna(t)
    if bad_rows.any():
        X_df = X_df.loc[~bad_rows].reset_index(drop=True)
        t    = t[~bad_rows]
    X_df = X_df.interpolate(method="linear", axis=0, limit_direction="both")
    X_df = X_df.fillna(X_df.mean(axis=0))
    X = X_df.to_numpy(dtype=float)
    if X.shape[0] != len(t):
        if X.shape[1] == len(t): X = X.T
        else: raise ValueError(f"Shape mismatch after cleaning: X{X.shape}, t{t.shape}")
    T, N = X.shape
    dt = np.median(np.diff(t))
    s  = max(1, int(smooth_sigma_h/dt))
    Xp = np.zeros_like(X)
    for j in range(N):
        x = gaussian_filter1d(X[:,j].astype(float), sigma=s, mode="nearest")
        if detrend:
            tt = np.arange(T); A = np.vstack([tt, np.ones(T)]).T
            m,c = np.linalg.lstsq(A, x, rcond=None)[0]
            x = x - (m*tt + c)
        x = (x - np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)+1e-9)
        Xp[:,j] = x
    theta = np.angle(hilbert(Xp, axis=0))
    z     = np.exp(1j*theta).mean(axis=1)
    r_emp = np.abs(z); psi_emp = np.angle(z)
    np.save(OUT/"t_empirical.npy", t)
    np.save(OUT/"r_empirical.npy", r_emp)
    np.save(OUT/"psi_empirical.npy", psi_emp)
    plt.figure(figsize=(9,5)); plt.plot(t, r_emp, lw=2)
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Empirical r(t) from SCN1 (Hilbert phase)")
    savefig(OUT/"r_empirical.png")
    return t, r_emp, psi_emp

def simulate_r_with_W(W, T_h, dt, K_gaba=0.30, K_gly=0.15, phi_g=np.pi, sigma=0.08, domega=0.06, vip_factor=1.0, seed=11):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    t = np.arange(0.0, T_h, dt)
    omega0 = 2*np.pi/24.0
    omega  = rng.normal(omega0, domega, size=N)
    theta  = rng.uniform(0, 2*np.pi, size=N)
    r_tr   = np.zeros_like(t); psi_tr = np.zeros_like(t)
    Kg = vip_factor*K_gaba; Kl = vip_factor*K_gly
    for k,_ in enumerate(t):
        z = np.exp(1j*theta).mean()
        r_tr[k] = np.abs(z); psi_tr[k] = np.angle(z)
        X  = np.exp(1j*theta)
        H  = W.dot(X)
        phi = np.angle(H) - theta
        coup = (Kg)*np.sin(phi + phi_g) + (Kl)*np.sin(phi)
        dth = omega + coup + sigma*np.sqrt(dt)*rng.normal(0,1,size=N)
        theta = (theta + dth*dt) % (2*np.pi)
    return t, r_tr, psi_tr, theta

def objective_empirical(p, W, t_emp, r_emp, dt, T_h, last_avg_h):
    K_gaba, K_gly, sigma, domega, phi_g = p
    t1, r1, _, _ = simulate_r_with_W(W, T_h, dt, K_gaba, K_gly, phi_g, sigma, domega, vip_factor=1.0, seed=7)
    r1i = np.interp(t_emp, t1, r1)
    mask = (t_emp >= (t_emp.max() - last_avg_h))
    return 0.7*np.mean((r1i[mask]-r_emp[mask])**2) + 0.3*np.mean((r1i-r_emp)**2)

def fit_params_to_empirical(W, t_emp, r_emp, last_avg_h=24.0, maxiter=60, popsize=18, seed=1234):
    dt = float(np.median(np.diff(t_emp))); T_h = float(t_emp.max())
    bounds = [(0,0.6),(0,0.4),(0.04,0.15),(0.03,0.10),(0,np.pi)]
    log("[FIT] start global optimization (DE, multi-core) ...")
    try:
        res = differential_evolution(
            objective_empirical, bounds,
            args=(W, t_emp, r_emp, dt, T_h, last_avg_h),
            maxiter=maxiter, popsize=popsize,
            tol=1e-3, polish=True, seed=seed,
            workers=-1, updating='deferred'
        )
    except TypeError:
        res = differential_evolution(
            objective_empirical, bounds,
            args=(W, t_emp, r_emp, dt, T_h, last_avg_h),
            maxiter=maxiter, popsize=popsize,
            tol=1e-3, polish=True, seed=seed
        )
    pbest = res.x
    np.save(OUT/"fit_params.npy", pbest)
    log(f"[FIT] done. loss={res.fun:.4g} params={pbest}")
    t1, r1, _, _ = simulate_r_with_W(W, T_h, dt, pbest[0], pbest[1], pbest[4], pbest[2], pbest[3], vip_factor=1.0, seed=999)
    r1i = np.interp(t_emp, t1, r1)
    plt.figure(figsize=(9,5))
    plt.plot(t_emp, r_emp, lw=2, label="Empirical r(t)")
    plt.plot(t_emp, r1i,  "--", lw=2, label="Fitted model r(t)")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r"); plt.legend(frameon=False)
    plt.title("Data-constrained network fit (SCN1)")
    savefig(OUT/"r_emp_vs_model_fit.png")
    return pbest

def _simulate_point(args):
    (W, Kg, Kl, phig, sigma, domega, vip_factor, T_h, dt, seed) = args
    _, r, *_ = simulate_r_with_W(W, T_h, dt, Kg, Kl, phig, sigma, domega, vip_factor=vip_factor, seed=seed)
    return float(np.mean(r[-int(24.0/dt):]))

def sweep_panel_W_parallel(W, params, title, mode, vip_on, fname, grid_n=20, T_h=96.0, dt=0.08):
    Kg0, Kl0, s, dw, phig = params
    vf = 1.0 if vip_on else 0.0
    sig = s if vip_on else s*1.15
    Gg = np.linspace(0.0, 1.0, grid_n)
    Gl = np.linspace(0.0, 1.0, grid_n)
    R  = np.zeros((grid_n, grid_n), float)
    tmpfile = OUT/f"__tmp_{fname}.pkl"
    done = set(); idx_map = {}
    if tmpfile.exists():
        try:
            with open(tmpfile,"rb") as f: obj=pickle.load(f)
            R[:,:] = obj.get("R", R); done = set(map(tuple, obj.get("done", [])))
            log(f"[SWEEP] resume from checkpoint: {fname}")
        except Exception:
            pass
    tasks = []
    for i, kg in enumerate(Gg):
        for j, kl in enumerate(Gl):
            if (i,j) in done: 
                continue
            Kg = (kg if mode in ("both","gaba") else 0.0) * Kg0 * vf / max(Kg0,1e-9)
            Kl = (kl if mode in ("both","gly")  else 0.0) * Kl0 * vf / max(Kl0,1e-9)
            tasks.append((W, Kg, Kl, phig, sig, dw, 1.0, T_h, dt, 777+i*13+j*7))
            idx_map[len(tasks)-1] = (i,j)
    total = len(tasks)
    if total == 0:
        X, Y = np.meshgrid(Gl, Gg)
        plt.figure(figsize=(8.2,6.4))
        im = plt.pcolormesh(X, Y, R, shading="auto", vmin=0.0, vmax=1.0)
        plt.xlabel("Glycine K"); plt.ylabel("GABA K")
        cb = plt.colorbar(im, label="Final r (mean over last 24 h)")
        CS = plt.contour(X, Y, R, levels=[0.2,0.4,0.6,0.8,0.9], colors="k", linewidths=0.7)
        plt.clabel(CS, inline=1, fontsize=8, fmt="r=%.1f")
        plt.title(title + "  (data-constrained W)")
        savefig(OUT/fname)
        return
    procs = max(1, os.cpu_count() or 1)
    log(f"[SWEEP] {title}: {grid_n}x{grid_n} points, remaining={total}, procs={procs}")
    done_count = 0
    last_print = time.time()
    with mp.get_context("spawn").Pool(processes=procs) as pool:
        for k, val in enumerate(pool.imap_unordered(_simulate_point, tasks, chunksize=4)):
            i,j = idx_map[k]
            R[i,j] = val
            done.add((i,j))
            done_count += 1
            now = time.time()
            if (now-last_print) > 0.5 or done_count==total:
                pct = 100.0 * done_count / total
                sys.stdout.write(f"\r    progress: {done_count}/{total} ({pct:5.1f}%)"); sys.stdout.flush()
                last_print = now
                try:
                    with open(tmpfile,"wb") as f: pickle.dump({"R": R, "done": list(done)}, f)
                except Exception:
                    pass
    print()
    X, Y = np.meshgrid(Gl, Gg)
    plt.figure(figsize=(8.2,6.4))
    im = plt.pcolormesh(X, Y, R, shading="auto", vmin=0.0, vmax=1.0)
    plt.xlabel("Glycine K"); plt.ylabel("GABA K")
    cb = plt.colorbar(im, label="Final r (mean over last 24 h)")
    CS = plt.contour(X, Y, R, levels=[0.2,0.4,0.6,0.8,0.9], colors="k", linewidths=0.7)
    plt.clabel(CS, inline=1, fontsize=8, fmt="r=%.1f")
    plt.title(title + "  (data-constrained W)")
    savefig(OUT/fname)
    try: (OUT/f"__tmp_{fname}.pkl").unlink()
    except Exception: pass

def all_sweeps_W_parallel(W, params, grid_n=20):
    sweep_panel_W_parallel(W, params, "VIP ON / GABA only",  "gaba", True,  "coupling_sweep_20x20_VIPON_gaba.png", grid_n)
    sweep_panel_W_parallel(W, params, "VIP ON / Gly only",   "gly",  True,  "coupling_sweep_20x20_VIPON_gly.png",  grid_n)
    sweep_panel_W_parallel(W, params, "VIP ON / GABA+Gly",   "both", True,  "coupling_sweep_20x20_VIPON_both.png", grid_n)
    sweep_panel_W_parallel(W, params, "VIP KO / GABA only",  "gaba", False, "coupling_sweep_20x20_VIPKO_gaba.png", grid_n)
    sweep_panel_W_parallel(W, params, "VIP KO / Gly only",   "gly",  False, "coupling_sweep_20x20_VIPKO_gly.png",  grid_n)
    sweep_panel_W_parallel(W, params, "VIP KO / GABA+Gly",   "both", False, "coupling_sweep_20x20_VIPKO_both.png", grid_n)

def vip_on_vs_ko(W, params, dt, T_h=120.0, vip_ko=0.0):
    Kg, Kl, s, dw, phig = params
    t, r_on, *_ = simulate_r_with_W(W, T_h, dt, Kg, Kl, phig, s, dw, vip_factor=1.0, seed=10)
    t, r_ko, *_ = simulate_r_with_W(W, T_h, dt, Kg, Kl, phig, s*1.15, dw, vip_factor=vip_ko, seed=10)
    plt.figure(figsize=(9,5))
    plt.plot(t, r_on, lw=2, label="VIP ON (GABA+Gly)")
    plt.plot(t, r_ko, lw=2, label=f"VIP KO (factor={vip_ko:.2f})")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Network synchronization: VIP ON vs KO (data-constrained W)")
    plt.legend(frameon=False)
    savefig(OUT/"vip_on_vs_ko_r_vs_time.png")

def noise_vs_r_W(W, params):
    Kg0, Kl0, s, dw, phig = params
    sigmas = np.linspace(0.0, 0.5, 9)
    labels = [
        "Noise only (no coupling)",
        "VIP ON / GABA only",
        "VIP ON / Gly only",
        "VIP ON / GABA+Gly",
        "VIP KO / GABA+Gly",
    ]
    curves = []
    for idx, label in enumerate(labels):
        tasks = []
        for k, sg in enumerate(sigmas):
            if label.startswith("Noise only"):
                Kg, Kl, vf, sscale = 0.0, 0.0, 1.0, 1.0
            elif "GABA only" in label and "VIP ON" in label:
                Kg, Kl, vf, sscale = Kg0, 0.0, 1.0, 1.0
            elif "Gly only" in label and "VIP ON" in label:
                Kg, Kl, vf, sscale = 0.0, Kl0, 1.0, 1.0
            elif "GABA+Gly" in label and "VIP ON" in label:
                Kg, Kl, vf, sscale = Kg0, Kl0, 1.0, 1.0
            else:
                Kg, Kl, vf, sscale = Kg0, Kl0, 0.0, 1.15
            tasks.append((W, Kg*vf, Kl*vf, phig, sg*sscale, dw, 1.0, 96.0, 0.06, 2000+idx*37+k))
        procs = max(1, os.cpu_count() or 1)
        with mp.get_context("spawn").Pool(processes=procs) as pool:
            ys = list(pool.imap_unordered(_simulate_point, tasks, chunksize=2))
        y_sorted = [ys[i] for i in np.argsort(sigmas)]
        curves.append((sigmas, y_sorted, label))
    plt.figure(figsize=(9,5))
    for xs, ys, lb in curves:
        plt.plot(xs, ys, marker="o", label=lb)
    plt.xlabel("Noise sigma"); plt.ylabel("Final r (mean over last 24 h)")
    plt.title("Noise impact under different coupling conditions (data-constrained W)")
    plt.legend(frameon=False)
    savefig(OUT/"noise_vs_r_multi.png")

def jetlag_recovery_W(W, params, shift_h=6.0, T_h=120.0, dt=0.05):
    Kg, Kl, s, dw, phig = params
    t, r, psi, _ = simulate_r_with_W(W, T_h, dt, Kg, Kl, phig, s, dw, vip_factor=1.0, seed=42)
    k0 = int(48.0/dt)
    shift = 2*np.pi*(shift_h/24.0)
    psi2 = psi.copy(); psi2[k0:] = ((psi2[k0:]+shift+np.pi)%(2*np.pi))-np.pi
    x = t[k0:] - t[k0]
    y = np.abs((psi2[k0:]-psi2[k0]+np.pi)%(2*np.pi)-np.pi) + 1e-6
    try:
        def model(x,a,tau,b): return a*np.exp(-x/tau)+b
        popt,_ = curve_fit(model, x, y, p0=(y[0],8.0,0.05), maxfev=10000)
        tau_est = float(popt[1])
    except Exception:
        tau_est = np.nan
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1); ax1.plot(t, r, lw=2, label="r(t)")
    ax1.axvline(48, color="k", lw=1, alpha=0.5); ax1.set_ylabel("Order parameter r"); ax1.legend(frameon=False)
    ax1.set_title(f"Jet-lag recovery: r and mean phase ψ  (τ ≈ {tau_est:.1f} h)")
    ax2 = plt.subplot(2,1,2, sharex=ax1); ax2.plot(t, psi2, lw=2, label="ψ(t)")
    ax2.axvline(48, color="k", lw=1, alpha=0.5); ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Mean phase ψ (rad)"); ax2.legend(frameon=False)
    savefig(OUT/"jetlag_recovery_time_phase.png")

def phase_rose(theta, outpath, bins=36):
    counts, edges = np.histogram(theta%(2*np.pi), bins=bins, range=(0,2*np.pi))
    centers = (edges[:-1] + edges[1:])/2
    ax = plt.subplot(111, projection='polar')
    ax.bar(centers, counts, width=(2*np.pi/bins), align='center')
    ax.set_title("Phase rose (final state)")
    savefig(outpath)

def main():
    log("[STEP 1/5] Single-cell fit + residuals + period histogram")
    T_est = run_single_cell_block()
    log("[STEP 2/5] Build W from MIC & empirical r(t)")
    W = build_W_from_MIC()
    t_emp, r_emp, psi_emp = empirical_r_from_TS(smooth_sigma_h=1.0, detrend=True)
    dt_emp = float(np.median(np.diff(t_emp)))
    log("[STEP 3/5] Global fit (DE, multi-core)")
    best = fit_params_to_empirical(W, t_emp, r_emp, last_avg_h=24.0, maxiter=60, popsize=18, seed=1234)
    log("[STEP 4/5] VIP ON vs KO + noise curves + jet-lag")
    vip_on_vs_ko(W, best, dt_emp, T_h=120.0, vip_ko=0.0)
    noise_vs_r_W(W, best)
    jetlag_recovery_W(W, best, shift_h=6.0, T_h=120.0, dt=dt_emp)
    log("[STEP 5/5] 6x 20x20 sweeps (parallel, with checkpoint & progress)")
    all_sweeps_W_parallel(W, best, grid_n=20)
    t, r, psi, th = simulate_r_with_W(W, 96.0, dt_emp, best[0], best[1], best[4], best[2], best[3], vip_factor=1.0, seed=99)
    phase_rose(th, OUT/"phase_rose_highres.png", bins=36)
    log("[DONE] All figures saved in: " + str(OUT.resolve()))

if __name__ == "__main__":
    mp.freeze_support()
    main()
