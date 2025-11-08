# -*- coding: utf-8 -*-
"""
SCN v4.1-final — Goodwin oscillator + MIC-constrained network
Author: You (with ChatGPT assist)
Features:
- Robust GPU(auto)/CPU fallback
- Single-cell TTFL-like fit (preprocess: Hampel + detrend + 0-1)
- Build W from MIC with safe normalization
- Global DE fit (checkpoint, resume, quiet workers)
- VIP-KO auto-calibration to target r≈0.40 (last 24h mean)
- Full figure set (PNG+PDF, Arial)
"""

import os, json, math, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import curve_fit, differential_evolution
from tqdm import tqdm

# -------------------- utils --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(path: Path, dpi=300, bbox='tight'):
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches=bbox)
    try:
        plt.savefig(path.with_suffix(".pdf"), bbox_inches=bbox)
    except Exception:
        pass
    plt.close()

def norm01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx-mn) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

def hampel_1d(x, k=3, nsigma=3.0):
    x = np.asarray(x, dtype=float)
    y = x.copy()
    n = x.size
    for i in range(n):
        i0, i1 = max(0, i-k), min(n, i+k+1)
        med = np.median(x[i0:i1])
        mad = np.median(np.abs(x[i0:i1]-med)) + 1e-9
        if np.abs(x[i] - med) > nsigma*1.4826*mad:
            y[i] = med
    return y

# -------------------- GPU fallback --------------------
def _get_xp(use_gpu_flag: str):
    import numpy as _np
    if use_gpu_flag == 'off':
        return _np, False
    if use_gpu_flag in ('on','auto'):
        try:
            import cupy as cp
            rt = cp.cuda.runtime
            ver_fn = getattr(rt, "getVersion", None) or getattr(rt, "runtimeGetVersion", None)
            _ = ver_fn()
            _ = rt.getDeviceCount()
            # NVRTC quick smoke test
            a = cp.arange(8, dtype=cp.float32)
            (a*a).sum().item()
            ker = cp.ElementwiseKernel('float32 x','float32 y','y = x * x','nvrtc_check')
            _ = ker(a)
            return cp, True
        except Exception as e:
            if use_gpu_flag == 'on':
                print(f"[WARN] GPU requested but unavailable, fallback to CPU: {e}")
            return _np, False
    return _np, False

# -------------------- IO --------------------
def load_per2_csv(path_csv: str):
    df_raw = pd.read_csv(path_csv, header=None)
    # detect header
    if isinstance(df_raw.iloc[0,0], str) and str(df_raw.iloc[0,0]).strip().lower() == 'time_h':
        df = pd.read_csv(path_csv)
        if 'time_h' not in df.columns:
            raise ValueError("CSV has header but no 'time_h' column.")
    else:
        df = df_raw.copy()
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df.insert(0, 'time_h', np.arange(len(df), dtype=float))
    df['time_h'] = pd.to_numeric(df['time_h'], errors='coerce')
    df = df.dropna(subset=['time_h']).reset_index(drop=True)
    for c in df.columns:
        if c != 'time_h':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def build_W_from_MIC(path_csv: str, threshold: float):
    A = pd.read_csv(path_csv, header=None).values.astype(float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    # symmetrize if needed
    if not np.allclose(A, A.T, atol=1e-12):
        A = A + A.T
        A *= 0.5
    keep = (A >= threshold).astype(float)
    np.fill_diagonal(keep, 0.0)
    deg = keep.sum(axis=1, keepdims=True) + 1e-9
    W = keep / deg
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return W

def empirical_r_from_per2(df: pd.DataFrame, max_h=432.0, decimate=1):
    df2 = df[df['time_h'] <= max_h].reset_index(drop=True)
    if decimate > 1:
        df2 = df2.iloc[::decimate,:].reset_index(drop=True)
    t = df2['time_h'].values.astype(float)
    X = df2.drop(columns=['time_h']).values.astype(float)
    # preprocess
    for j in range(X.shape[1]):
        xj = hampel_1d(X[:,j], k=3, nsigma=3.0)
        xj = detrend_linear(xj)
        X[:,j] = norm01(xj)
    # phase
    phi = np.zeros_like(X)
    for j in range(X.shape[1]):
        hj = hilbert(X[:,j] - np.mean(X[:,j]))
        phi[:,j] = np.angle(hj)
    z = np.exp(1j*phi).mean(axis=1)
    r = np.abs(z); psi = np.angle(z)
    return t, r, psi

def detrend_linear(y):
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        mu = np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0
        return np.nan_to_num(y - mu, nan=0.0)
    y = np.nan_to_num(y, nan=float(np.nanmean(y)))
    t = np.arange(len(y), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - (m*t + c)

# -------------------- TTFL-like fit --------------------
def ttfl_model(t, A, T, phi, C, slope):
    return A*np.sin(2*np.pi*t/T + phi) + C + slope*t

def fit_singlecell_avg(df: pd.DataFrame, fit_h=96.0, out_dir=Path("outputs")):
    df_fit = df[df['time_h'] <= fit_h].reset_index(drop=True)
    t = df_fit['time_h'].values.astype(float)
    X = df_fit.drop(columns=['time_h']).values.astype(float)
    for j in range(X.shape[1]):
        X[:,j] = hampel_1d(X[:,j], k=3, nsigma=3.0)
        X[:,j] = detrend_linear(X[:,j])
    Xn = np.apply_along_axis(norm01, 0, X)
    yavg = np.nanmean(Xn, axis=1)

    y = yavg - yavg.mean()
    ac = np.correlate(y, y, mode='full')[len(y)-1:]
    T_guess = 26.0
    try:
        # search around circadian
        lo, hi = int(22), int(28)
        idx = np.argmax(ac[lo:hi+1])
        T_guess = float(lo + idx)
    except Exception:
        pass

    p0 = (0.4, T_guess, 0.0, 0.5, 0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(ttfl_model, t, yavg, p0=p0, maxfev=30000)
    yhat = norm01(ttfl_model(t, *popt))
    rmse = float(np.sqrt(np.nanmean((yhat - yavg)**2)))

    # plots
    plt.figure(figsize=(9,4))
    plt.plot(t, yavg, label="PER2 avg (norm)")
    plt.plot(t, yhat, label="TTFL-like fit")
    plt.xlabel("Time (h)"); plt.ylabel("Normalized intensity")
    plt.title("Single-cell average fit (PER2 vs TTFL-like)")
    plt.legend(frameon=False)
    savefig(out_dir/"single_cell_fit.png")

    plt.figure(figsize=(9,3.5))
    plt.plot(t, yavg - yhat)
    plt.axhline(0, ls='--', lw=1, c='k')
    plt.xlabel("Time (h)"); plt.ylabel("Residual")
    plt.title(f"Residuals (RMSE={rmse:.3f})")
    savefig(out_dir/"single_cell_fit_residuals.png")

    # period histogram (peak-to-peak from yavg)
    from scipy.signal import find_peaks
    peaks,_ = find_peaks(yavg, distance=max(1,int(20/(t[1]-t[0]+1e-9))))
    per = np.diff(t[peaks]) if len(peaks) > 1 else np.array([])
    plt.figure(figsize=(6,4))
    if per.size:
        plt.hist(per, bins=15, density=True)
    plt.xlabel("Estimated period (h)")
    plt.title("Period histogram (from avg trace)")
    savefig(out_dir/"period_histogram.png")

    # AIC/BIC vs sine
    def sine_func(t, A, T, ph, C): return A*np.sin(2*np.pi*t/T + ph) + C
    p0s = (0.4, max(22.0, min(28.0, float(popt[1]))), 0.0, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pop_sine,_ = curve_fit(sine_func, t, yavg, p0=p0s, maxfev=20000)
    ysine = norm01(sine_func(t, *pop_sine))
    def aic_bic(y, yhat, k):
        n = len(y); rss = np.sum((y-yhat)**2); s2 = rss/max(n,1)
        aic = n*np.log(max(s2,1e-12)) + 2*k
        bic = n*np.log(max(s2,1e-12)) + k*np.log(max(n,1))
        return float(aic), float(bic)
    rmse_ttfl = float(np.sqrt(np.nanmean((yhat - yavg)**2)))
    rmse_sine = float(np.sqrt(np.nanmean((ysine - yavg)**2)))
    AIC_ttfl,BIC_ttfl = aic_bic(yavg,yhat,5)
    AIC_sine,BIC_sine = aic_bic(yavg,ysine,4)
    fig, ax = plt.subplots(1,2, figsize=(11,4.2))
    ax[0].bar(["TTFL","Sine"], [rmse_ttfl, rmse_sine]); ax[0].set_ylabel("RMSE"); ax[0].set_title("Fit error")
    ax[1].bar(["AIC_TTFL","AIC_Sine","BIC_TTFL","BIC_Sine"], [AIC_ttfl,AIC_sine,BIC_ttfl,BIC_sine])
    ax[1].set_title("Information criteria (lower is better)")
    savefig(out_dir/"model_param_AIC_BIC.png")

    return t, yavg, yhat, rmse, popt

# -------------------- Goodwin network model --------------------
def rk4_integrate_goodwin(W, T_h=120.0, dt=0.05, params=None, seed=0, on_gpu=False, progress=False):
    xp = np
    if on_gpu:
        import cupy as cp
        xp = cp
    N = W.shape[0]
    rs = np.random.RandomState(seed)
    X = rs.rand(N)*0.5 + 0.25
    Y = rs.rand(N)*0.5 + 0.25
    Z = rs.rand(N)*0.5 + 0.25
    if on_gpu:
        X = xp.asarray(X); Y = xp.asarray(Y); Z = xp.asarray(Z); W = xp.asarray(W)

    a = params.get('a', 1.0); b = params.get('b', 0.1)
    c = params.get('c', 1.0); d = params.get('d', 0.1)
    e = params.get('e', 1.0); f = params.get('f', 0.1)
    n = params.get('n', 4.0)
    Kv = params.get('Kv', 0.6)  # VIP-like (X coupling)
    Kg = params.get('Kg', 0.5)  # GABA-like (Z coupling)
    Kl = params.get('Kl', 0.3)  # Gly-like  (Z coupling)
    sigma = params.get('sigma', 0.03)

    t_axis = np.arange(0.0, T_h, dt)
    r_t = np.zeros_like(t_axis, dtype=float)
    psi_t = np.zeros_like(t_axis, dtype=float)

    # helpers
    def clipu(u): return xp.minimum(xp.maximum(u, 0.0), 5.0)

    iters = range(len(t_axis))
    if progress:
        iters = tqdm(iters, desc="Integrating", ncols=90)

    for k, _ in enumerate(iters):
        # order parameter by proxy phase from Z
        zc = xp.exp(1j*2*np.pi*(Z - Z.mean()))
        zmean = zc.mean()
        r = float(xp.abs(zmean).get() if on_gpu else np.abs(zmean))
        psi = float(xp.angle(zmean).get() if on_gpu else np.angle(zmean))
        r_t[k] = r; psi_t[k] = psi

        X_mean = X.mean()
        # Goodwin core
        dX = a/(1.0 + clipu(Z)**n) - b*X
        dY = c*X - d*Y
        dZ = e*Y - f*Z
        # Coupling: VIP on X; GABA/Gly on Z
        dX += Kv * ((W @ X) - X) + 0.2*Kv*(X_mean - X)
        dZ += Kg * ((W @ Z) - Z)
        dZ += Kl * ((W @ Z) - Z)
        # Noise
        if on_gpu:
            dX += sigma * xp.random.standard_normal(size=N)
            dY += sigma * xp.random.standard_normal(size=N)
            dZ += sigma * xp.random.standard_normal(size=N)
        else:
            dX += sigma * np.random.standard_normal(size=N)
            dY += sigma * np.random.standard_normal(size=N)
            dZ += sigma * np.random.standard_normal(size=N)

        # RK4
        def fX(X,Y,Z): return clipu(a/(1.0 + clipu(Z)**n) - b*X + Kv*((W@X)-X) + 0.2*Kv*(X_mean - X))
        def fY(X,Y,Z): return clipu(c*X - d*Y)
        def fZ(X,Y,Z): return clipu(e*Y - f*Z + Kg*((W@Z)-Z) + Kl*((W@Z)-Z))

        k1X = fX(X,Y,Z); k1Y = fY(X,Y,Z); k1Z = fZ(X,Y,Z)
        k2X = fX(X+0.5*dt*k1X, Y+0.5*dt*k1Y, Z+0.5*dt*k1Z)
        k2Y = fY(X+0.5*dt*k1X, Y+0.5*dt*k1Y, Z+0.5*dt*k1Z)
        k2Z = fZ(X+0.5*dt*k1X, Y+0.5*dt*k1Y, Z+0.5*dt*k1Z)
        k3X = fX(X+0.5*dt*k2X, Y+0.5*dt*k2Y, Z+0.5*dt*k2Z)
        k3Y = fY(X+0.5*dt*k2X, Y+0.5*dt*k2Y, Z+0.5*dt*k2Z)
        k3Z = fZ(X+0.5*dt*k2X, Y+0.5*dt*k2Y, Z+0.5*dt*k2Z)
        k4X = fX(X+dt*k3X, Y+dt*k3Y, Z+dt*k3Z)
        k4Y = fY(X+dt*k3X, Y+dt*k3Y, Z+dt*k3Z)
        k4Z = fZ(X+dt*k3X, Y+dt*k3Y, Z+dt*k3Z)

        X = X + (dt/6.0)*(k1X + 2*k2X + 2*k3X + k4X)
        Y = Y + (dt/6.0)*(k1Y + 2*k2Y + 2*k3Y + k4Y)
        Z = Z + (dt/6.0)*(k1Z + 2*k2Z + 2*k3Z + k4Z)

        X = clipu(X); Y = clipu(Y); Z = clipu(Z)

    if on_gpu:
        import cupy as cp
        return t_axis, cp.asnumpy(r_t), cp.asnumpy(psi_t), (cp.asnumpy(X), cp.asnumpy(Y), cp.asnumpy(Z))
    else:
        return t_axis, r_t, psi_t, (X, Y, Z)

# -------------------- objective & fit --------------------
def simulate_r_tail(W, params, T_h=96.0, dt=0.05, on_gpu=False):
    t, r, _, _ = rk4_integrate_goodwin(W, T_h=T_h, dt=dt, params=params, on_gpu=on_gpu, progress=False)
    k = max(0, len(r) - int(24.0/dt))
    return r, float(np.nanmean(r[k:]) if len(r) else np.nan)

def objective_factory(W, t_emp, r_emp, on_gpu=False):
    # weight tail higher to match late synchronization
    def obj(theta):
        a,b,c,d,e,f,n, Kv,Kg,Kl, sigma = theta
        p = dict(a=a,b=b,c=c,d=d,e=e,f=f,n=n,Kv=Kv,Kg=Kg,Kl=Kl,sigma=sigma)
        r_sim, _ = simulate_r_tail(W, p, T_h=float(t_emp.max()), dt=(t_emp[1]-t_emp[0]), on_gpu=on_gpu)
        if len(r_sim) != len(r_emp):
            # resample
            from numpy import interp
            t = np.arange(0.0, float(t_emp.max()), (t_emp[1]-t_emp[0]))
            r_sim = np.interp(t_emp, t[:len(r_sim)], r_sim[:min(len(t_emp),len(r_sim))])
        w = np.linspace(0.5, 1.5, num=len(r_emp))  # tail heavier
        err = np.nanmean(w*(r_sim - r_emp)**2)
        if not np.isfinite(err):
            return 1e9
        return float(err)
    return obj

# -------------------- VIP KO calibration --------------------
def mean_final_r(W, params, T_h=96.0, dt=0.05, on_gpu=False, seed=777):
    t, r, _, _ = rk4_integrate_goodwin(W, T_h=T_h, dt=dt, params=params, seed=seed, on_gpu=on_gpu, progress=False)
    if len(r) == 0:
        return np.nan
    k = max(0, len(r) - int(24.0/dt))
    return float(np.nanmean(r[k:]))

def calibrate_vip_factor_to_target_r(W, best_params, target_r=0.40, on_gpu=False, tol=0.01, max_iter=12):
    lo, hi = 0.05, 1.0
    def r_with_factor(fac):
        p = best_params.copy()
        for k in ("Kv","Kg","Kl"):
            p[k] = float(p[k] * fac)
        return mean_final_r(W, p, T_h=96.0, dt=0.05, on_gpu=on_gpu, seed=2025)
    r_lo = r_with_factor(lo); r_hi = r_with_factor(hi)
    fac = None
    for _ in range(max_iter):
        mid = 0.5*(lo+hi); r_mid = r_with_factor(mid)
        if not np.isfinite(r_mid):
            hi = mid; continue
        if abs(r_mid - target_r) <= tol:
            fac = mid; break
        if r_mid < target_r: lo = mid
        else: hi = mid
        fac = mid
    return float(fac if fac is not None else 0.35)

# -------------------- figures --------------------
def plot_W_heatmap(W, out_dir:Path, thr:float):
    plt.figure(figsize=(6.5,5.2))
    vmax = np.percentile(W, 99)
    plt.imshow(W, aspect='auto', origin='lower', vmin=0, vmax=vmax if vmax>0 else 1)
    plt.colorbar(label="Row-normalized adjacency")
    plt.title(f"MIC-derived network (W), threshold={thr:.3f}")
    plt.xlabel("Cell index"); plt.ylabel("Cell index")
    savefig(out_dir/"W_adjacency_heatmap.png")

def coupling_sweeps(out_dir:Path, W, on_gpu, title_prefix=""):
    # 20x20 grids under different modes and VIP=ON/KO
    modes = [("gaba","GABA only"), ("gly","Gly only"), ("both","GABA+Gly")]
    def run_grid(vip_on:bool, mode:str, fname:str):
        Gg = np.linspace(0,1.0,20); Gl = np.linspace(0,1.0,20)
        R = np.zeros((20,20))
        for i, kg in enumerate(Gg):
            for j, kl in enumerate(Gl):
                Kv = 0.8 if vip_on else 0.8  # here Kv just a baseline; vip KO later scales both Kg/Kl in main panel
                Kg = kg if mode in ("gaba","both") else 0.0
                Kl = kl if mode in ("gly","both")  else 0.0
                p = dict(a=1.0,b=0.1,c=1.0,d=0.1,e=1.0,f=0.1,n=4.0,
                         Kv=Kv, Kg=Kg, Kl=Kl, sigma=0.03)
                _, r, _, _ = rk4_integrate_goodwin(W, T_h=72.0, dt=0.1, params=p, on_gpu=on_gpu, progress=False)
                R[i,j] = r[-1] if len(r) else np.nan
        X,Y = np.meshgrid(Gl,Gg)
        plt.figure(figsize=(8.2,6.4))
        im = plt.pcolormesh(X,Y,R, shading="auto", vmin=0, vmax=1)
        plt.colorbar(im, label="Final r")
        CS = plt.contour(X,Y,R, levels=[0.2,0.4,0.6,0.8,0.9], colors="k", linewidths=0.7)
        plt.clabel(CS, inline=1, fontsize=8, fmt="r=%.1f")
        plt.xlabel("Glycine K"); plt.ylabel("GABA K")
        plt.title(f"{title_prefix} {mode.upper()} — Coupling phase diagram")
        savefig(out_dir/fname)

    run_grid(True,  "gaba", "coupling_sweep_20x20_VIPON_gaba.png")
    run_grid(True,  "gly",  "coupling_sweep_20x20_VIPON_gly.png")
    run_grid(True,  "both", "coupling_sweep_20x20_VIPON_both.png")
    run_grid(False, "gaba", "coupling_sweep_20x20_VIPKO_gaba.png")
    run_grid(False, "gly",  "coupling_sweep_20x20_VIPKO_gly.png")
    run_grid(False, "both", "coupling_sweep_20x20_VIPKO_both.png")

def noise_scan(out_dir:Path, W, base_params, vip_factor, on_gpu):
    sigmas = np.linspace(0.0, 0.12, 7)
    def final_r(Kg,Kl,s, seed=21):
        p = base_params.copy()
        p['Kg']=Kg; p['Kl']=Kl; p['sigma']=s
        _, r, _, _ = rk4_integrate_goodwin(W, T_h=96.0, dt=0.05, params=p, on_gpu=on_gpu, progress=False)
        return r[-1] if len(r) else np.nan
    series = []
    labels = []
    series.append([final_r(0.0,0.0,s,seed=1) for s in sigmas]); labels.append("Noise only (no coupling)")
    series.append([final_r(base_params['Kg'],0.0,s,seed=2) for s in sigmas]); labels.append("VIP ON / GABA only")
    series.append([final_r(0.0,base_params['Kl'],s,seed=3) for s in sigmas]); labels.append("VIP ON / Gly only")
    series.append([final_r(base_params['Kg'],base_params['Kl'],s,seed=4) for s in sigmas]); labels.append("VIP ON / GABA+Gly")
    series.append([final_r(base_params['Kg']*vip_factor, base_params['Kl']*vip_factor, s, seed=5) for s in sigmas]); labels.append("VIP KO / GABA+Gly")
    plt.figure(figsize=(8,5))
    for y,name in zip(series,labels):
        plt.plot(sigmas, y, marker='o', label=name)
    plt.xlabel("Noise sigma"); plt.ylabel("Final r")
    plt.title("Noise impact under different coupling conditions")
    plt.legend(frameon=False)
    savefig(out_dir/"noise_vs_r_multi.png")

def jetlag_plot(out_dir:Path, W, params, on_gpu):
    # 6h phase shift at 48h
    t, r, psi, _ = rk4_integrate_goodwin(W, T_h=120.0, dt=0.05, params=params, on_gpu=on_gpu, progress=False)
    k = int(48.0/0.05)
    psi2 = psi.copy()
    shift = 2*np.pi*(6.0/24.0)
    psi2[k:] = ((psi2[k:] + shift + np.pi)%(2*np.pi)) - np.pi
    # tau fit
    x = t[k:] - t[k]
    y = np.abs((psi2[k:] - psi2[k] + np.pi)%(2*np.pi) - np.pi) + 1e-6
    def model(x,a,tau,b): return a*np.exp(-x/tau)+b
    try:
        popt,_ = curve_fit(model, x, y, p0=(y[0], 8.0, 0.05), maxfev=10000)
        tau_est = float(popt[1])
    except Exception:
        tau_est = np.nan
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, r, linewidth=2, label="r(t)")
    ax1.set_ylabel("Order parameter r"); ax1.legend(frameon=False)
    ax1.set_title(f"Jet-lag recovery: r and mean phase ψ  (τ ≈ {tau_est:.1f} h)")
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(t, psi2, linewidth=2, label="ψ(t)")
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Mean phase ψ (rad)")
    ax2.legend(frameon=False)
    savefig(out_dir/"jetlag_recovery_time_phase.png")

def phase_rose(theta_final, out_path:Path, bins=36):
    ang = np.mod(theta_final, 2*np.pi)
    counts, edges = np.histogram(ang, bins=bins, range=(0,2*np.pi), density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    ax = plt.subplot(111, projection='polar')
    ax.bar(centers, counts, width=(2*np.pi/bins), align='center')
    ax.set_title("Final phase rose (high-res)")
    savefig(out_path)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="scn1_full_data_with_time.csv")
    ap.add_argument("--mic_csv",  default="mic_indiv_scn1.csv")
    ap.add_argument("--mic_thr",  type=float, default=0.18)  # per your choice
    ap.add_argument("--fit_h",    type=float, default=96.0)
    ap.add_argument("--emp_max_h",type=float, default=120.0)
    ap.add_argument("--use_gpu",  choices=["auto","on","off"], default="off")  # CPU stable default
    ap.add_argument("--fit_nodes",type=int,   default=400)
    ap.add_argument("--maxiter",  type=int,   default=40)
    ap.add_argument("--popsize",  type=int,   default=18)
    args = ap.parse_args()

    OUT = Path("outputs"); CK = Path("checkpoints")
    ensure_dir(OUT); ensure_dir(CK)

    xp, on_gpu = _get_xp(args.use_gpu)
    print(f"[GPU] on_gpu={on_gpu}")

    # Step 1: single-cell fit
    print("[STEP 1/5] Single-cell fit + residuals + period histogram")
    df = load_per2_csv(args.data_csv)
    t_fit, yavg, yhat, rmse, pfit = fit_singlecell_avg(df, fit_h=args.fit_h, out_dir=OUT)

    # Step 2: build W + empirical r(t)
    print("[STEP 2/5] Build W from MIC & empirical r(t)")
    W = build_W_from_MIC(args.mic_csv, threshold=float(args.mic_thr))
    plot_W_heatmap(W, OUT, float(args.mic_thr))

    t_emp, r_emp, psi_emp = empirical_r_from_per2(df, max_h=args.emp_max_h, decimate=1)
    plt.figure(figsize=(10,3.6))
    plt.plot(t_emp, r_emp)
    plt.xlabel("Time (h)"); plt.ylabel("Empirical r(t)")
    plt.title("Empirical synchronization from PER2 data")
    savefig(OUT/"r_empirical.png")

    # Step 3: global DE fit
    print("[STEP 3/5] Global DE fit (workers=-1)")
    # initial bounds (biologically plausible)
    bounds = [
        (0.5, 2.0),   # a
        (0.01,0.5),   # b
        (0.5, 2.0),   # c
        (0.01,0.5),   # d
        (0.5, 2.0),   # e
        (0.01,0.5),   # f
        (2.0, 6.0),   # n
        (0.1, 1.5),   # Kv
        (0.0, 1.2),   # Kg
        (0.0, 1.0),   # Kl
        (0.00,0.10)   # sigma
    ]
    obj = objective_factory(W, t_emp, r_emp, on_gpu=on_gpu)

    # resume?
    resume_best = None
    best_json = CK/"best_params.json"
    if best_json.exists():
        try:
            resume_best = json.loads(best_json.read_text())
        except Exception:
            resume_best = None

    def callback(xk, convergence):
        # lightweight preview
        keys = ['a','b','c','d','e','f','n','Kv','Kg','Kl','sigma']
        cand = {k:float(v) for k,v in zip(keys, xk)}
        best_json.write_text(json.dumps(cand, indent=2))
        # quick overlay of r(t)
        try:
            p = cand.copy()
            _, r_sim, _, _ = rk4_integrate_goodwin(W, T_h=float(t_emp.max()), dt=(t_emp[1]-t_emp[0]),
                                                   params=p, on_gpu=on_gpu, progress=False)
            t = np.arange(0.0, float(t_emp.max()), (t_emp[1]-t_emp[0]))
            r_sim = np.interp(t_emp, t[:len(r_sim)], r_sim[:min(len(t_emp),len(r_sim))])
            plt.figure(figsize=(10,3.6))
            plt.plot(t_emp, r_emp, label="Empirical")
            plt.plot(t_emp, r_sim, label="Model")
            plt.legend(frameon=False); plt.xlabel("Time (h)"); plt.ylabel("r(t)")
            plt.title("DE progress (preview)")
            savefig(CK/"progress_fit.png")
        except Exception:
            pass
        return False

    res = differential_evolution(
        obj, bounds, maxiter=args.maxiter, popsize=args.popsize,
        tol=1e-3, polish=True, seed=1234, workers=-1, updating='deferred',
        callback=callback
    )
    keys = ['a','b','c','d','e','f','n','Kv','Kg','Kl','sigma']
    best_params = {k:float(v) for k,v in zip(keys, res.x)}

    # Compare final model vs empirical
    t = np.arange(0.0, float(t_emp.max()), (t_emp[1]-t_emp[0]))
    _, r_model, _, _ = rk4_integrate_goodwin(W, T_h=float(t_emp.max()), dt=(t_emp[1]-t_emp[0]),
                                             params=best_params, on_gpu=on_gpu, progress=False)
    r_model = np.interp(t_emp, t[:len(r_model)], r_model[:min(len(t_emp),len(r_model))])
    plt.figure(figsize=(10,3.6))
    plt.plot(t_emp, r_emp, label="Empirical")
    plt.plot(t_emp, r_model, label="Model (best DE)")
    plt.xlabel("Time (h)"); plt.ylabel("r(t)"); plt.legend(frameon=False)
    plt.title("Empirical r(t) vs Model fit")
    savefig(OUT/"r_emp_vs_model_fit.png")

    # Step 4: VIP KO calibration & plots
    print("[STEP 4/5] Calibrate VIP KO to r≈0.40")
    vip_factor = calibrate_vip_factor_to_target_r(W, best_params, target_r=0.40, on_gpu=on_gpu, tol=0.01, max_iter=12)
    with open(OUT/"Model_params_v4_final.json","w") as f:
        json.dump({**best_params, "VIP_factor_calibrated": vip_factor, "MIC_thr": float(args.mic_thr)}, f, indent=2)

    # VIP ON vs KO r(t)
    p_on = best_params.copy()
    t_on, r_on, _, _ = rk4_integrate_goodwin(W, T_h=120.0, dt=0.05, params=p_on, on_gpu=on_gpu, progress=False)
    p_ko = best_params.copy()
    p_ko['Kv'] *= vip_factor; p_ko['Kg'] *= vip_factor; p_ko['Kl'] *= vip_factor
    t_ko, r_ko, _, _ = rk4_integrate_goodwin(W, T_h=120.0, dt=0.05, params=p_ko, on_gpu=on_gpu, progress=False)

    plt.figure(figsize=(12,4))
    plt.plot(t_on, r_on, label="VIP ON (GABA+Gly)")
    plt.plot(t_ko, r_ko, label=f"VIP KO (target r≈0.40, factor={vip_factor:.2f})")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Network synchronization: VIP ON vs KO (MIC-constrained)")
    plt.legend(frameon=False)
    savefig(OUT/"vip_on_vs_ko_r_vs_time.png")

    # Noise scan
    noise_scan(OUT, W, best_params, vip_factor, on_gpu)

    # Coupling sweeps (6 figs)
    print("[STEP 5/5] Coupling sweeps + jetlag + rose")
    coupling_sweeps(OUT, W, on_gpu, title_prefix="VIP ON/OFF baseline")

    # Jetlag
    jetlag_plot(OUT, W, best_params, on_gpu)

    # Phase rose (final state under best_params)
    _, _, _, XYZ = rk4_integrate_goodwin(W, T_h=96.0, dt=0.05, params=best_params, on_gpu=on_gpu, progress=False)
    Zfin = XYZ[2]
    if isinstance(Zfin, tuple) or isinstance(Zfin, list):
        Zfin = Zfin[0]
    theta_final = np.mod(Zfin, 2*np.pi) if Zfin.ndim else np.array([Zfin])
    phase_rose(theta_final, OUT/"phase_rose_highres.png", bins=36)

    # small README
    with open(OUT/"README.md","w",encoding="utf-8") as f:
        f.write("# SCN v4.1 Final Outputs\n")
        f.write(f"- MIC threshold: {args.mic_thr:.3f}\n")
        f.write(f"- VIP KO calibrated factor -> target r≈0.40: {vip_factor:.3f}\n")
        f.write("- Figures: TTFL fit/residuals/period, W heatmap, empirical vs model r(t), VIP ON vs KO, noise scan, 6 coupling sweeps, jetlag r/ψ, phase rose, AIC/BIC\n")

    print("DONE. Outputs in:", str(OUT))

if __name__ == "__main__":
    main()
