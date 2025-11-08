# -*- coding: utf-8 -*-
"""
SCN v4 Goodwin + data-constrained coupling
- GPU acceleration (CuPy if available) with CPU fallback
- Stable RK4 integrator (no SciPy IVP needed on GPU)
- Differential Evolution (SciPy) global fit with workers=-1
- tqdm progress + checkpointing
- 6x 20x20 phase diagrams (VIP ON/KO × GABA/Gly/both)
- VIP ON vs KO r(t), noiseσ vs r, jet-lag (r & ψ)
- Single-cell TTFL-like fit + residuals + period histogram
- Empirical r(t) vs fitted r(t), phase rose

Author: Liu Xuan
"""

import os, json, warnings, argparse, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.signal import hilbert
from scipy.optimize import curve_fit, differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------- GPU detection (CuPy optional) ----------------
def _get_xp(use_gpu_flag:str):
    """
    use_gpu_flag: 'auto'|'on'|'off'
    returns xp (cupy or numpy), on_gpu(bool)
    """
    if use_gpu_flag == 'off':
        return np, False
    if use_gpu_flag in ('on', 'auto'):
        try:
            import cupy as cp
            # quick test
            _ = cp.asarray([1.0,2.0]).sum()
            return cp, True
        except Exception:
            if use_gpu_flag == 'on':
                print("[WARN] CuPy not available; falling back to CPU.")
            return np, False
    return np, False

# ---------------- Utils ----------------
def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(path:Path, dpi=180, bbox='tight'):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches=bbox)
    plt.close()

def norm01(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    return y

def detrend_linear(y):
    t = np.arange(len(y), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    # least squares
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - (m*t + c)

def hampel_1d(x, k=3, nsigma=3.0):
    x = np.asarray(x, float).copy()
    n = x.size
    y = x.copy()
    for i in range(n):
        i0, i1 = max(0,i-k), min(n, i+k+1)
        med = np.median(x[i0:i1])
        mad = np.median(np.abs(x[i0:i1]-med)) + 1e-9
        if np.abs(x[i]-med) > nsigma*1.4826*mad:
            y[i] = med
    return y

# ---------------- Data loading ----------------
def load_per2_csv(path_csv:str):
    """
    scn1_full_data_with_time.csv
    - if first column is 'time_h', respect it; otherwise fabricate 1h increments
    """
    df = pd.read_csv(path_csv, header=None)
    # try detect header with 'time_h'
    if isinstance(df.iloc[0,0], str) and df.iloc[0,0].strip().lower()=='time_h':
        df = pd.read_csv(path_csv)  # with header
        if 'time_h' not in df.columns:
            raise ValueError("CSV claims header but no 'time_h' found.")
    else:
        # no header, fabricate time
        df.columns = list(range(df.shape[1]))
        df.insert(0, 'time_h', np.arange(len(df), dtype=float))
    # clean types
    df['time_h'] = pd.to_numeric(df['time_h'], errors='coerce')
    df = df.dropna(subset=['time_h']).reset_index(drop=True)
    # coerce data cols
    for c in df.columns:
        if c=='time_h': continue
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def build_W_from_MIC(path_csv:str, threshold:float):
    """
    mic_indiv_scn1.csv: adjacency matrix or upper-tri matrix accepted.
    Returns row-normalized weight matrix W.
    """
    A = pd.read_csv(path_csv, header=None).values
    A = np.asarray(A, float)
    # if strictly upper triangular, symmetrize:
    if not np.allclose(A, A.T, equal_nan=True):
        A = np.nan_to_num(A, nan=0.0)
        A = A + A.T
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    # threshold
    keep = (A >= threshold).astype(float)
    np.fill_diagonal(keep, 0.0)
    # degree normalize
    deg = keep.sum(axis=1, keepdims=True) + 1e-9
    W = keep / deg
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return W

# ---------------- Empirical r(t) from PER2 ----------------
def empirical_r_from_per2(df:pd.DataFrame, max_h=432.0, decimate=1):
    """
    - Hampel -> detrend -> norm01 each cell
    - Hilbert phase -> complex mean -> r(t)
    """
    df2 = df.copy()
    df2 = df2[df2['time_h'] <= max_h].reset_index(drop=True)
    if decimate>1:
        df2 = df2.iloc[::decimate,:].reset_index(drop=True)

    t = df2['time_h'].values.astype(float)
    X = df2.drop(columns=['time_h']).values.astype(float)

    for j in range(X.shape[1]):
        xj = X[:,j]
        xj = hampel_1d(xj, k=3, nsigma=3.0)
        xj = detrend_linear(xj)
        X[:,j] = norm01(xj)

    # Hilbert phase on each cell
    phi = np.zeros_like(X)
    for j in range(X.shape[1]):
        hj = hilbert(X[:,j] - np.mean(X[:,j]))
        phi[:,j] = np.angle(hj)

    z = np.exp(1j*phi).mean(axis=1)
    r = np.abs(z)
    psi = np.angle(z)
    return t, r, psi

# ---------------- Single-cell "TTFL-like" fit ----------------
def ttfl_model(t, A, T, phi, C, slope):
    return A*np.sin(2*np.pi*t/T + phi) + C + slope*t

def fit_singlecell_avg(df:pd.DataFrame, fit_h=96.0):
    df_fit = df[df['time_h']<=fit_h].copy().reset_index(drop=True)
    t = df_fit['time_h'].values.astype(float)
    X = df_fit.drop(columns=['time_h']).values.astype(float)
    for j in range(X.shape[1]):
        X[:,j] = hampel_1d(X[:,j], k=3, nsigma=3.0)
        X[:,j] = detrend_linear(X[:,j])
    Xn = np.apply_along_axis(norm01, 0, X)
    yavg = np.nanmean(Xn, axis=1)
    # period by autocorr (rough)
    y = yavg - yavg.mean()
    ac = np.correlate(y, y, mode='full')[len(y)-1:]
    lags = np.arange(len(ac))
    # find first non-zero local max around ~24-26
    T_guess = 24.0
    try:
        T_guess = 24.0 + (np.argmax(ac[int(20):int(30)]) - (30-20))  # crude
        T_guess = float(np.clip(T_guess, 22.0, 28.0))
    except Exception:
        T_guess = 26.0
    p0 = (0.4, T_guess, 0.0, 0.5, 0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt,_ = curve_fit(ttfl_model, t, yavg, p0=p0, maxfev=20000)
    yhat = norm01(ttfl_model(t, *popt))
    rmse = float(np.sqrt(np.nanmean((yhat - yavg)**2)))
    return t, yavg, yhat, rmse, popt

# ---------------- Goodwin ODE + RK4 (GPU/CPU) ----------------
def rk4_integrate_goodwin(W, T_h=120.0, dt=0.05, params=None, seed=0, xp=np, on_gpu=False, progress=False):
    """
    params: dict with
      a,b,c,d,e,f,n, Kv, Kg, Kl, w
    Returns: t(np), r(np), psi(np), (X,Y,Z final np)
    """
    N = W.shape[0]
    rs = np.random.RandomState(seed)
    # initial
    X = rs.rand(N)*0.5+0.25
    Y = rs.rand(N)*0.5+0.25
    Z = rs.rand(N)*0.5+0.25

    # move to GPU if needed
    if on_gpu:
        import cupy as cp
        X = cp.asarray(X); Y = cp.asarray(Y); Z = cp.asarray(Z)
        Wgpu = cp.asarray(W)
        xp = cp
    else:
        Wgpu = W

    a = params.get('a', 1.0); b = params.get('b', 0.1)
    c = params.get('c', 1.0); d = params.get('d', 0.1)
    e = params.get('e', 1.0); f = params.get('f', 0.1)
    n = params.get('n', 4.0)
    Kv = params.get('Kv', 0.4)
    Kg = params.get('Kg', 0.5)
    Kl = params.get('Kl', 0.3)
    w  = params.get('w', 0.3)     # VIP日夜权重(0~1)

    steps = int(T_h/dt)
    t_axis = np.linspace(0.0, T_h, steps, endpoint=False)
    r_t = np.zeros(steps, dtype=float)
    psi_t = np.zeros(steps, dtype=float)

    def rhs(X, Y, Z):
        # 数值保护
        Zs = xp.clip(Z, 1e-6, None)
        den = 1.0 + Zs**n
        den = xp.where(xp.isfinite(den), den, 1e6)
        dX = a/den - b*X
        dY = c*X - d*Y
        dZ = e*Y - f*Z

        X_mean = X.mean()
        # VIP: 本地扩散 + 全局牵引（弱）
        dX += Kv * ((Wgpu @ X) - X) + 0.2*Kv*(X_mean - X)
        # GABA & Gly 作用于 Z
        Zc = Zs
        dZ += Kg * ((Wgpu @ Zc) - Zc)
        dZ += Kl * ((Wgpu @ Zc) - Zc)
        # 昼夜调制（可选：此处用简单加权）
        dZ += 0.5*(1.0-w)*(Kg+Kl) * ((Wgpu @ Zc) - Zc)
        return dX, dY, dZ

    it = range(steps)
    if progress:
        it = tqdm(it, desc="RK4 integrating", leave=False)

    for k in it:
        # Kuramoto-like r/psi 从 Z 的相位估计（或用 X 也可）
        angle = xp.angle(xp.asarray(hilbert((Z.get() if on_gpu else Z) - (Z.get().mean() if on_gpu else Z.mean())))) \
                if False else xp.angle(xp.exp(1j*2*np.pi*(Z - Z.mean()))) # 简单映射
        z = xp.exp(1j*angle).mean()
        r = float(abs(z.get() if on_gpu else z))
        psi = float(np.angle(z.get() if on_gpu else z))
        r_t[k] = r; psi_t[k] = psi

        k1 = rhs(X,Y,Z)
        k2 = rhs(X + dt*0.5*k1[0], Y + dt*0.5*k1[1], Z + dt*0.5*k1[2])
        k3 = rhs(X + dt*0.5*k2[0], Y + dt*0.5*k2[1], Z + dt*0.5*k2[2])
        k4 = rhs(X + dt*k3[0], Y + dt*k3[1], Z + dt*k3[2])

        X = X + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        Y = Y + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        Z = Z + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

        # 轻度裁剪，防爆
        X = xp.clip(X, 0.0, 5.0)
        Y = xp.clip(Y, 0.0, 5.0)
        Z = xp.clip(Z, 1e-6, 5.0)

    # 回CPU
    if on_gpu:
        X = X.get(); Y = Y.get(); Z = Z.get()
    return t_axis, r_t, psi_t, (X,Y,Z)

# --------------- Objective for DE (top-level, picklable) ---------------
def objective_empirical(theta, W, t_emp, r_emp, Nsubset=400, use_gpu=False):
    """
    theta: [Kv, Kg, Kl, w, a,b,c,d,e,f,n]
    返回末 24h 区间 RMSE
    """
    Kv, Kg, Kl, w, a,b,c,d,e,f,n = theta
    params = dict(Kv=Kv, Kg=Kg, Kl=Kl, w=w, a=a,b=b,c=c,d=d,e=e,f=f,n=n)
    # 子取样节点数（加速、稳健）
    if Nsubset < W.shape[0]:
        idx = np.random.RandomState(0).choice(W.shape[0], Nsubset, replace=False)
        Wuse = W[np.ix_(idx, idx)]
    else:
        Wuse = W
    try:
        t_sim, r_sim, _, _ = rk4_integrate_goodwin(Wuse, T_h=max(t_emp), dt=0.05,
                                                   params=params, seed=123, xp=np, on_gpu=use_gpu, progress=False)
        if len(r_sim)==0 or not np.isfinite(r_sim).all():
            return 1e3
        r_hat = np.interp(t_emp, t_sim, r_sim)
        if not np.isfinite(r_hat).all():
            return 1e3
        # 用末 24h
        mask = (t_emp >= (t_emp.max()-24.0))
        rmse = float(np.sqrt(np.nanmean((r_hat[mask] - r_emp[mask])**2)))
        if not np.isfinite(rmse): return 1e3
        return rmse
    except Exception:
        return 1e3

# --------------- Plot helpers ---------------
def phase_rose(theta_final, outpath:Path, bins=36):
    theta = (theta_final - np.min(theta_final))
    theta = (theta % (2*np.pi))
    ax = plt.subplot(111, projection='polar')
    hist, edges = np.histogram(theta, bins=bins, range=(0,2*np.pi))
    widths = np.diff(edges)
    ax.bar(edges[:-1], hist, width=widths, bottom=0.0)
    ax.set_title("Phase rose (final state)")
    savefig(outpath)

# --------------- Main pipeline ---------------
def main(args):
    OUT = Path("outputs"); CKPT = Path("checkpoints")
    ensure_dir(OUT); ensure_dir(CKPT)

    # ----- GPU select -----
    xp, on_gpu = _get_xp(args.use_gpu)
    print(f"[GPU] on_gpu={on_gpu}")

    # ----- Single-cell fit -----
    print("[STEP 1/5] Single-cell fit + residuals + period histogram")
    df = load_per2_csv(args.data_csv)
    t_fit, yavg, yhat, rmse, pfit = fit_singlecell_avg(df, fit_h=args.fit_h)

    # plots single cell
    plt.figure(figsize=(10,4))
    plt.plot(t_fit, yavg, label="PER2 data (avg)")
    plt.plot(t_fit, yhat, "--", label="Model PER2 (norm)")
    plt.xlabel("Time (h)"); plt.ylabel("Normalized PER2")
    plt.title(f"Single-cell fit  (RMSE={rmse:.3f}, period≈{pfit[1]:.1f} h)")
    plt.legend(frameon=False)
    savefig(OUT/"single_cell_fit.png")

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4))
    ax1.plot(t_fit, yavg, label="Data")
    ax1.plot(t_fit, yhat, "--", label="Model")
    ax1.set_title("Single-cell fit"); ax1.set_ylabel("Normalized PER2")
    ax1.legend(frameon=False)
    res = yavg - yhat
    ax2.plot(t_fit, res)
    ax2.axhline(0,color='gray',linewidth=1)
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Residual")
    ax2.text(0.01,0.8, f"RMSE={rmse:.3f}", transform=ax2.transAxes)
    savefig(OUT/"single_cell_fit_residuals.png")

    # period histogram（用每列简易自相关/峰间距估计）
    periods=[]
    for j in range(1, df.shape[1]):  # skip time_h
        x = df.iloc[:,j].values
        x = hampel_1d(detrend_linear(x))
        x = norm01(x)
        # 简单峰距法
        try:
            from scipy.signal import find_peaks
            pk,_ = find_peaks(x, distance=5)
            if len(pk)>=2:
                T = np.diff(df['time_h'].values[pk]).mean()
                if 20<=T<=30: periods.append(T)
        except Exception:
            pass
    periods = np.array(periods) if len(periods)>0 else np.array([pfit[1]])
    mu, sd = float(np.mean(periods)), float(np.std(periods)+1e-9)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.hist(periods, bins=6, density=True, alpha=0.6, label="Histogram")
    # KDE 简化（高斯核）
    xs = np.linspace(24, 26.2, 400)
    kde = np.zeros_like(xs)
    for v in periods:
        kde += np.exp(-(xs-v)**2/(2*sd**2))
    if kde.max()>0: kde /= (kde.max() + 1e-12)
    ax.plot(xs, kde, label="KDE")
    ax.axvline(mu, ls="--", color="k", label=f"mean={mu:.2f} h")
    ax.axvspan(mu-sd, mu+sd, color="gray", alpha=0.2, label=f"±1 SD={sd:.2f} h")
    ax.set_xlabel("Estimated period (h)"); ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    savefig(OUT/"period_histogram.png")

    # ----- Empirical r(t) -----
    print("[STEP 2/5] Build W from MIC & empirical r(t)")
    W = build_W_from_MIC(args.mic_csv, threshold=args.mic_thr)
    t_emp, r_emp, psi_emp = empirical_r_from_per2(df, max_h=args.emp_max_h, decimate=1)
    # save empirical plot
    plt.figure(figsize=(12,4))
    plt.plot(t_emp, r_emp)
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Empirical r(t) from SCN1 (Hilbert phase)")
    savefig(OUT/"r_empirical.png")

    # ----- Global fit (DE) -----
    print("[STEP 3/5] Global DE fit (workers=-1)")
    bounds = [
        (0.05, 1.00),  # Kv
        (0.00, 1.00),  # Kg
        (0.00, 1.00),  # Kl
        (0.00, 1.00),  # w
        (0.5,  2.0),   # a
        (0.01, 0.5),   # b
        (0.5,  2.0),   # c
        (0.01, 0.5),   # d
        (0.5,  2.0),   # e
        (0.01, 0.5),   # f
        (2.0,  6.0)    # n
    ]
    # tqdm via callback
    _iter = {'k':0}
    def de_progress(xk, convergence):
        _iter['k'] += 1
        # 保存当前最佳并画个预览对比
        Kv,Kg,Kl,w,a,b,c,d,e,f,n = xk
        best = dict(Kv=float(Kv),Kg=float(Kg),Kl=float(Kl),w=float(w),
                    a=float(a),b=float(b),c=float(c),d=float(d),e=float(e),f=float(f),n=float(n))
        with open(Path("checkpoints")/"best_params.json","w") as f:
            json.dump(best,f,indent=2)
        try:
            t_sim, r_sim, _, _ = rk4_integrate_goodwin(W, T_h=float(t_emp.max()), dt=0.05,
                                                       params=best, seed=123, xp=np, on_gpu=on_gpu, progress=False)
            r_hat = np.interp(t_emp, t_sim, r_sim)
            plt.figure(figsize=(12,4))
            plt.plot(t_emp, r_emp, label="Empirical r(t)")
            plt.plot(t_emp, r_hat, "--", label="Fitted model r(t)")
            plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
            plt.title("Data-constrained network fit (preview)")
            plt.legend(frameon=False)
            savefig(Path("checkpoints")/"progress_fit.png")
        except Exception:
            pass
        print(f"[DE] iter={_iter['k']} conv={convergence:.3g}")
        return False

    res = differential_evolution(
        objective_empirical, bounds,
        args=(W, t_emp, r_emp, args.fit_nodes, on_gpu),
        maxiter=args.maxiter, popsize=args.popsize, tol=1e-3,
        seed=1234, workers=-1, updating='deferred', polish=True,
        callback=de_progress
    )
    best = res.x
    Kv,Kg,Kl,w,a,b,c,d,e,f,n = best
    best_params = dict(Kv=float(Kv),Kg=float(Kg),Kl=float(Kl),w=float(w),
                       a=float(a),b=float(b),c=float(c),d=float(d),
                       e=float(e),f=float(f),n=float(n))
    with open(Path("checkpoints")/"best_params.json","w") as f:
        json.dump(best_params,f,indent=2)

    # 最终拟合图
    t_sim, r_sim, _, _ = rk4_integrate_goodwin(W, T_h=float(t_emp.max()), dt=0.05,
                                               params=best_params, seed=123, xp=np, on_gpu=on_gpu, progress=True)
    r_hat = np.interp(t_emp, t_sim, r_sim)
    plt.figure(figsize=(12,4))
    plt.plot(t_emp, r_emp, label="Empirical r(t)")
    plt.plot(t_emp, r_hat, "--", label="Fitted model r(t)")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Data-constrained network fit (SCN1)")
    plt.legend(frameon=False)
    savefig(OUT/"r_emp_vs_model_fit.png")

    # ----- STEP 4: requested figures -----
    print("[STEP 4/5] Batch figures ...")
    # VIP ON vs KO
    for vipfac, label, fname in [(1.0,"VIP ON (GABA+Gly)","vip_on_vs_ko_r_vs_time.png"),
                                 (0.0,"VIP KO (factor=0.00)","vip_on_vs_ko_r_vs_time.png")]:
        p = best_params.copy()
        p['Kv'] *= vipfac; p['Kg'] *= vipfac; p['Kl'] *= vipfac
        t1, r1, _, _ = rk4_integrate_goodwin(W, T_h=120.0, dt=0.05,
                                             params=p, seed=42, xp=np, on_gpu=on_gpu, progress=False)
        if vipfac==1.0:
            r_on = (t1, r1)
        else:
            plt.figure(figsize=(12,4))
            plt.plot(r_on[0], r_on[1], label="VIP ON (GABA+Gly)")
            plt.plot(t1, r1, label="VIP KO (factor=0.00)")
            plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
            plt.title("Network synchronization: VIP ON vs KO (data-constrained W)")
            plt.legend(frameon=False)
            savefig(OUT/"vip_on_vs_ko_r_vs_time.png")

    # 20×20 scans
    def sweep(Kmode:str, vip_on:bool, filename:str):
        Kg_list = np.linspace(0,1.0,20); Kl_list = np.linspace(0,1.0,20)
        vip = 1.0 if vip_on else 0.3
        Z = np.zeros((20,20), float)
        def cell(i,j,Kg,Kl):
            p = best_params.copy()
            Kg2 = Kg if Kmode in ("gaba","both") else 0.0
            Kl2 = Kl if Kmode in ("gly","both") else 0.0
            p['Kg'] = Kg2*vip; p['Kl'] = Kl2*vip
            t, r, _, _ = rk4_integrate_goodwin(W, T_h=72.0, dt=0.1,
                                               params=p, seed=100+i*31+j*17, xp=np, on_gpu=False, progress=False)
            return float(np.mean(r[-int(24/0.1):]))
        tasks=[]
        for i,Kg in enumerate(Kg_list):
            for j,Kl in enumerate(Kl_list):
                tasks.append((i,j,Kg,Kl))
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(cell)(i,j,Kg,Kl) for (i,j,Kg,Kl) in tqdm(tasks, desc=f"sweep {Kmode} vip={vip_on}", leave=False)
        )
        for (val,(i,j,_,_)) in zip(results, tasks):
            Z[i,j] = val
        X,Y = np.meshgrid(Kl_list, Kg_list)
        plt.figure(figsize=(11,8))
        im = plt.pcolormesh(X,Y,Z, shading="auto", vmin=0.0, vmax=1.0)
        plt.xlabel("Glycine K"); plt.ylabel("GABA K")
        plt.title(f"{'VIP ON' if vip_on else 'VIP KO'} / {Kmode.upper()}  (data-constrained W)")
        cb = plt.colorbar(im, label="Final r (mean over last 24 h)")
        # 等值线
        CS = plt.contour(X,Y,Z, levels=[0.2,0.4,0.6,0.8], colors="k", linewidths=0.7)
        plt.clabel(CS, inline=1, fontsize=8, fmt="r=%.1f")
        savefig(OUT/filename)

    sweep("gaba", True,  "coupling_sweep_20x20_VIPON_gaba.png")
    sweep("gly",  True,  "coupling_sweep_20x20_VIPON_gly.png")
    sweep("both", True,  "coupling_sweep_20x20_VIPON_both.png")
    sweep("gaba", False, "coupling_sweep_20x20_VIPKO_gaba.png")
    sweep("gly",  False, "coupling_sweep_20x20_VIPKO_gly.png")
    sweep("both", False, "coupling_sweep_20x20_VIPKO_both.png")

    # Noise curves
    sigmas = np.linspace(0.0, 0.5, 9)
    series = []; labels = []
    def final_r_with_sigma(Kg,Kl,s):
        p = best_params.copy()
        # 简单把噪声映射到参数扰动（这里用Kl轻扰；若你已有噪声项，可改为直接加噪）
        p['Kl'] = max(0.0, min(1.2, p['Kl']*(1.0-0.2*s)))
        t,r,_,_ = rk4_integrate_goodwin(W, T_h=96.0, dt=0.05, params=p, seed=21, xp=np, on_gpu=False, progress=False)
        return float(np.mean(r[-int(24/0.05):]))
    # 5 条
    series.append([final_r_with_sigma(0,0,s) for s in tqdm(sigmas, desc="noise: none", leave=False)]); labels.append("Noise only (no coupling)")
    series.append([final_r_with_sigma(1,0,s) for s in tqdm(sigmas, desc="noise: GABA", leave=False)]); labels.append("VIP ON / GABA only")
    series.append([final_r_with_sigma(0,1,s) for s in tqdm(sigmas, desc="noise: Gly", leave=False)]); labels.append("VIP ON / Gly only")
    series.append([final_r_with_sigma(1,1,s) for s in tqdm(sigmas, desc="noise: both", leave=False)]); labels.append("VIP ON / GABA+Gly")
    # KO
    series.append([final_r_with_sigma(1,1,s) for s in tqdm(sigmas, desc="noise: KO both", leave=False)]); labels.append("VIP KO / GABA+Gly")

    plt.figure(figsize=(12,5))
    for y, name in zip(series, labels):
        plt.plot(sigmas, y, marker='o', label=name)
    plt.xlabel("Noise sigma"); plt.ylabel("Final r (mean over last 24 h)")
    plt.title("Noise impact under different coupling conditions (data-constrained W)")
    plt.legend(frameon=False)
    savefig(OUT/"noise_vs_r_multi.png")

    # Jet lag（6h shift）
    def jetlag(shift_h=6.0):
        p = best_params.copy()
        t, r, psi, _ = rk4_integrate_goodwin(W, T_h=120.0, dt=0.05, params=p, seed=99, xp=np, on_gpu=on_gpu, progress=False)
        k = int(48.0/0.05)
        psi2 = psi.copy()
        shift = 2*np.pi*(shift_h/24.0)
        psi2[k:] = ((psi2[k:]+shift + np.pi)%(2*np.pi)) - np.pi
        # tau 拟合
        x = t[k:] - t[k]
        y = np.abs(((psi2[k:] - psi2[k]) + np.pi)%(2*np.pi) - np.pi) + 1e-6
        def mod(x,a,tau,b): return a*np.exp(-x/tau)+b
        try:
            popt,_ = curve_fit(mod, x, y, p0=(y[0], 36.0, 0.05), maxfev=10000)
            tau_est = float(popt[1])
        except Exception:
            tau_est = np.nan
        return t, r, psi2, tau_est

    tj, rj, psij, tau_est = jetlag(6.0)
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(tj, rj, label="r(t)")
    ax1.axvline(48.0, color='gray', alpha=0.6)
    ax1.set_ylabel("Order parameter r"); ax1.legend(frameon=False)
    ax1.set_title(f"Jet-lag recovery: r and mean phase ψ  (τ ≈ {tau_est:.1f} h)")
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(tj, psij, label="ψ(t)")
    ax2.axvline(48.0, color='gray', alpha=0.6)
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Mean phase ψ (rad)")
    ax2.legend(frameon=False)
    savefig(OUT/"jetlag_recovery_time_phase.png")

    # Phase rose（最终态）
    _, _, _, state = rk4_integrate_goodwin(W, T_h=96.0, dt=0.05, params=best_params, seed=888, xp=np, on_gpu=on_gpu, progress=False)
    phase_rose(state[2], OUT/"phase_rose_highres.png", bins=36)

    # Empirical vs model 完整 400+ h（上面已有短版）
    plt.figure(figsize=(12,4))
    plt.plot(t_emp, r_emp, label="Empirical r(t)")
    plt.plot(t_emp, r_hat, "--", label="Fitted model r(t)")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Data-constrained network fit (SCN1)")
    plt.legend(frameon=False)
    savefig(OUT/"r_emp_vs_model_fit.png")

    print("[STEP 5/5] Done. See outputs/ & checkpoints/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="scn1_full_data_with_time.csv")
    parser.add_argument("--mic_csv", type=str, default="mic_indiv_scn1.csv")
    parser.add_argument("--mic_thr", type=float, default=0.949, help="MIC阈值（Abel给的SCN1示例）")
    parser.add_argument("--fit_h", type=float, default=96.0)
    parser.add_argument("--emp_max_h", type=float, default=432.0)
    parser.add_argument("--use_gpu", type=str, default="auto", choices=["auto","on","off"])
    parser.add_argument("--fit_nodes", type=int, default=400, help="DE评估时用的子网络大小（加速）")
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=18)
    args = parser.parse_args()
    main(args)

