# scn_v4_goodwin.py
# Goodwin极限环 + 数据约束W + VIP/GABA/Gly耦合 + 并行DE拟合 + 进度条
# 输入: scn1_full_data_with_time.csv, mic_indiv_scn1.csv
# 输出: outputs_v4/ 下生成 14+ 图与参数表
import os, warnings, math, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import hilbert, detrend
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- I/O ----------------
DATA_TS = "scn1_full_data_with_time.csv"
DATA_MIC = "mic_indiv_scn1.csv"
OUT = Path("outputs_v4"); OUT.mkdir(exist_ok=True)

# ---------------- 工具函数 ----------------
def savefig(p, dpi=200):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.savefig(p.with_suffix(".pdf"))
    plt.close()

def norm01(x):
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-9: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def circ_mean(z):
    # z: complex unit vectors
    return np.angle(np.nanmean(z))

def circ_dist(a, b):
    # shortest angular distance
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return d

# ---------------- 数据读取/预处理 ----------------
def load_timeseries(file_csv):
    # 允许重复表头行，用第一列作为时间/或合成
    df_raw = pd.read_csv(file_csv, header=None)
    # 去掉重复header行
    mask_header = df_raw.apply(lambda r: (r == "time_h").any(), axis=1)
    df = df_raw.loc[~mask_header].reset_index(drop=True).copy()
    # 若第一列可数值→认为是时间，否则构造
    try:
        time_col = pd.to_numeric(df.iloc[:,0], errors="raise").values
        vals = df.iloc[:,1:].apply(pd.to_numeric, errors="coerce").values
        cols = ["time_h"] + [f"cell_{j}" for j in range(vals.shape[1])]
        df = pd.DataFrame(np.column_stack([time_col, vals]), columns=cols)
    except Exception:
        # 无时间列，构造 1h 采样
        vals = df.apply(pd.to_numeric, errors="coerce").values
        t = np.arange(len(vals))
        cols = ["time_h"] + [f"cell_{j}" for j in range(vals.shape[1])]
        df = pd.DataFrame(np.column_stack([t, vals]), columns=cols)
    # 去线性趋势 & 0-1 归一化
    for c in df.columns:
        if c == "time_h": continue
        v = df[c].values.astype(float)
        v = detrend(np.nan_to_num(v, nan=np.nanmean(v)))
        df[c] = norm01(v)
    return df

def empirical_phase_and_r(df):
    t = df["time_h"].values.astype(float)
    X = df.drop(columns=["time_h"]).values.astype(float)
    # 使用 Hilbert 相位
    theta = np.angle(hilbert(X, axis=0))
    z = np.exp(1j*theta)
    r = np.abs(np.mean(z, axis=1))
    return t, theta, r

def build_W_from_MIC(file_csv, thr=0.935, symmetrize=True, renorm=True):
    M = pd.read_csv(file_csv, header=None).values.astype(float)
    A = (M >= thr).astype(float)
    if symmetrize:
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    # 度归一化
    if renorm:
        deg = A.sum(axis=1, keepdims=True)
        deg[deg==0] = 1.0
        W = A / deg
    else:
        W = A
    return W

# ---------------- Goodwin 单细胞动力学 ----------------
# 变量: X=mRNA, Y=protein, Z=repressor
# dX/dt = a/(1+Z^n) - bX + Coupling_in
# dY/dt = cX - dY
# dZ/dt = eY - fZ + Noise
# 时间尺度: 通过 scale 调节使本征周期~26h
def goodwin_rhs(t, U, N, a,b,c,d,e,f,n, scale,
                W, Kv, Kg, Kl, vip_on=True,
                phase_mod=False, day_phase=0.0, sigma=0.02, rng=None):
    # U: (3N,) 向量
    X = U[0:N]; Y = U[N:2*N]; Z = U[2*N:3*N]
    # 内禀
    dX = a/(1.0 + Z**n) - b*X
    dY = c*X - d*Y
    dZ = e*Y - f*Z
    # 网络耦合（扩散/指向平均）：GABA/Gly on Z, VIP on X（全局+W）
    # VIP 全局 + 局部（通过W）
    if vip_on:
        X_mean = X.mean()
        dX += Kv * ((W @ X) - X) + 0.2*Kv*(X_mean - X)  # 局部+全局
    # GABA & Gly（均用W扩散，但符号/强度不同）
    dZ += Kg * ((W @ Z) - Z)    # GABA: 同步/去同步视强度而定
    dZ += Kl * ((W @ Z) - Z)    # Gly: 与GABA可叠加；若要抑制可给负值

    # 昼夜调制（选配）：耦合强度在“夜间”增强
    if phase_mod:
        omega0 = 2*np.pi/24.0
        w = 0.5*(1.0 + np.cos(omega0*t - day_phase))  # [0,1]
        # 夜间（w小）增强抑制→对Z耦合加权
        dZ += 0.5*(1-w) * (Kg+Kl) * ((W @ Z) - Z)

    # 噪声（在Z上添加，Ito）
    if sigma > 0 and rng is not None:
        dZ += sigma * rng.normal(0.0, 1.0, size=N)

    # 时间尺度统一缩放
    return np.concatenate([dX, dY, dZ]) / scale

def simulate_goodwin(t_span, dt, N, params, W,
                     Kv, Kg, Kl, vip_on=True, sigma=0.02, seed=0,
                     phase_mod=False, day_phase=0.0, discard_h=24.0):
    rng = np.random.default_rng(seed)
    a,b,c,d,e,f,n,scale = params
    t0, t1 = t_span
    t_eval = np.arange(t0, t1+1e-9, dt)
    # 初值：随机
    X0 = rng.uniform(0.1, 0.9, size=N)
    Y0 = rng.uniform(0.1, 0.9, size=N)
    Z0 = rng.uniform(0.1, 0.9, size=N)
    U0 = np.concatenate([X0,Y0,Z0])
    rhs = lambda t,U: goodwin_rhs(t,U,N,a,b,c,d,e,f,n,scale,W,Kv,Kg,Kl,
                                  vip_on,phase_mod,day_phase,sigma,rng)
    sol = solve_ivp(rhs, (t0, t1), U0, t_eval=t_eval, method="RK45",
                    rtol=1e-5, atol=1e-7)
    U = sol.y.T  # (T, 3N)
    X = U[:,0:N]; Y = U[:,N:2*N]; Z = U[:,2*N:3*N]
    # 用 Z 取相位（或 X、Y），用 Hilbert
    theta = np.angle(hilbert(Z, axis=0))
    r = np.abs(np.mean(np.exp(1j*theta), axis=1))
    # 丢弃初期过渡
    keep = t_eval >= (t0 + discard_h)
    return t_eval[keep], r[keep], theta[keep,:], (X[keep,:],Y[keep,:],Z[keep,:])

# ---------------- 目标函数（顶层，便于并行pickle） ----------------
def objective_empirical(x, W, t_emp, r_emp, N):
    # x: [Kv, Kg, Kl, sigma, scale]
    Kv, Kg, Kl, sigma, scale = x
    # 参数边界保护
    if not (0.0 <= Kv <= 2.0 and -2.0 <= Kg <= 2.0 and -2.0 <= Kl <= 2.0 and
            0.0 <= sigma <= 0.5 and 5.0 <= scale <= 60.0):
        return 1e3
    # 固定 Goodwin 内参（可后续纳入拟合）
    a,b,c,d,e,f,n = 2.0, 0.5, 1.0, 0.3, 1.0, 0.2, 3.0
    params = (a,b,c,d,e,f,n,scale)
    try:
        t_sim, r_sim, _, _ = simulate_goodwin(
            t_span=(t_emp[0], t_emp[-1]), dt=0.2, N=N, params=params, W=W,
            Kv=Kv, Kg=Kg, Kl=Kl, vip_on=True, sigma=sigma, seed=123,
            phase_mod=True, day_phase=0.0, discard_h=0.0)
        # 插值对齐
        r_model = np.interp(t_emp, t_sim, r_sim)
        w = np.ones_like(r_model)
        # 强化深谷结构匹配（经验中谷值明显）
        w[r_emp < (np.percentile(r_emp, 20))] = 2.0
        rmse = np.sqrt(np.nanmean(w*(r_model - r_emp)**2))
        # 防止“振幅根本不变”的假拟合
        penalty = 0.0
        if (r_model.max() - r_model.min()) < 0.05:
            penalty += 0.2
        return rmse + penalty
    except Exception:
        return 1e3

# ---------------- 画图（与v3保持一致的命名） ----------------
def plot_empirical_r(t_emp, r_emp):
    plt.figure(figsize=(12,4))
    plt.plot(t_emp, r_emp, lw=2)
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Empirical r(t) from SCN1 (Hilbert phase)")
    savefig(OUT/"r_empirical.png")

def plot_fit_vs_emp(t_emp, r_emp, r_model):
    plt.figure(figsize=(12,4))
    plt.plot(t_emp, r_emp, lw=2, label="Empirical r(t)")
    plt.plot(t_emp, r_model, lw=2, ls="--", label="Fitted model r(t)")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Data-constrained network fit (SCN1)")
    plt.legend(frameon=False)
    savefig(OUT/"r_emp_vs_model_fit.png")

def plot_noise_sweep(W, best, N):
    Kv, Kg, Kl, _, scale = best
    a,b,c,d,e,f,n = 2.0,0.5,1.0,0.3,1.0,0.2,3.0
    params = (a,b,c,d,e,f,n,scale)
    sigmas = np.linspace(0.0, 0.5, 11)
    series = {}
    for name, (kv,kg,kl,vip_on) in {
        "Noise only (no coupling)": (0.0,0.0,0.0,True),
        "VIP ON / GABA only": (Kv, Kg, 0.0, True),
        "VIP ON / Gly only": (Kv, 0.0, Kl, True),
        "VIP ON / GABA+Gly": (Kv, Kg, Kl, True),
        "VIP KO / GABA+Gly": (0.0, Kg, Kl, False),
    }.items():
        ys = []
        for s in sigmas:
            t, r, *_ = simulate_goodwin((0,120), 0.2, N, params, W, kv, kg, kl,
                                        vip_on=vip_on, sigma=s, seed=11)
            ys.append(r[-120:][-1] if len(r)>0 else np.nan)
        series[name] = np.array(ys)
    plt.figure(figsize=(12,5))
    for k,v in series.items():
        plt.plot(sigmas, v, marker='o', label=k)
    plt.xlabel("Noise sigma"); plt.ylabel("Final r (mean over last 24 h)")
    plt.title("Noise impact under different coupling conditions (data-constrained W)")
    plt.legend(frameon=False)
    savefig(OUT/"noise_vs_r_multi.png")

def plot_vip_on_off(W, best, N):
    Kv, Kg, Kl, sigma, scale = best
    a,b,c,d,e,f,n = 2.0,0.5,1.0,0.3,1.0,0.2,3.0
    params = (a,b,c,d,e,f,n,scale)
    t_on, r_on, *_ = simulate_goodwin((0,120), 0.2, N, params, W,
                                      Kv, Kg, Kl, True, sigma, seed=7)
    t_ko, r_ko, *_ = simulate_goodwin((0,120), 0.2, N, params, W,
                                      0.0, Kg, Kl, False, sigma, seed=7)
    plt.figure(figsize=(12,5))
    plt.plot(t_on, r_on, label="VIP ON (GABA+Gly)")
    plt.plot(t_ko, r_ko, label="VIP KO (factor=0.00)")
    plt.xlabel("Time (h)"); plt.ylabel("Order parameter r")
    plt.title("Network synchronization: VIP ON vs KO (data-constrained W)")
    plt.legend(frameon=False)
    savefig(OUT/"vip_on_vs_ko_r_vs_time.png")

def coupling_sweep_20x20(W, best, N, vip_on, mode, title, fname):
    Kv, Kg, Kl, sigma, scale = best
    a,b,c,d,e,f,n = 2.0,0.5,1.0,0.3,1.0,0.2,3.0
    params = (a,b,c,d,e,f,n,scale)
    Gg = np.linspace(0, 1.0, 20)
    Gl = np.linspace(0, 1.0, 20)
    R = np.zeros((20,20))
    for i, kg in enumerate(tqdm(Gg, desc=f"sweep {fname}", leave=False)):
        for j, kl in enumerate(Gl):
            kv = Kv if vip_on else 0.0
            kg_ = kg if mode in ("gaba","both") else 0.0
            kl_ = kl if mode in ("gly","both") else 0.0
            t, r, *_ = simulate_goodwin((0, 96), 0.2, N, params, W,
                                        kv, kg_, kl_, vip_on, sigma=0.05, seed=100+i*20+j)
            R[i,j] = np.mean(r[-120:]) if len(r)>120 else (r[-1] if len(r)>0 else np.nan)
    X,Y = np.meshgrid(Gl, Gg)
    plt.figure(figsize=(10,7.5))
    im = plt.pcolormesh(X,Y,R, shading="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, label="Final r (mean over last 24 h)")
    plt.xlabel("Glycine K"); plt.ylabel("GABA K")
    plt.title(f"{title}  (data-constrained W)")
    # 等值线
    try:
        CS = plt.contour(X,Y,R, levels=[0.2,0.4,0.6,0.8], colors="k", linewidths=0.8)
        plt.clabel(CS, inline=1, fontsize=8, fmt="r=%.1f")
    except Exception:
        pass
    savefig(OUT/fname)

def jetlag_demo(W, best, N, shift_h=6.0):
    Kv, Kg, Kl, sigma, scale = best
    a,b,c,d,e,f,n = 2.0,0.5,1.0,0.3,1.0,0.2,3.0
    params = (a,b,c,d,e,f,n,scale)
    # 先模拟一段达到稳态
    t1, r1, th1, _ = simulate_goodwin((0,48), 0.2, N, params, W, Kv, Kg, Kl, True, sigma, seed=66)
    # 注入“时差”：相位整体平移
    th_shift = th1.copy()
    k0 = len(t1)-1
    shift = 2*np.pi * (shift_h/24.0)
    th_shift[k0:,:] = ((th_shift[k0:,:] + shift + np.pi)%(2*np.pi)) - np.pi
    # 继续模拟
    t2, r2, th2, _ = simulate_goodwin((t1[-1], 120), 0.2, N, params, W, Kv, Kg, Kl, True, sigma, seed=66)
    t = np.concatenate([t1, t2])
    r = np.concatenate([r1, r2])
    # 计算群体平均相位轨迹 ψ(t)
    psi1 = np.angle(np.mean(np.exp(1j*th1), axis=1))
    psi2 = np.angle(np.mean(np.exp(1j*th2), axis=1))
    psi = np.concatenate([psi1, psi2])
    # 画图
    plt.figure(figsize=(12,7))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, r, label="r(t)"); ax1.axvline(t1[-1], color='0.5')
    ax1.set_ylabel("Order parameter r"); ax1.legend(frameon=False)
    ax1.set_title("Jet-lag recovery: r and mean phase ψ")
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(t, psi, label="ψ(t)")
    ax2.axvline(t1[-1], color='0.5')
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Mean phase ψ (rad)")
    ax2.legend(frameon=False)
    savefig(OUT/"jetlag_recovery_time_phase.png")

def phase_rose(theta_final, fname, bins=36):
    ang = np.angle(np.exp(1j*theta_final))  # ensure [-pi,pi]
    counts, edges = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:])/2
    width = edges[1]-edges[0]
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(centers, counts, width=width, bottom=0.0)
    ax.set_title("Phase rose (final state)")
    savefig(OUT/fname)

# ---------------- 主流程 ----------------
def main():
    print("[STEP 1] load timeseries & empirical r(t)")
    df = load_timeseries(DATA_TS)
    t_emp, theta_emp, r_emp = empirical_phase_and_r(df)
    plot_empirical_r(t_emp, r_emp)

    print("[STEP 2] build W from MIC + threshold")
    W = build_W_from_MIC(DATA_MIC, thr=0.949, symmetrize=True, renorm=True)
    N = W.shape[0]
    print(f"[W] N={N}, edges={(W>0).sum()}, dens={W.mean():.4f}")

    print("[STEP 3] global DE fit (workers=-1)")
    bounds = [(0.0, 2.0),   # Kv
              (-1.0, 2.0),  # Kg
              (-1.0, 2.0),  # Kl
              (0.0, 0.4),   # sigma
              (8.0, 40.0)]  # scale
    # 注意：Windows 并行需要顶层函数
    res = differential_evolution(
        objective_empirical, bounds,
        args=(W, t_emp, r_emp, N),
        maxiter=40, popsize=16, tol=1e-3, polish=True,
        seed=1234, workers=-1, updating='deferred'
    )
    best = res.x
    print("[BEST]", best.tolist())

    # 回代并出图
    print("[STEP 4] simulate with best & make figures")
    Kv, Kg, Kl, sigma, scale = best
    a,b,c,d,e,f,n = 2.0,0.5,1.0,0.3,1.0,0.2,3.0
    params = (a,b,c,d,e,f,n,scale)
    t_sim, r_sim, theta_sim, state = simulate_goodwin(
        (t_emp[0], t_emp[-1]), 0.2, N, params, W,
        Kv, Kg, Kl, True, sigma, seed=777, phase_mod=True, day_phase=0.0, discard_h=0.0
    )
    r_model = np.interp(t_emp, t_sim, r_sim)
    plot_fit_vs_emp(t_emp, r_emp, r_model)

    # 其他图
    plot_vip_on_off(W, best, N)
    plot_noise_sweep(W, best, N)
    coupling_sweep_20x20(W, best, N, True,  "gaba", "VIP ON / GABA only",  "coupling_sweep_20x20_VIPON_gaba.png")
    coupling_sweep_20x20(W, best, N, True,  "gly",  "VIP ON / Gly only",   "coupling_sweep_20x20_VIPON_gly.png")
    coupling_sweep_20x20(W, best, N, True,  "both", "VIP ON / GABA+Gly",   "coupling_sweep_20x20_VIPON_both.png")
    coupling_sweep_20x20(W, best, N, False, "gaba", "VIP KO / GABA only",  "coupling_sweep_20x20_VIPKO_gaba.png")
    coupling_sweep_20x20(W, best, N, False, "gly",  "VIP KO / Gly only",   "coupling_sweep_20x20_VIPKO_gly.png")
    coupling_sweep_20x20(W, best, N, False, "both", "VIP KO / GABA+Gly",   "coupling_sweep_20x20_VIPKO_both.png")

    # Jet-lag
    jetlag_demo(W, best, N)

    # 期末相位玫瑰
    theta_final = theta_sim[-1,:]
    phase_rose(theta_final, "phase_rose_highres.png", bins=36)

    # 保存参数
    param_row = {"Kv":Kv, "Kg":Kg, "Kl":Kl, "sigma":sigma, "scale":scale,
                 "N":N, "MIC_thr":0.949, "obj":float(res.fun)}
    pd.DataFrame([param_row]).to_csv(OUT/"Model_params_v4.csv", index=False)
    with open(OUT/"README_v4.md","w",encoding="utf-8") as f:
        f.write("# SCN v4 (Goodwin) results\n")
        f.write(json.dumps(param_row, indent=2))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

