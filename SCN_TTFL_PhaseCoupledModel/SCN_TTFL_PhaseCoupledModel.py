# -*- coding: utf-8 -*-
"""
SCN circadian model (RP-consistent, final integrated) — NO AUTOSAVE
Modules:
  1) Single-cell TTFL ODE (~24 h) .................... Eq.(1)–(4)
  2) Phase network + SDE (VIP/GABA/Gly) .............. Eq.(5)–(8),(10)–(11),(13)
     + TTFL-driven slow modulation of natural frequency (enhancement)
  3) Kuramoto R(t) ................................... Eq.(9) (full 0–72 h)
  4) Heatmaps R̄(g_GABA, g_Gly) (last 12 h avg) ...... Eq.(16), VIP ON/KO
  5) Phase variance & bimodal demo ................... Eq.(14)–(15)

All figures show on screen (you can save manually).
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.integrate import solve_ivp

# =========================
# Utilities
# =========================
def estimate_period(t, x):
    x = x - np.mean(x)
    if np.allclose(x, 0.0): return np.nan
    dt = np.median(np.diff(t))
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    ac[0] = 0.0
    k = np.argmax(ac)
    return k*dt if k > 0 else np.nan

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# =========================
# (1) Single-cell TTFL ODE (~24 h) — Eq.(1)–(4)
# =========================
def ttfl_rhs_core(t, y, p):
    # y=[M,Pc,Pn]
    M, Pc, Pn = y
    ts = p["ts"]
    dM  = p["v_s"]/(1.0 + (Pn/p["K_I"])**p["n_hill"]) - p["v_m"]*M/(p["K_m"] + M)
    dPc = p["k_s"]*M - p["V_d"]*Pc/(p["K_d"] + Pc) - p["k1"]*Pc + p["k2"]*Pn
    dPn = p["k1"]*Pc - p["k2"]*Pn - p["V_n"]*Pn/(p["K_n"] + Pn)
    return ts*np.array([dM, dPc, dPn])

def simulate_ttfl_core(params, T=200.0, y0=None):
    if y0 is None: y0=np.array([0.6,0.5,0.4])
    sol = solve_ivp(lambda t,y: ttfl_rhs_core(t,y,params),
                    (0.0,T), y0, max_step=0.05, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y

def calibrate_timescale_24h(BASE, target=24.0, T=240.0, discard=80.0,
                            ts_lo=0.2, ts_hi=5.0, max_iter=14, tol=0.8):
    p = dict(BASE)
    def measure(ts):
        p["ts"]=ts
        t,Y = simulate_ttfl_core(p,T=T)
        mask = t>=discard
        return estimate_period(t[mask], Y[0][mask])
    lo,hi = ts_lo, ts_hi
    best_ts, best_err = None, 1e9
    for _ in range(max_iter):
        mid = np.sqrt(lo*hi)
        P = measure(mid)
        if np.isnan(P): hi=mid; continue
        err = abs(P-target)
        if err<best_err: best_ts, best_err = mid, err
        if P>target: lo=mid
        else: hi=mid
        if err<=tol: break
    p["ts"] = best_ts if best_ts is not None else mid
    if best_err>2.0:  # fallback to a validated timescale
        p["ts"] = 0.83
    return p, p["ts"], best_err

def demo_single_cell_100h():
    BASE = dict(
        v_s=1.0, K_I=1.0, n_hill=4.0,
        v_m=0.7, K_m=0.5,
        k_s=0.9, V_d=1.5, K_d=0.2,
        k1=0.7, k2=0.35,
        V_n=0.6, K_n=0.2,
        ts=1.0
    )
    CAL, ts_opt, err = calibrate_timescale_24h(BASE)
    print(f"[TTFL] Calibrated ts={CAL['ts']:.3f} (raw {ts_opt:.3f}, err≈{err:.2f}h)")
    t, Y = simulate_ttfl_core(CAL, T=100.0)
    M,Pc,Pn = Y
    r  = 5.0 + 12.0*sigmoid(1.0 - 1.2*Pn + 0.2*M)   # downstream demo
    Ca = 0.25 + 0.15*sigmoid(r - 6.0)
    fig,axs = plt.subplots(5,1,figsize=(10,8),sharex=True)
    for ax,s,lab,col in zip(axs,[M,Pc,Pn,Ca,r],['M','P_c','P_n','[Ca$^{2+}$]','r (firing)'],['C0','C1','C2','C3','C4']):
        ax.plot(t,s,color=col); ax.set_ylabel(lab); ax.grid(alpha=.3)
    for x in [24,48,72,96]: axs[-1].axvline(x,ls='--',color='k',alpha=.35)
    axs[-1].set_xlabel("Time (h)")
    plt.suptitle("Single-cell TTFL ~24 h (display 100 h)")
    plt.tight_layout(); plt.show()
    return CAL

# =========================
# TTFL-driven slow modulation helpers (增强块)
# =========================
def build_ttfl_drive_from_singlecell(TTFL_CAL, T=72.0, dt=0.01,
                                     invert=True, zscore=True):
    """Return slow drive D(t) from single-cell Pn(t), length = int(T/dt)."""
    T_sim = max(T+48.0, 120.0)
    t_ttfl, Y = simulate_ttfl_core(TTFL_CAL, T=T_sim)
    Pn = Y[2]
    t_grid = np.arange(0.0, T, dt)
    t0 = t_ttfl[-1] - T
    D = np.interp(t0 + t_grid, t_ttfl, Pn)
    if zscore:
        D = (D - D.mean())/(D.std()+1e-9)
    if invert:
        D = -D  # night-high Pn -> lower freq
    return D

def build_sinusoidal_drive(T=72.0, dt=0.01, omega0=2*np.pi/24.0, phase0=0.0):
    t_grid = np.arange(0.0, T, dt)
    return np.sin(omega0*t_grid + phase0)

# =========================
# (2) Phase network + SDE — Eq.(5)–(8),(10)–(11),(13)
# 启用 TTFL→ω(t) 的慢调制（eps_ttfl, drive_mode, shared_drive）
# =========================
def run_phase_network_demo(N=100, T=72.0, dt=0.01, p_conn=0.15, seed=7,
                           K_VIP=0.10, K_GABA=0.08, K_GLY=0.06,
                           psi_gaba=0.6*np.pi, psi_gly=0.5*np.pi,
                           sigma_phi=0.025, vip_on=True,
                           include_vip=True, include_gaba=True, include_gly=True,
                           eps_ttfl=0.06, drive_mode="ttfl",
                           TTFL_CAL=None, shared_drive=None):
    """
    dφ_i = ω_i(t) + (K/deg) Σ sin(...) + σ dW
    ω_i(t) = ω0 * heter * [1 + eps_ttfl * D_i(t)]
    D_i(t): TTFL-driven (from Pn) or sinusoidal proxy, with per-cell phase offset.
    """
    rng = default_rng(seed)
    steps = int(T/dt)

    # Graph
    A = (rng.random((N,N)) < p_conn).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1); deg[deg==0]=1.0

    omega0 = 2*np.pi/24.0

    # Build slow drive D(t)
    if shared_drive is not None:
        D_base = np.array(shared_drive, dtype=float)
        if len(D_base) != steps:
            raise ValueError("shared_drive length mismatch")
    else:
        if drive_mode == "ttfl":
            if TTFL_CAL is None:
                raise ValueError("drive_mode='ttfl' requires TTFL_CAL")
            D_base = build_ttfl_drive_from_singlecell(TTFL_CAL, T=T, dt=dt,
                                                      invert=True, zscore=True)
        else:
            D_base = build_sinusoidal_drive(T=T, dt=dt, omega0=omega0, phase0=0.0)

    # Per-cell offset θ_i (diversity)
    rng = default_rng(seed)
    theta_i = rng.uniform(-0.6*np.pi, 0.6*np.pi, N)

    # Static heterogeneity ~ ±3%
    heter = 1.0 + 0.03*rng.standard_normal(N)

    # Initial phases
    phi = rng.uniform(-np.pi, np.pi, N)

    # Coupling switches
    Kvip = K_VIP if (vip_on and include_vip) else 0.0
    Kg   = K_GABA if include_gaba else 0.0
    Kl   = K_GLY  if include_gly  else 0.0

    t_series = np.zeros(steps); R_series = np.zeros(steps)

    for k in range(steps):
        # Per-cell slow drive with offset
        D_t = np.sin(np.arcsin(np.clip(D_base[k], -1, 1)) + theta_i)
        omega_t = omega0 * heter * (1.0 + eps_ttfl * D_t)

        sphi = np.sin(phi); cphi = np.cos(phi)
        S0 = A @ sphi; C0 = A @ cphi
        sin_diff = S0*cphi - C0*sphi
        cos_diff = C0*cphi + S0*sphi
        sin_gaba = sin_diff*np.cos(psi_gaba) + cos_diff*np.sin(psi_gaba)
        sin_gly  = sin_diff*np.cos(psi_gly ) + cos_diff*np.sin(psi_gly )

        dphi = omega_t \
             + (Kvip/deg) * sin_diff \
             + (Kg  /deg) * sin_gaba \
             + (Kl  /deg) * sin_gly \
             + sigma_phi * rng.standard_normal(N)/np.sqrt(dt)

        phi = (phi + dphi*dt + np.pi) % (2*np.pi) - np.pi

        z = np.exp(1j*phi)
        R_series[k] = np.abs(np.mean(z))     # Eq.(9)
        t_series[k] = (k+1)*dt

    return dict(t=t_series, R=R_series)

def plot_R_series_full(out, title):
    plt.figure(figsize=(9,3.8))
    plt.plot(out["t"], out["R"], 'k')
    plt.ylim(0,1.05); plt.xlim(0, out["t"][-1])
    plt.xlabel("Time (h)"); plt.ylabel("R(t)")
    plt.title(title); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()

# =========================
# (3) Heatmaps R̄(g_GABA, g_Gly) — Eq.(16), last 12 h avg
# =========================
def heatmap_R(N=100, T=72.0, dt=0.01, p_conn=0.15, seed=7,
              K_VIP=0.10, K_GABA=0.08, K_GLY=0.06,
              psi_gaba=0.6*np.pi, psi_gly=0.5*np.pi,
              sigma_phi=0.025, vip_on=True,
              eps_ttfl=0.06, drive_mode="ttfl",
              TTFL_CAL=None, shared_drive=None):
    gG_list = np.linspace(0.0, 0.30, 7)
    gL_list = np.linspace(0.0, 0.30, 7)
    R_avg = np.zeros((len(gG_list), len(gL_list)))
    for i,gG in enumerate(gG_list):
        for j,gL in enumerate(gL_list):
            out = run_phase_network_demo(
                N=N, T=T, dt=dt, p_conn=p_conn, seed=seed + i*11 + j,
                K_VIP=K_VIP, K_GABA=gG, K_GLY=gL,
                psi_gaba=psi_gaba, psi_gly=psi_gly, sigma_phi=sigma_phi,
                vip_on=vip_on, include_vip=True, include_gaba=True, include_gly=True,
                eps_ttfl=eps_ttfl, drive_mode=drive_mode,
                TTFL_CAL=TTFL_CAL, shared_drive=shared_drive
            )
            mask = out["t"] >= (T - 12.0)
            R_avg[i,j] = float(np.mean(out["R"][mask]))
    plt.figure(figsize=(6.6,5.2))
    im = plt.pcolormesh(gL_list, gG_list, R_avg, shading='auto')
    plt.colorbar(im, label="Average R (last 12 h)")
    plt.xlabel("g_Gly"); plt.ylabel("g_GABA")
    plt.title(f"R(g_GABA, g_Gly) | VIP {'ON' if vip_on else 'KO'} (Phase+TTFL drive)")
    plt.tight_layout(); plt.show()

# =========================
# (4) Phase variance & bimodal demo — Eq.(14)–(15)
# =========================
def phase_variance_bimodal_demo():
    rng = default_rng(0); N=3000
    phi_u = np.mod(rng.normal(0.0,0.6,N)+np.pi,2*np.pi)-np.pi
    mix = rng.random(N)<0.5
    phi_b = np.where(mix, rng.normal(0.0,0.35,N), rng.normal(np.pi,0.35,N))
    phi_b = np.mod(phi_b+np.pi,2*np.pi)-np.pi
    def circ_var(phi): return 1 - np.abs(np.mean(np.exp(1j*phi)))
    vu,vb = circ_var(phi_u), circ_var(phi_b)
    print(f"[Phase stats] Circular variance — unimodal: {vu:.3f} | bimodal: {vb:.3f}")
    fig,ax = plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    ax[0].hist(phi_u,bins=60,density=True); ax[0].set_title("Unimodal P(φ)")
    ax[1].hist(phi_b,bins=60,density=True); ax[1].set_title("Bimodal P(φ) (Eq.15)")
    for a in ax: a.set_xlabel("φ"); a.grid(alpha=.3)
    ax[0].set_ylabel("Density")
    plt.tight_layout(); plt.show()

# =========================
# (5) Main
# =========================
if __name__ == "__main__":
    # ---- Single-cell TTFL (~24 h) for RP figure ----
    TTFL_CAL = demo_single_cell_100h()

    # ---- Build shared TTFL drive (72 h) for fair comparison ----
    D_shared = build_ttfl_drive_from_singlecell(TTFL_CAL, T=72.0, dt=0.01, invert=True, zscore=True)

    # ---- Phase network configs (可按需微调) ----
    cfg = dict(N=100, T=72.0, dt=0.01, p_conn=0.15, seed=7,
               K_VIP=0.10, K_GABA=0.08, K_GLY=0.06,
               psi_gaba=0.6*np.pi, psi_gly=0.5*np.pi, sigma_phi=0.025,
               eps_ttfl=0.06, drive_mode="ttfl", TTFL_CAL=TTFL_CAL, shared_drive=D_shared)

    # ① Noise only (no coupling)
    out0 = run_phase_network_demo(**cfg, include_vip=False, include_gaba=False, include_gly=False)
    plot_R_series_full(out0, "R(t) — noise only (TTFL-driven ω(t) enabled)")

    # ② GABA+Gly only
    out1 = run_phase_network_demo(**cfg, include_vip=False, include_gaba=True, include_gly=True)
    plot_R_series_full(out1, "R(t) — GABA+Gly only (TTFL-driven ω(t))")

    # ③ VIP ON (with GABA+Gly)
    out2 = run_phase_network_demo(**cfg, vip_on=True, include_vip=True, include_gaba=True, include_gly=True)
    plot_R_series_full(out2, "R(t) — VIP ON (Phase + TTFL-driven ω(t))")

    # ④ VIP KO (with GABA+Gly) — expect lower R
    out3 = run_phase_network_demo(**cfg, vip_on=False, include_vip=True, include_gaba=True, include_gly=True)
    plot_R_series_full(out3, "R(t) — VIP KO (Phase + TTFL-driven ω(t))")

    # ---- Heatmaps (last-12h average) ----
    heatmap_R(vip_on=True,  **cfg)
    heatmap_R(vip_on=False, **cfg)

    # ---- Phase variance & bimodal demonstration ----
    phase_variance_bimodal_demo()

    print("[Done] All figures shown (manual save).")

