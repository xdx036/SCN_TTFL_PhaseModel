# SCN_TTFL_PhaseModel  
*A preliminary, proof-of-concept simulation of TTFL-driven phase network synchronization in the SCN*

---

## Overview  
This repository contains an **early-stage exploratory model** developed to conceptually validate the research framework proposed for circadian rhythm synchronization in the mammalian suprachiasmatic nucleus (SCN). The model follows the mechanistic hierarchy expected in SCN physiology and is structured to match the multi-layered logic described in my research proposal under the supervision of **Prof. Koichiro Uriu (Tokyo Tech).**

 **Important note:**  
> This is a **proof-of-concept prototype** built to demonstrate how TTFL oscillations can be linked to neuronal phase synchronization via neurochemical coupling (VIP, GABA, glycine). Biological calibration (e.g., parameter fitting, receptor kinetics) is intentionally simplified here and will be refined under academic guidance.

---

## Purpose of this model  
To confirm that the RP’s conceptual hierarchy is computationally realizable.  
To explore how TTFL oscillations may modulate phase-based SCN models.  
To test whether VIP-KO effects and inhibitory coupling dynamics can qualitatively reduce synchronization.  
To prepare a computational foundation for further refinement with real SCN parameters under laboratory supervision.

---

##  Model hierarchy (aligned with RP structure)

| Layer | Mechanism | Equation Group | Model Role |
|-------|----------|---------------|----------|
| Molecular | TTFL (Per/Cry loop) | Eq.(1–4) | Generates intrinsic ~24h rhythm |
| Cellular | Firing/Ca²⁺ proxy | Derived downstream | Links TTFL to excitability |
| Network | Phase oscillation + SDE | Eq.(5–8),(10–11),(13) | Neurons coupled by VIP, GABA, Gly |
| Synchrony metric | Kuramoto order parameter | Eq.(9) | Quantifies population coherence |
| Parameter map | R̄(g_GABA, g_Gly) | Eq.(16) | Compares VIP ON vs KO regimes |
| Phase statistics | Variance / bimodality | Eq.(14–15) | Shows desynchronization effects |

---

## Implementation Highlights  
TTFL ODE is calibrated to ~24h automatically.  
A sinusoid-like phase network is augmented with TTFL-driven slow frequency modulation.  
VIP promotes synchronization; GABA/Gly introduce inhibitory phase shifts.  
Noise (σ) introduces biological stochasticity.  
VIP-KO leads to reduced or unstable synchronization.  
Heatmaps illustrate coupling-dependent synchrony landscapes.  
Phase bimodality is demonstrated as a potential SCN cluster state.

---

##  Example qualitative outcomes  
| Condition | Expected behavior |
|----------|------------------|
| Noise only | R(t) remains low & fluctuating |
| GABA/Gly only | May suppress phase-locking |
| VIP ON | R(t) rises steadily → high coherence |
| VIP KO | Partial/weak synchrony or drifting |
| Strong inhibition | Possible phase-splitting or cluster formation |

(*Exact values depend on parameters; calibrations will be refined later*)

---

## How to run  
### 1️ Install dependencies:
pip install numpy scipy matplotlib
### 2 Run the main simulation:
python SCN_TTFL_PhaseCoupledModel.py

Figures are displayed sequentially (no auto-save).
Parameter blocks are clearly marked and modifiable.

## Future development plan (after supervision begins)

Replace generic parameters with experimentally validated SCN data
Introduce VIP receptor dynamics (e.g. VPAC2 binding)
Incorporate Ca²⁺-dependent TTFL feedback loops
Validate Gly role timing (day/night dependent)
Extend to PDE population density formulation
Possibly derive Fokker-Planck approximation for phase dynamics

## Acknowledgment

This model was developed by Liu Xuan as a conceptual preparation for future research under the mentorship of Prof. Koichiro Uriu, Department of Life Science and Technology, Tokyo Institute of Technology.
I look forward to refining this framework further with formal scientific guidance.

## Citation (if referenced academically)

Liu, X. (2025). SCN_TTFL_PhaseModel: A preliminary prototype for TTFL-to-phase coupling in SCN synchronization. GitHub repository.
https://github.com/xdx036/SCN_TTFL_PhaseModel
