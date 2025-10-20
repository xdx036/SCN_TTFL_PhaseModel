# SCN_TTFL_PhaseModel
Mathematical modeling of TTFL-SCN phase network with VIP/GABA/Gly coupling

Overview

This repository contains a complete implementation of a circadian network model inspired by the research direction of Uriu Laboratory (Tokyo Tech), integrating the transcription–translation feedback loop (TTFL) with neuronal phase synchronization via VIP, GABA, and glycine signaling.

The model reproduces hierarchical dynamics of circadian organization:

Molecular level – 24 h TTFL oscillation (Per/Cry loop).
Cellular level – TTFL-modulated firing and Ca²⁺ rhythmicity.
Network level – Phase coupling through neuropeptides and neurotransmitters.
This framework allows exploring how VIP knockout, GABA/Gly coupling, and stochastic noise affect global SCN synchrony.

Model structure

The code follows the same modular structure as the author’s research proposal:
Single-cell TTFL ODEs — Eq.(1–4)
Generates intrinsic ~24 h oscillations in gene expression.
Calibrated automatically to maintain circadian period.
Phase oscillator network with SDE noise — Eq.(5–8), (10–11), (13)
Each neuron’s intrinsic frequency is slowly modulated by its TTFL state.
Coupling through VIP (synchronizing), GABA/Gly (inhibitory with phase shifts).
Kuramoto order parameter R(t) — Eq.(9)
Quantifies network synchrony over time.
Parameter map R̄(g_GABA, g_Gly) — Eq.(16)
Heatmap comparing VIP ON vs KO conditions.
Phase variance & bimodal state demo — Eq.(14–15)
Illustrates uni- vs bi-modal phase distributions.

Environment

Python ≥ 3.9
Required packages:
pip install numpy scipy matplotlib

How to run

Clone or download the repository:
git clone https://github.com/yourname/SCN_TTFL_PhaseModel.git
cd SCN_TTFL_PhaseModel

Run the main script:
python SCN_TTFL_PhaseCoupledModel.py

Figures will appear sequentially (no auto-save).
You may manually save them from the plot window.

Simulation contents
Step	Description	Related Eq.	Figure
1	Single-cell TTFL oscillation (~24 h)	(1–4)	M, Pₙ, Ca²⁺ traces
2	Phase network with SDE noise	(5–8),(10–11),(13)	R(t) curves
3	Kuramoto order parameter R(t)	(9)	synchrony dynamics
4	Heatmap R̄(g_GABA, g_Gly)	(16)	VIP ON vs KO comparison
5	Phase variance & bimodal demo	(14–15)	distribution plots

Example results

VIP ON → high R(t) (~0.8–1.0), stable synchronization.
VIP KO → lower R(t) (~0.3–0.5), partial or clustered rhythms.
Moderate GABA/Gly coupling → maximal coherence; too strong → anti-phase clusters.

Interpretation

This model supports the concept that circadian coherence emerges from TTFL-driven modulation of neural excitability and reciprocal network coupling.
It provides a computational testbed for hypotheses on SCN synchronization, noise resilience, and neurochemical compensation (e.g. glycinergic support under VIP deficiency).

Acknowledgment

Developed by Liu Xuan,
under guidance and scientific discussion with Prof. Koichiro Uriu,
Department of Life Science and Technology, Tokyo Institute of Technology.

Citation

If you refer to or build upon this model, please cite as:

Liu Xuan (2025). SCN_TTFL_PhaseModel: A TTFL–Phase Coupled Network Simulation for Circadian Synchronization. GitHub repository, Tokyo Tech.
