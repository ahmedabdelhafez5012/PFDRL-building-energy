# PFDRL: federated deep reinforcement learning for building energy management

Code, analysis scripts and aggregated results for the M.Sc. thesis
*"Energy Management in Smart Buildings with Privacy Preservation Using Federated Deep Reinforcement Learning"*
(Ahmed A. A. Abdelgawad, Capital University (formerly Helwan University), Faculty of Engineering, Department of Smart Electrical Grids Engineering, 2026).

[![DOI](https://zenodo.org/badge/1447626244.svg)](https://doi.org/10.5281/zenodo.23991621)

## Status

Research code, not a product. The main finding is negative: in this simplified environment, federation did not reduce cost relative to independent training. The original campaign had implementation issues disclosed in the thesis (Limitation L10: the FedProx arm applied no proximal term and only critics were aggregated). The corrective experiments (`exp_lean.py`, `exp_multi.py`) supersede it for the federation questions.

## Contents

### Core modules
- `environment.py`: building energy environment (15-min steps, seven-variable state, HVAC and battery actions)
- `ddpg_agent.py`: DDPG agent
- `federated_learning.py`: aggregation and noise injection
- `xai_module.py`: SHAP and LIME explanations of the battery action
- `data_loader.py`, `data_loader_office.py`, `data_loader_commercial.py`, `data_loader_mixeduse.py`: archetype data loaders

### Baselines
- `baseline_b1_rule_based.py`, `baseline_b2_mpc.py`, `compare_baselines.py`

### Original campaign (H1-H4, B1-B4)
- `main.py` (early smoke test), `main_real_federation.py`
- `scenario_s3_mu_sweep.py`, `scenario_s4_epsilon_sweep.py`, `lambda_sweep.py` (superseded)
- `multi_seed_h1_test.py`, `multi_seed_h2_test.py`, `multi_seed_h3_latency_test.py`, `multi_seed_h4_test.py`
- `00_power_analysis.py`, `01_full_ablation_b1_b4.py`, `02_extended_n_multiseed.py`, `03_multiseed_s3_s4.py`, `04_merge_and_generate_tables.py`
- `05_reward_normalisation_ablation.py`, `06_capacity_interference_pilot.py`, `07_weighted_sampling_pilot.py` (reward-scale diagnosis)
- `09a_inspect_residential_sources.py`, `09b_build_residential_expansion.py`, `10_residential_scale_federation.py` (residential scale-up)
- `15_b3_vs_b2.py` (paired B3 vs B2 test, N=30)
- `13_resolve_open_items.py`, `14_check_residential.py`, `diagnose_office_actor.py` (audit utilities)
- `extract_s5_demo_decisions.py` (decisions shown in the operator survey)
- `analyze_s5_results.py` (reproduces Tables E.1 and E.2 from the private survey file; the data are not included)

### Corrective experiments (supersede the original campaign for the federation questions)
- `exp_lean.py`: data-scarcity and cold-start pilots
- `build_multi_data.py`, `exp_multi.py`: multi-building experiments, cold start, compute-controlled check
- `equiv_test.py`: equivalence analysis
- `patch_control.py`, `fix_cold.py`: patches kept for provenance (`exp_multi.py` and `equiv_test.py` here already include them)

`results/` holds the aggregated per-seed results (JSON). Text logs of the runs (`output_log*.txt`, `*_results.txt`) are included.

## Data

Raw data are not included. See [DATA.md](DATA.md).

## Software

Python 3.14 was used. Library minimum versions are in `requirements.txt`. Install them with:

    pip install -r requirements.txt

## Reproducing the multi-building experiments

    python build_multi_data.py --k 3
    python exp_multi.py --mode train --k 3 --n_seeds 8 --data_frac 0.2
    python exp_multi.py --mode train --k 3 --n_seeds 24 --data_frac 0.2 --conditions indep fed_arch
    python exp_multi.py --mode train --k 3 --n_seeds 12 --data_frac 0.2 --conditions indep3x --only_arch residential
    python exp_multi.py --mode cold --k 3 --n_seeds 6 --rounds 4 --steps 800
    python exp_multi.py --mode analyze
    python equiv_test.py --folder results\fed_multi\frac0.2 --margin 2

Training ran at roughly 75 environment steps per second; one seed of the three conditions on 12 buildings took about 35 minutes.

## License and citation

MIT License (see `LICENSE`). Please cite the thesis and this archive (see `CITATION.cff`).
