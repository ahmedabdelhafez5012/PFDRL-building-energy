\# PFDRL: federated deep reinforcement learning for building energy management



Code, analysis scripts and aggregated results for the M.Sc. thesis

"Energy Management in Smart Buildings with Privacy Preservation Using Federated Deep Reinforcement Learning"

(Ahmed A. A. Khater, Helwan University, Faculty of Engineering, Department of Smart Electrical Grids Engineering, 2026).



\## Status

Research code, not a product. The main finding is negative: in this simplified environment, federation did

not reduce cost relative to independent training. The original campaign had implementation issues disclosed in

the thesis (Limitation L10: the FedProx arm applied no proximal term and only critics were aggregated). The

corrective experiments (exp\_lean.py, exp\_multi.py) supersede it for the federation questions.



\## Contents

Core modules

\- environment.py: building energy environment (15-min steps, seven-variable state, HVAC and battery actions)

\- ddpg\_agent.py: DDPG agent

\- federated\_learning.py: aggregation and noise injection

\- xai\_module.py: SHAP and LIME explanations of the battery action

\- data\_loader.py, data\_loader\_office.py, data\_loader\_commercial.py, data\_loader\_mixeduse.py: archetype data loaders



Baselines

\- baseline\_b1\_rule\_based.py, baseline\_b2\_mpc.py, compare\_baselines.py



Original campaign (H1-H4, B1-B4)

\- main.py (early smoke test), main\_real\_federation.py

\- scenario\_s3\_mu\_sweep.py, scenario\_s4\_epsilon\_sweep.py, lambda\_sweep.py (superseded)

\- multi\_seed\_h1\_test.py, multi\_seed\_h2\_test.py, multi\_seed\_h3\_latency\_test.py, multi\_seed\_h4\_test.py

\- 00\_power\_analysis.py, 01\_full\_ablation\_b1\_b4.py, 02\_extended\_n\_multiseed.py, 03\_multiseed\_s3\_s4.py, 04\_merge\_and\_generate\_tables.py

\- 05\_reward\_normalisation\_ablation.py, 06\_capacity\_interference\_pilot.py, 07\_weighted\_sampling\_pilot.py (reward-scale diagnosis)

\- 09a\_inspect\_residential\_sources.py, 09b\_build\_residential\_expansion.py, 10\_residential\_scale\_federation.py (residential scale-up)

\- 15\_b3\_vs\_b2.py (paired B3 vs B2 test, N=30)

\- 13\_resolve\_open\_items.py, 14\_check\_residential.py, diagnose\_office\_actor.py (audit utilities)

\- extract\_s5\_demo\_decisions.py (decisions shown in the operator survey)



Corrective experiments (supersede the original campaign for the federation questions)

\- exp\_lean.py: data-scarcity and cold-start pilots

\- build\_multi\_data.py, exp\_multi.py: multi-building experiments, cold start, compute-controlled check

\- equiv\_test.py: equivalence analysis

\- patch\_control.py, fix\_cold.py: patches kept for provenance (exp\_multi.py and equiv\_test.py here already include them)



results/: aggregated per-seed results (JSON). Text logs of the runs (output\_log\*.txt, \*\_results.txt) are included.



\## Data

Raw data are not included. See DATA.md.



\## Software

Python 3.14 was used. Library minimum versions are in requirements.txt:

pip install -r requirements.txt



\## Reproducing the multi-building experiments

python build\_multi\_data.py --k 3

python exp\_multi.py --mode train --k 3 --n\_seeds 8 --data\_frac 0.2

python exp\_multi.py --mode train --k 3 --n\_seeds 24 --data\_frac 0.2 --conditions indep fed\_arch

python exp\_multi.py --mode train --k 3 --n\_seeds 12 --data\_frac 0.2 --conditions indep3x --only\_arch residential

python exp\_multi.py --mode cold --k 3 --n\_seeds 6 --rounds 4 --steps 800

python exp\_multi.py --mode analyze

python equiv\_test.py --folder results\\fed\_multi\\frac0.2 --margin 2

Training ran at roughly 75 environment steps per second; one seed of the three conditions on 12 buildings took about 35 minutes.



\## License and citation

MIT License (see LICENSE). Please cite the thesis and this archive (see CITATION.cff).

