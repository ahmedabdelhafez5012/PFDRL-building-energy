# Corrected differential privacy with accounting (`exp_dp.py`)

This experiment closes thesis limitations L6 and L10-ii. The original implementation added unseeded noise to model weights with no privacy accounting, so the nominal ε = 8 composed to roughly ε ≈ 124 over 50 rounds. `exp_dp.py` replaces it with:

- client-level DP (privacy unit = one building's entire dataset);
- the update delta (not the weights) clipped to L2 norm C = 1;
- seeded Gaussian noise;
- Rényi-DP accounting composed over all R = 10 rounds (full participation, δ = 1e-5);
- two threat models: `client` (each building noises its own update; untrusted server, sensitivity 2C) and `server` (trusted aggregator adds noise to the sum; sensitivity C).

Sanity check: the accountant reproduces the thesis estimate, giving ε ≈ 124 for σ = 0.6056 over 50 rounds.

All runs use the 12-building corrective setup of `exp_multi.py` (3 buildings per archetype, data fraction 0.2, 400 steps per round, μ = 0.05). Baselines `indep`, `fed_arch`, `fed_all`, `b1` and `noop` are read from `results/fed_multi/frac0.2`. Statistics are paired Wilcoxon tests across seeds; the primary test is ε = 8 versus independent training in four archetypes (α = 0.05/4 = 0.0125).

## Experiment A — whole model shared within each archetype (3 clients, client mode)

Mean change versus independent training, all buildings (`results/fed_dp/frac0.2`):

| Condition | Seeds | Cost | Discomfort |
| --- | --- | --- | --- |
| Clipping only | 12 | +0.53% (p = 0.47) | −48.7% (p = 0.0005) |
| ε = 128 | 8 | +0.07% | +336.2% |
| ε = 64 | 8 | +1.69% | +339.1% |
| ε = 32 | 8 | +4.38% | +340.6% |
| ε = 16 | 8 | +2.70% | +341.1% |
| ε = 8 | 12 | −1.02% (p = 0.52) | +349.8% (p = 0.0005) |
| ε = 1 | 2 | +16.22% | +399.0% |

At ε = 8 the discomfort increase is significant in every archetype (p = 0.0005), while cost does not change significantly in any of them. Privacy noise destroys the policy: with 9,667 shared parameters and 3 clients, the noise norm on the averaged update is about 248 times the clipped signal.

## Noise-to-signal ratio

Expected noise norm on the averaged update divided by the clip norm C: z·√d/√n in client mode, z·√d/n in server mode (d = shared parameters, n = clients).

| Setting at ε = 8 | Whole model (d = 9,667) | Last layer only (d = 195) |
| --- | --- | --- |
| client, 3 clients | 248 | 35 |
| server, 3 clients | 72 | 10 |
| client, 12 clients | 124 | 18 |
| server, 12 clients | 18 | 2.5 |

## Experiment B — last layer only, trusted aggregator, 12 clients, 400 local fine-tuning steps

Results in `results/fed_dp_all_server_head_ft400/frac0.2`.

Primary test, ε = 8 versus independent training (12 seeds):

| Archetype | Cost | p | Discomfort | p |
| --- | --- | --- | --- | --- |
| Commercial | +0.11% | 0.97 | +43.7% | 0.043 |
| Mixed-use | −0.78% | 0.57 | +63.2% | 0.021 |
| Office | −1.40% | 0.57 | +47.2% | 0.0068 |
| Residential | −1.19% | 0.68 | +41.9% | 0.043 |
| All | −0.70% | 0.62 | +50.1% | 0.0024 |

Effect of the noise alone, ε = 8 versus the same configuration without noise (clipping only, same budget, 12 seeds):

| Archetype | Cost | p | Discomfort | p |
| --- | --- | --- | --- | --- |
| Commercial | +1.86% | 0.23 | +8.6% | 0.97 |
| Mixed-use | +0.77% | 0.73 | +25.8% | 0.15 |
| Office | −0.51% | 0.85 | +19.6% | 0.62 |
| Residential | +1.46% | 0.79 | +9.3% | 0.57 |
| All | +0.74% | 0.62 | +16.3% | 0.73 |

No significant effect of the noise was detected. Most of the discomfort penalty against independent training comes from sharing one last layer across different building types: the same configuration without noise already shows +41.0% (p = 0.034).

Privacy–utility frontier versus the no-noise configuration (all buildings; 4 seeds except ε = 8):

| ε | Noise / signal | Cost | Discomfort |
| --- | --- | --- | --- |
| 16 | 1.4 | +2.45% | +4.1% |
| 8 | 2.5 | +0.74% | +16.3% |
| 4 | 4.8 | −1.84% | +48.0% |
| 1 | 18.0 | +0.70% | +238.5% |

Control: the last layer only with 3 clients in client mode (noise/signal 35) fails as predicted, giving +321.3% discomfort at ε = 8 (4 seeds).

Equivalence of the noise effect (`dp_summary.py`, 90% CI = TOST at 5%, 12 seeds): cost [−0.87%, +2.35%], so equivalence within ±2% is narrowly not established; discomfort [−9.2%, +41.7%].

## Experiment C — last layer only, trusted aggregator, 16 residential clients, 400 fine-tuning steps

Results in `results/fed_dp_res16_server_head_ft400/frac0.2` (the 16 residential clients of Appendix C.15; independent training, B1 and no control rerun on the same buildings; 8 seeds; noise/signal 1.9).

| Comparison at ε = 8 | Cost | p | Discomfort | p |
| --- | --- | --- | --- | --- |
| vs independent training | −0.16% | 0.95 | +23.7% | 0.055 |
| No-noise configuration vs independent training | +1.14% | 0.46 | +18.3% | 0.11 |
| vs no-noise configuration (noise effect) | −1.27% | 0.31 | +5.3% | 0.25 |
| vs B1 | +11.13% | 0.0078 | −11.7% | 0.016 |

Equivalence of the noise effect: cost [−3.05%, +0.51%], discomfort [−1.6%, +12.2%]. The discomfort penalty of the large homogeneous federation is about half that of the mixed-type federation in Experiment B, and the noise effect is bounded to at most about 12%. Clients 1–15 use a modelled HVAC load.

## Summary figure

`results/dp_nsr_figure.png` (produced by `dp_summary.py`) plots discomfort against the expected noise-to-signal ratio for every configuration and ε: the penalty is small near a ratio of one and saturates at about +340% to +400% above a ratio of about 20.

## Interpretation and limits

Formal DP at ε = 8 is workable only when the noise-to-signal ratio is small: few shared parameters, many clients and a trusted aggregator. This creates a design tension. Privacy needs many clients, while data heterogeneity favours few similar ones.

Limits: the frontier points other than ε = 8 use 4 seeds; `server` mode assumes a trusted aggregator; local fine-tuning adds 10% to the training budget of independent training; all runs use the reduced 0.2 data budget; non-significance is not equivalence (see the TOST intervals above).

## Reproduce

```
python exp_dp.py --mode sigma
python exp_dp.py --mode train --n_seeds 12 --eps 8
python exp_dp.py --mode train --n_seeds 8 --eps 128 64 32 16 8
python exp_dp.py --mode train --share head --dp_mode server --group all --finetune 400 --n_seeds 12 --eps 8
python exp_dp.py --mode train --share head --dp_mode server --group all --finetune 400 --n_seeds 4 --eps 16 4 1
python exp_dp.py --mode train --share head --n_seeds 4 --eps 8
python exp_dp.py --mode train --clients res16 --share head --dp_mode server --finetune 400 --n_seeds 8 --eps 8
python exp_dp.py --mode analyze [same --clients/--share/--dp_mode/--group/--finetune options]
python dp_summary.py --folder results/fed_dp_all_server_head_ft400/frac0.2
```

`exp_multi.py` now ends with `if __name__ == "__main__": main()`, so it can be imported without starting a run.
