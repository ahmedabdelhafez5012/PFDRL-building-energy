"""
federated_learning.py
Aggregation strategies (Section 3.3, Equations 3.6-3.9 in the theoretical
background chapter) and the Gaussian differential-privacy mechanism
(Section 3.4) applied to transmitted critic/actor weight updates.
"""
import copy
import numpy as np
import torch


def state_dict_to_vector(sd):
    return torch.cat([p.flatten() for p in sd.values()])


def add_gaussian_dp_noise(state_dict, epsilon=8.0, delta=1e-5, clip_norm=1.0, seed=None):
    """Calibrated Gaussian mechanism (Section 3.4): clips the update to
    clip_norm, then adds noise with sigma computed for the target (eps,delta).
    sigma = clip_norm * sqrt(2 ln(1.25/delta)) / epsilon   (standard Gaussian
    mechanism calibration used in the thesis, sigma~1.96 for eps=8, delta=1e-5)
    """
    rng = np.random.default_rng(seed)
    sigma = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    vec = state_dict_to_vector(state_dict)
    norm = torch.norm(vec).item() + 1e-12
    scale = min(1.0, clip_norm / norm)

    noised = {}
    for k, v in state_dict.items():
        clipped = v * scale
        noise = torch.tensor(rng.normal(0, sigma, size=v.shape), dtype=v.dtype)
        noised[k] = clipped + noise
    return noised, sigma


def fedavg_aggregate(client_state_dicts, weights=None):
    n = len(client_state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    agg = copy.deepcopy(client_state_dicts[0])
    for k in agg.keys():
        agg[k] = sum(w * sd[k] for w, sd in zip(weights, client_state_dicts))
    return agg


def fedprox_aggregate(client_state_dicts, global_state_dict, mu=0.05, weights=None):
    """FedProx (Eq. 3.7): each client's local objective already includes the
    proximal term during local training; aggregation itself is a weighted
    average (identical mechanics to FedAvg at the server). mu affects the
    CLIENT-side training loss (see ddpg_agent.train_step's mu-blended loss),
    not the server-side averaging formula itself.
    """
    return fedavg_aggregate(client_state_dicts, weights)


def fedadam_aggregate(client_state_dicts, global_state_dict, server_state,
                       lr=0.01, beta1=0.9, beta2=0.99, eps=1e-3, weights=None):
    """FedAdam (Reddi et al. 2021): treats the averaged client delta as a
    pseudo-gradient and applies Adam-style server-side momentum.
    server_state holds 'm' and 'v' moment estimates across rounds.
    """
    n = len(client_state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    avg_client = fedavg_aggregate(client_state_dicts, weights)

    delta = {k: avg_client[k] - global_state_dict[k] for k in global_state_dict}
    if not server_state:
        server_state = {"m": {k: torch.zeros_like(v) for k, v in delta.items()},
                         "v": {k: torch.zeros_like(v) for k, v in delta.items()}}

    new_global = {}
    for k in global_state_dict:
        server_state["m"][k] = beta1 * server_state["m"][k] + (1 - beta1) * delta[k]
        server_state["v"][k] = beta2 * server_state["v"][k] + (1 - beta2) * (delta[k] ** 2)
        new_global[k] = global_state_dict[k] + lr * server_state["m"][k] / (
            torch.sqrt(server_state["v"][k]) + eps)
    return new_global, server_state


def scaffold_aggregate(client_state_dicts, client_control_variates, global_control_variate,
                        global_state_dict, weights=None):
    """SCAFFOLD (Karimireddy et al. 2020): simplified server-side aggregation.
    Full client-side control-variate correction is applied during local
    training; here we aggregate both the model weights and the control
    variates, and additionally return the communication overhead in
    elements (extra to the model itself) implied by transmitting c_i.
    """
    n = len(client_state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    new_global = fedavg_aggregate(client_state_dicts, weights)
    new_control = {k: sum(w * c[k] for w, c in zip(weights, client_control_variates))
                    for k in global_control_variate}
    extra_comm_elements = sum(v.numel() for v in global_control_variate.values())
    return new_global, new_control, extra_comm_elements


def communication_cost_mb(state_dict, extra_elements=0, dtype_bytes=4):
    n_params = sum(p.numel() for p in state_dict.values()) + extra_elements
    return n_params * dtype_bytes / (1024 ** 2)


if __name__ == "__main__":
    from ddpg_agent import Critic
    torch.manual_seed(0)

    c1, c2, c3 = Critic(7, 2), Critic(7, 2), Critic(7, 2)
    sds = [c1.state_dict(), c2.state_dict(), c3.state_dict()]

    avg = fedavg_aggregate(sds)
    print("FedAvg OK, param count:", sum(v.numel() for v in avg.values()))

    noised, sigma = add_gaussian_dp_noise(sds[0], epsilon=8.0, delta=1e-5)
    print(f"DP noise sigma (eps=8, delta=1e-5): {sigma:.4f}")

    cost = communication_cost_mb(avg)
    print(f"Communication cost per round (single critic): {cost:.4f} MB")

    global_sd = copy.deepcopy(avg)
    server_state = {}
    new_g, server_state = fedadam_aggregate(sds, global_sd, server_state)
    print("FedAdam OK, param count:", sum(v.numel() for v in new_g.values()))
