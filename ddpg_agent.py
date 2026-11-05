"""
ddpg_agent.py
Deep Deterministic Policy Gradient agent (Section 3.2, Equations 3.3-3.5).
Actor mu(s|theta_mu) -> deterministic continuous action in [-1,1]^d.
Critic Q(s,a|theta_Q) -> scalar value estimate.
Includes target networks (Polyak averaging) and a replay buffer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, init_w=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        # DDPG-standard small-scale final-layer initialisation (Lillicrap et
        # al. 2015 [16], Section 7): PyTorch's default Linear init can be
        # large enough to push tanh's output near +-1 from the very first
        # forward pass for some action dimensions, whose gradient then
        # vanishes (tanh' -> 0 near +-1) before training can correct it - a
        # likely explanation for the office actor's ESS-dimension policy
        # collapse documented in Appendix C.11's addendum. Initialising the
        # final layer's weights/bias uniformly in [-init_w, init_w] keeps
        # the initial policy near-linear (away from saturation) for every
        # action dimension, a standard and well-founded mitigation.
        final_layer = self.net[-2]
        nn.init.uniform_(final_layer.weight, -init_w, init_w)
        nn.init.uniform_(final_layer.bias, -init_w, init_w)

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(s2, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


class OUNoise:
    """Ornstein-Uhlenbeck exploration noise (standard for DDPG), with linear
    decay support so exploration reduces as training progresses (this was
    missing before: constant-sigma noise throughout training was one likely
    cause of B3's earlier underperformance vs. the simple B1 baseline).
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.3, sigma_min=0.02, decay_steps=15000):
        self.action_dim = action_dim
        self.mu, self.theta = mu, theta
        self.sigma_start = sigma
        self.sigma_min = sigma_min
        self.decay_steps = decay_steps
        self.t = 0
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def current_sigma(self):
        frac = min(1.0, self.t / max(1, self.decay_steps))
        return self.sigma_start + frac * (self.sigma_min - self.sigma_start)

    def sample(self):
        sigma = self.current_sigma()
        dx = self.theta * (self.mu - self.state) + sigma * np.random.randn(self.action_dim)
        self.state += dx
        self.t += 1
        return self.state


class DDPGAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, lr_actor=1e-3, lr_critic=1e-3,
                 noise_sigma=0.3, noise_decay_steps=15000):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.noise = OUNoise(action_dim, sigma=noise_sigma, decay_steps=noise_decay_steps)
        self.action_dim = action_dim

    def act(self, state, explore=True):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).squeeze(0).numpy()
        if explore:
            a = a + self.noise.sample()
        return np.clip(a, -1.0, 1.0)

    def soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self, batch, personalisation_target=None, mu=1.0):
        """One DDPG update. If personalisation_target (a global critic state
        dict) and mu < 1 are supplied, blends the critic loss per the
        L2-personalisation loss (Eq. 4.6): L' = mu*Lc + (1-mu)*D(Q, Q_global).
        """
        s, a, r, s2, d = batch

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target_next = self.critic_target(s2, a2)
            y = r + self.gamma * (1 - d) * q_target_next

        q = self.critic(s, a)
        local_critic_loss = F.mse_loss(q, y)

        if personalisation_target is not None and mu < 1.0:
            temp_global = Critic(s.shape[1], a.shape[1])
            temp_global.load_state_dict(personalisation_target)
            with torch.no_grad():
                q_global = temp_global(s, a)
            distance_term = F.mse_loss(q, q_global)
            critic_loss = mu * local_critic_loss + (1 - mu) * distance_term
        else:
            critic_loss = local_critic_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


    def action_diversity_check(self, recent_actions, threshold=0.01):
        """Health check for the policy-collapse failure mode documented in
        Appendix C.11's addendum: warns if any action dimension's std across
        recent actions has collapsed to near-constant. Call periodically
        during training (e.g., every few hundred steps) with a buffer of
        recent agent.act(..., explore=False) outputs.
        """
        recent_actions = np.array(recent_actions)
        stds = recent_actions.std(axis=0)
        collapsed_dims = np.where(stds < threshold)[0]
        if len(collapsed_dims) > 0:
            print(f"  [WARNING] Action dimension(s) {list(collapsed_dims)} show near-zero "
                  f"variance (std={stds[collapsed_dims]}) - possible policy collapse in "
                  f"progress (see Appendix C.11's addendum for this known failure mode).")
        return stds


if __name__ == "__main__":
    from environment import BuildingEnergyEnv

    env = BuildingEnergyEnv(seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim)
    buf = ReplayBuffer()

    s = env.reset()
    ep_reward = 0.0
    for step in range(200):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        ep_reward += r
        if len(buf) > 32:
            batch = buf.sample(32)
            losses = agent.train_step(batch)
        if done:
            s = env.reset()
    print(f"Smoke-test complete: 200 steps, cumulative reward = {ep_reward:.4f}")
    print("Final losses:", losses)
