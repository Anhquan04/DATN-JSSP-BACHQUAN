
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.network(state)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        return torch.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class A2CAgent:
    """
    Advantage Actor-Critic Agent (PyTorch).

    Workflow:
        select_action() → store_transition() → update() → save/load()
    """

    def __init__(
        self,
        state_dim   : int,
        action_dim  : int,
        hidden_dim  : int   = 128,
        lr_actor    : float = 3e-4,
        lr_critic   : float = 5e-4,
        gamma       : float = 0.99,
        entropy_coef: float = 0.05,
        device      : str   = "auto",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.action_dim   = action_dim
        self.gamma        = gamma
        self.entropy_coef = entropy_coef

        self.actor  = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._reset_trajectory()

        n = (sum(p.numel() for p in self.actor.parameters()) +
             sum(p.numel() for p in self.critic.parameters()))
        print(f"✅ A2CAgent (PyTorch) | device={self.device} | params={n:,}")

    # ── Inference 

    def select_action(
        self,
        state        : np.ndarray,
        valid_actions: List[int],
    ) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask = torch.zeros(self.action_dim, device=self.device)
        for a in valid_actions:
            mask[a] = 1.0

        with torch.no_grad():
            probs = self.actor(s, mask.unsqueeze(0))
            value = self.critic(s)

        dist = torch.distributions.Categorical(probs)
        a    = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.squeeze().item())

    def store_transition(
        self,
        state        : np.ndarray,
        action       : int,
        reward       : float,
        done         : bool,
        log_prob     : float,
        value        : float,
        valid_actions: List[int],
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.valid_actions_list.append(valid_actions)

    # ── Training 

    def update(self) -> dict:
        """Cập nhật Actor + Critic sau 1 episode."""
        if not self.rewards:
            return {}

        returns   = self._compute_returns()
        states_t  = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        values_t  = torch.FloatTensor(self.values).to(self.device)

        # Advantage
        adv = returns_t - values_t
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Critic
        critic_loss = nn.MSELoss()(self.critic(states_t).squeeze(), returns_t)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        # Actor
        actor_loss_sum = torch.tensor(0.0, device=self.device)
        entropy_sum    = torch.tensor(0.0, device=self.device)
        n = len(self.states)

        for i in range(n):
            s_i    = states_t[i].unsqueeze(0)
            mask_i = torch.zeros(self.action_dim, device=self.device)
            for a in self.valid_actions_list[i]:
                mask_i[a] = 1.0
            probs_i = self.actor(s_i, mask_i.unsqueeze(0))
            dist_i  = torch.distributions.Categorical(probs_i)
            actor_loss_sum -= dist_i.log_prob(actions_t[i]) * adv[i].detach()
            entropy_sum    += dist_i.entropy()

        total_actor = actor_loss_sum / n - self.entropy_coef * entropy_sum / n
        self.actor_opt.zero_grad()
        total_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        self._reset_trajectory()
        return {
            "actor_loss" : float((actor_loss_sum / n).item()),
            "critic_loss": float(critic_loss.item()),
            "entropy"    : float((entropy_sum / n).item()),
        }

    # ── Checkpoint

    def save(self, path: str):
        """Lưu ra file .pt"""
        torch.save({
            "actor_state_dict" : self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_opt"        : self.actor_opt.state_dict(),
            "critic_opt"       : self.critic_opt.state_dict(),
        }, path)
        print(f"💾 Saved → {path}")

    def load(self, path: str):
        """Load từ file .pt"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        print(f"✅ Loaded ← {path}")

    # ── Private 

    def _compute_returns(self) -> List[float]:
        """G_t = r_t + γ·G_{t+1}"""
        returns, G = [], 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _reset_trajectory(self):
        self.states             = []
        self.actions            = []
        self.rewards            = []
        self.dones              = []
        self.log_probs          = []
        self.values             = []
        self.valid_actions_list = []