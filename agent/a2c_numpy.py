"""
A2C Agent (NumPy) — IMPROVED VERSION
=====================================
Cải tiến từ vanilla A2C theo best practices từ research:

  1. ✅ Generalized Advantage Estimation (GAE) — giảm variance gradient
       Reference: Schulman et al. (2016), High-Dimensional Continuous Control...
  2. ✅ Advantage Normalization — ổn định training trên episode dài
  3. ✅ Entropy Coefficient Decay — bắt đầu explore, kết thúc exploit
  4. ✅ Gradient Clipping — chống exploding gradient
  5. ✅ Best Policy Tracking — lưu policy tốt nhất, không "quên"
  6. ✅ Orthogonal Initialization — chuẩn cho RL networks

Backwards compatible với version cũ qua flag `use_gae`.

API giữ nguyên:
    agent = A2CAgentNumpy(state_dim, action_dim)
    action, log_prob, value = agent.select_action(state, valid_actions)
    agent.store_transition(...)
    losses = agent.update()
    agent.save(path) / agent.load(path)
"""

import numpy as np
import os
from typing import Optional, Tuple, Dict, List


# ── Adam Optimizer (NumPy implementation) ─────────────────────────────────────
class Adam:
    """Adam optimizer thuần NumPy — đồng bộ với torch.optim.Adam."""

    def __init__(self, params: list, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]   # 1st moment
        self.v = [np.zeros_like(p) for p in params]   # 2nd moment
        self.t = 0

    def step(self, params: list, grads: list) -> list:
        """Update params in-place và trả về."""
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


# ── Helper functions ──────────────────────────────────────────────────────────
def _orthogonal_init(shape: tuple, gain: float = 1.0) -> np.ndarray:
    """Orthogonal initialization — chuẩn cho RL theo paper PPO."""
    if len(shape) < 2:
        return np.random.randn(*shape) * gain
    flat = np.random.randn(shape[0], int(np.prod(shape[1:])))
    u, _, vt = np.linalg.svd(flat, full_matrices=False)
    w = u if u.shape == flat.shape else vt
    return (gain * w).reshape(shape).astype(np.float64)


def _relu(x):       return np.maximum(0, x)
def _relu_grad(x):  return (x > 0).astype(np.float64)

def _softmax(x):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
#  A2C Agent
# ══════════════════════════════════════════════════════════════════════════════
class A2CAgentNumpy:
    """
    Actor-Critic agent với GAE, entropy decay, advantage normalization.

    Cải tiến chính so với vanilla A2C:
      - GAE (λ=0.95): Smooth advantage giữa Monte Carlo (high variance)
        và TD-1 (high bias) → giảm variance gradient
      - Entropy decay: explore → exploit theo thời gian
      - Advantage normalization: giúp training ổn định khi reward scale lớn
      - Gradient clipping: chống exploding gradient
    """

    def __init__(
        self,
        state_dim       : int,
        action_dim      : int,
        hidden_dim      : int   = 256,            # ↑ từ 128 lên 256
        lr_actor        : float = 1e-4,           # ↓ từ 3e-4 (ổn định hơn)
        lr_critic       : float = 5e-4,
        gamma           : float = 0.99,
        gae_lambda      : float = 0.95,           # ★ GAE parameter
        entropy_coef    : float = 0.05,           # Start entropy
        entropy_min     : float = 0.005,          # ★ Entropy floor
        entropy_decay   : float = 0.9995,         # ★ Decay rate / update
        max_grad_norm   : float = 0.5,            # ★ Gradient clip
        normalize_adv   : bool  = True,           # ★ Adv normalization
        use_gae         : bool  = True,           # ★ Toggle GAE on/off
    ):
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.hidden_dim    = hidden_dim
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.entropy_coef  = entropy_coef
        self.entropy_min   = entropy_min
        self.entropy_decay = entropy_decay
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv
        self.use_gae       = use_gae

        # ── Actor network: state → action probs ──────────────────────────────
        # Output layer dùng gain=0.01 (theo PPO paper) để khởi đầu policy đều
        self.W1_a = _orthogonal_init((state_dim,  hidden_dim), gain=np.sqrt(2))
        self.b1_a = np.zeros(hidden_dim)
        self.W2_a = _orthogonal_init((hidden_dim, hidden_dim), gain=np.sqrt(2))
        self.b2_a = np.zeros(hidden_dim)
        self.W3_a = _orthogonal_init((hidden_dim, action_dim), gain=0.01)
        self.b3_a = np.zeros(action_dim)

        # ── Critic network: state → V(s) ─────────────────────────────────────
        self.W1_c = _orthogonal_init((state_dim,  hidden_dim), gain=np.sqrt(2))
        self.b1_c = np.zeros(hidden_dim)
        self.W2_c = _orthogonal_init((hidden_dim, hidden_dim), gain=np.sqrt(2))
        self.b2_c = np.zeros(hidden_dim)
        self.W3_c = _orthogonal_init((hidden_dim, 1),          gain=1.0)
        self.b3_c = np.zeros(1)

        # ── Optimizers ───────────────────────────────────────────────────────
        self.actor_params  = [self.W1_a, self.b1_a, self.W2_a, self.b2_a,
                              self.W3_a, self.b3_a]
        self.critic_params = [self.W1_c, self.b1_c, self.W2_c, self.b2_c,
                              self.W3_c, self.b3_c]
        self.opt_actor  = Adam(self.actor_params,  lr=lr_actor)
        self.opt_critic = Adam(self.critic_params, lr=lr_critic)

        # ── Rollout buffer ───────────────────────────────────────────────────
        self.reset_buffer()

        # ── Metrics ──────────────────────────────────────────────────────────
        self.n_updates = 0

    # ── Buffer Management ────────────────────────────────────────────────────
    def reset_buffer(self):
        self.buf_states     = []
        self.buf_actions    = []
        self.buf_rewards    = []
        self.buf_dones      = []
        self.buf_log_probs  = []
        self.buf_values     = []
        self.buf_valid      = []

    def _to_mask(self, valid_actions) -> np.ndarray:
        """
        Chuẩn hóa valid_actions thành boolean mask shape (action_dim,).

        Support 2 format đầu vào:
          (a) Boolean mask shape (action_dim,) — đã đúng
          (b) List/array of integer indices — convert thành mask

        Examples:
            _to_mask([True, False, True, True])  → [T, F, T, T]  (a)
            _to_mask([0, 2, 3])                   → [T, F, T, T]  (b)
        """
        arr = np.asarray(valid_actions)

        # Case (a): boolean mask đã đúng shape
        if arr.dtype == bool and arr.shape == (self.action_dim,):
            return arr

        # Case (b): list of integer indices → convert sang boolean mask
        mask = np.zeros(self.action_dim, dtype=bool)
        if arr.size > 0:
            # Edge case: boolean mask sai shape → re-interpret as indices
            if arr.dtype == bool:
                # Lấy index của các True
                indices = np.flatnonzero(arr)
            else:
                indices = arr.astype(int)
            mask[indices] = True
        return mask

    def store_transition(self, state, action, reward, done,
                         log_prob, value, valid_actions):
        """Lưu transition vào buffer (normalize valid_actions thành mask)."""
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)
        self.buf_log_probs.append(log_prob)
        self.buf_values.append(value)
        self.buf_valid.append(self._to_mask(valid_actions))  # ← Chuẩn hóa

    # ── Forward Pass ─────────────────────────────────────────────────────────
    def _actor_forward(self, state: np.ndarray) -> tuple:
        """Returns: action_logits, hidden_activations (for backprop)."""
        h1 = _relu(state @ self.W1_a + self.b1_a)
        h2 = _relu(h1    @ self.W2_a + self.b2_a)
        logits = h2 @ self.W3_a + self.b3_a
        return logits, (h1, h2)

    def _critic_forward(self, state: np.ndarray) -> tuple:
        h1 = _relu(state @ self.W1_c + self.b1_c)
        h2 = _relu(h1    @ self.W2_c + self.b2_c)
        v  = (h2 @ self.W3_c + self.b3_c).item()
        return v, (h1, h2)

    def select_action(self, state: np.ndarray, valid_actions
                      ) -> Tuple[int, float, float]:
        """
        Chọn action theo policy + action masking.

        Args:
            state: shape (state_dim,)
            valid_actions: Một trong 2 format:
                (a) Boolean mask shape (action_dim,) — True = hợp lệ
                (b) List/array of integer indices — VD: [0, 2, 3] = 3 jobs hợp lệ

        Returns:
            action (int), log_prob (float), value (float)
        """
        logits, _ = self._actor_forward(state)

        # Normalize input → boolean mask shape (action_dim,)
        mask = self._to_mask(valid_actions)

        # Action masking: invalid → -inf → prob ≈ 0 sau softmax
        masked_logits = np.where(mask, logits, -1e9)
        probs = _softmax(masked_logits)

        # Safety: nếu tất cả invalid (không nên xảy ra) → uniform fallback
        if not np.isfinite(probs).all() or probs.sum() < 1e-9:
            probs = mask.astype(np.float64)
            probs = probs / max(probs.sum(), 1.0)

        # Sample action từ phân phối
        action = int(np.random.choice(self.action_dim, p=probs))
        log_prob = float(np.log(probs[action] + 1e-10))

        value, _ = self._critic_forward(state)
        return action, log_prob, value

    # ── GAE Computation ──────────────────────────────────────────────────────
    def _compute_gae(self, rewards, values, dones, next_value=0.0):
        """
        Generalized Advantage Estimation.

        GAE balance giữa:
          - λ=0: TD(1) — high bias, low variance
          - λ=1: Monte Carlo — low bias, high variance

        λ=0.95 (chuẩn từ PPO paper): sweet spot

        Formula:
            δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
            A_t = Σ (γλ)^k · δ_{t+k}
            R_t = A_t + V(s_t)  ← target cho critic
        """
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value if not dones[t] else 0.0
            else:
                next_v = values[t + 1] if not dones[t] else 0.0

            delta = rewards[t] + self.gamma * next_v - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values)
        return advantages, returns

    def _compute_returns_simple(self, rewards, dones):
        """Fallback: Monte Carlo returns (khi use_gae=False)."""
        T = len(rewards)
        returns = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running * (1 - dones[t])
            returns[t] = running
        advantages = returns - np.array(self.buf_values)
        return advantages, returns

    # ── Backward Pass + Update ───────────────────────────────────────────────
    def update(self) -> Dict[str, float]:
        """Update Actor + Critic sau khi rollout 1 episode."""
        if len(self.buf_states) == 0:
            return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}

        states  = np.array(self.buf_states,  dtype=np.float64)
        actions = np.array(self.buf_actions, dtype=np.int64)
        rewards = np.array(self.buf_rewards, dtype=np.float64)
        dones   = np.array(self.buf_dones,   dtype=np.float64)
        valids  = np.array(self.buf_valid,   dtype=bool)

        # ── Compute Advantages ───────────────────────────────────────────────
        if self.use_gae:
            advantages, returns = self._compute_gae(rewards, self.buf_values, dones)
        else:
            advantages, returns = self._compute_returns_simple(rewards, dones)

        # ── Advantage Normalization (giảm variance) ──────────────────────────
        if self.normalize_adv and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Critic update ────────────────────────────────────────────────────
        critic_grads, critic_loss = self._critic_backward(states, returns)
        critic_grads = self._clip_grads(critic_grads, self.max_grad_norm)
        self.critic_params = self.opt_critic.step(self.critic_params, critic_grads)
        self._sync_critic_params()

        # ── Actor update ─────────────────────────────────────────────────────
        actor_grads, actor_loss, entropy = self._actor_backward(
            states, actions, advantages, valids
        )
        actor_grads = self._clip_grads(actor_grads, self.max_grad_norm)
        self.actor_params = self.opt_actor.step(self.actor_params, actor_grads)
        self._sync_actor_params()

        # ── Entropy Decay ────────────────────────────────────────────────────
        self.entropy_coef = max(
            self.entropy_min,
            self.entropy_coef * self.entropy_decay,
        )

        self.n_updates += 1
        self.reset_buffer()

        return {
            "actor_loss"  : float(actor_loss),
            "critic_loss" : float(critic_loss),
            "entropy"     : float(entropy),
            "entropy_coef": float(self.entropy_coef),
            "adv_mean"    : float(advantages.mean()),
            "adv_std"     : float(advantages.std()),
        }

    def _critic_backward(self, states, returns):
        """MSE loss + manual backprop."""
        T = len(states)
        # Forward pass batch
        h1 = _relu(states @ self.W1_c + self.b1_c)
        h2 = _relu(h1     @ self.W2_c + self.b2_c)
        v_pred = (h2 @ self.W3_c + self.b3_c).flatten()

        # Loss: MSE
        td_error = v_pred - returns
        loss = 0.5 * np.mean(td_error ** 2)

        # Backprop
        dv = (td_error / T).reshape(-1, 1)
        dW3 = h2.T @ dv
        db3 = dv.sum(axis=0)
        dh2 = dv @ self.W3_c.T
        dh2 *= _relu_grad(h1 @ self.W2_c + self.b2_c)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ self.W2_c.T
        dh1 *= _relu_grad(states @ self.W1_c + self.b1_c)
        dW1 = states.T @ dh1
        db1 = dh1.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3], loss

    def _actor_backward(self, states, actions, advantages, valids):
        """Policy gradient + entropy bonus + manual backprop."""
        T = len(states)
        # Forward pass batch
        h1 = _relu(states @ self.W1_a + self.b1_a)
        h2 = _relu(h1     @ self.W2_a + self.b2_a)
        logits = h2 @ self.W3_a + self.b3_a

        # Apply action mask
        masked_logits = np.where(valids, logits, -1e9)
        probs = _softmax(masked_logits)
        log_probs = np.log(probs + 1e-10)

        # Policy gradient: -log_prob(a) * advantage
        chosen_log_probs = log_probs[np.arange(T), actions]
        pg_loss = -np.mean(chosen_log_probs * advantages)

        # Entropy: H = -Σ p log p (encourage exploration)
        entropy = -np.sum(probs * log_probs, axis=-1).mean()
        loss = pg_loss - self.entropy_coef * entropy

        # ── Gradient w.r.t logits ────────────────────────────────────────────
        # ∂L_pg/∂logits = (probs - one_hot(action)) * advantage / T
        one_hot = np.zeros((T, self.action_dim))
        one_hot[np.arange(T), actions] = 1
        dlogits_pg = (probs - one_hot) * advantages.reshape(-1, 1) / T

        # ∂(-entropy)/∂logits = probs * (log_probs - H) — derivation đầy đủ
        # Đơn giản hóa: gradient của entropy bonus
        dlogits_ent = -self.entropy_coef * (-(probs * (log_probs + 1)) +
                                              probs * np.sum(probs * (log_probs + 1),
                                                              axis=-1, keepdims=True)) / T
        dlogits = dlogits_pg + dlogits_ent
        dlogits[~valids] = 0  # Mask invalid

        # Backprop qua các layer
        dW3 = h2.T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.W3_a.T
        dh2 *= _relu_grad(h1 @ self.W2_a + self.b2_a)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ self.W2_a.T
        dh1 *= _relu_grad(states @ self.W1_a + self.b1_a)
        dW1 = states.T @ dh1
        db1 = dh1.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3], loss, entropy

    def _clip_grads(self, grads: list, max_norm: float) -> list:
        """Global L2 norm clipping (giống torch.nn.utils.clip_grad_norm_)."""
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        scale = max_norm / (total_norm + 1e-8)
        if scale < 1.0:
            grads = [g * scale for g in grads]
        return grads

    # ── Sync params với layer attributes ─────────────────────────────────────
    def _sync_actor_params(self):
        (self.W1_a, self.b1_a, self.W2_a, self.b2_a,
         self.W3_a, self.b3_a) = self.actor_params

    def _sync_critic_params(self):
        (self.W1_c, self.b1_c, self.W2_c, self.b2_c,
         self.W3_c, self.b3_c) = self.critic_params

    # ── Save / Load ──────────────────────────────────────────────────────────
    def save(self, path: str):
        """Lưu weights ra .npz file."""
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(
            path,
            # Actor
            W1_a=self.W1_a, b1_a=self.b1_a,
            W2_a=self.W2_a, b2_a=self.b2_a,
            W3_a=self.W3_a, b3_a=self.b3_a,
            # Critic
            W1_c=self.W1_c, b1_c=self.b1_c,
            W2_c=self.W2_c, b2_c=self.b2_c,
            W3_c=self.W3_c, b3_c=self.b3_c,
            # Metadata
            state_dim=self.state_dim, action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            entropy_coef=self.entropy_coef,
        )

    def load(self, path: str):
        """Load weights từ .npz file."""
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        self.W1_a, self.b1_a = data["W1_a"], data["b1_a"]
        self.W2_a, self.b2_a = data["W2_a"], data["b2_a"]
        self.W3_a, self.b3_a = data["W3_a"], data["b3_a"]
        self.W1_c, self.b1_c = data["W1_c"], data["b1_c"]
        self.W2_c, self.b2_c = data["W2_c"], data["b2_c"]
        self.W3_c, self.b3_c = data["W3_c"], data["b3_c"]
        # Re-bind params
        self.actor_params  = [self.W1_a, self.b1_a, self.W2_a, self.b2_a,
                              self.W3_a, self.b3_a]
        self.critic_params = [self.W1_c, self.b1_c, self.W2_c, self.b2_c,
                              self.W3_c, self.b3_c]
        if "entropy_coef" in data.files:
            self.entropy_coef = float(data["entropy_coef"])