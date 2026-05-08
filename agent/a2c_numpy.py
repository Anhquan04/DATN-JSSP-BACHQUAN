"""
A2C Agent — NumPy Implementation (không cần PyTorch)
======================================================
Dùng để demo và verify logic. Khi có GPU, dùng a2c_agent.py (PyTorch).

Kiến trúc: 2-layer MLP cho cả Actor và Critic.
Tham chiếu: Sutton & Barto, Chương 13 (Policy Gradient Methods)
"""

import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY — Activation functions
# ─────────────────────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numerically stable."""
    x = x - np.max(x)          # Tránh overflow
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  MLP — Multi-Layer Perceptron đơn giản
# ─────────────────────────────────────────────────────────────────────────────

class MLP:
    """
    2-hidden-layer neural network.
    Forward pass: Linear → ReLU → Linear → ReLU → Linear
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Khởi tạo weights theo He initialization (tốt cho ReLU)
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = rng.normal(0, scale1, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.normal(0, 0.01, (hidden_dim, out_dim))
        self.b3 = np.zeros(out_dim)

        # Cache activations cho backward pass
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = relu(x @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3

        # Lưu cache cho backward
        self._cache = {"x": x, "h1": h1, "h2": h2, "out": out}
        return out

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params: List[np.ndarray]):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params


# ─────────────────────────────────────────────────────────────────────────────
#  ADAM OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class Adam:
    """
    Adam optimizer: adaptive learning rate.
    Paper: Kingma & Ba (2015) — https://arxiv.org/abs/1412.6980
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m     = None   # 1st moment
        self.v     = None   # 2nd moment

    def step(self, params: List[np.ndarray],
             grads: List[np.ndarray]) -> List[np.ndarray]:
        self.t += 1
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            updated.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return updated


# ─────────────────────────────────────────────────────────────────────────────
#  A2C AGENT (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class A2CAgentNumpy:
    """
    Advantage Actor-Critic — NumPy implementation.

    Dùng REINFORCE-style gradient (Monte Carlo returns).
    Ổn định nhờ advantage normalization và entropy regularization.
    """

    def __init__(
        self,
        state_dim   : int,
        action_dim  : int,
        hidden_dim  : int   = 128,
        lr_actor    : float = 5e-4,
        lr_critic   : float = 1e-3,
        gamma       : float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.entropy_coef = entropy_coef

        # Khởi tạo 2 mạng riêng biệt
        self.actor  = MLP(state_dim, hidden_dim, action_dim, seed=42)
        self.critic = MLP(state_dim, hidden_dim, 1,          seed=43)

        self.actor_opt  = Adam(lr=lr_actor)
        self.critic_opt = Adam(lr=lr_critic)

        # Buffer trajectory của episode
        self._reset_trajectory()

        print(f"✅ A2CAgentNumpy initialized")
        print(f"   State={state_dim} | Actions={action_dim} | Hidden={hidden_dim}")

    # ── Public API ────────────────────────────────────────────────────

    def select_action(
        self, state: np.ndarray, valid_actions: List[int]
    ) -> Tuple[int, float, float]:
        """
        Chọn action dựa trên policy hiện tại.

        Returns:
            (action_id, log_prob, state_value)
        """
        logits = self.actor.forward(state)

        # Action masking: gán -inf cho action không hợp lệ
        mask = np.full(self.action_dim, -1e9)
        for a in valid_actions:
            mask[a] = 0.0
        logits = logits + mask

        probs   = softmax(logits)
        action  = np.random.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        value = self.critic.forward(state)[0]

        return int(action), float(log_prob), float(value)

    def store_transition(self, state, action, reward, done, log_prob, value, valid_actions):
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.valid_actions_list.append(valid_actions[:])

    def update(self) -> Dict:
        """Cập nhật Actor và Critic sau 1 episode. Trả về loss info."""
        if not self.rewards:
            return {}

        # ── Tính discounted returns ────────────────────────────────
        returns = self._compute_returns()
        returns = np.array(returns, dtype=np.float64)
        values  = np.array(self.values, dtype=np.float64)

        # ── Advantage = Return - Value ──────────────────────────────
        advantages = returns - values
        # Normalize để training ổn định
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Update Critic (Gradient Descent trên MSE) ───────────────
        critic_grads = self._critic_gradients(returns)
        new_params   = self.critic_opt.step(self.critic.get_params(), critic_grads)
        self.critic.set_params(new_params)

        # ── Update Actor (Policy Gradient) ──────────────────────────
        actor_grads, entropy = self._actor_gradients(advantages)
        new_params = self.actor_opt.step(self.actor.get_params(), actor_grads)
        self.actor.set_params(new_params)

        # Tính critic loss để log
        pred_values  = np.array([self.critic.forward(s)[0] for s in self.states])
        critic_loss  = np.mean((returns - pred_values)**2)

        self._reset_trajectory()

        return {
            "critic_loss"    : float(critic_loss),
            "entropy"        : float(entropy),
            "mean_advantage" : float(advantages.mean()),
            "mean_return"    : float(returns.mean()),
        }

    def save(self, path: str):
        """Lưu weights ra file .npz."""
        np.savez(path,
            actor_W1=self.actor.W1,  actor_b1=self.actor.b1,
            actor_W2=self.actor.W2,  actor_b2=self.actor.b2,
            actor_W3=self.actor.W3,  actor_b3=self.actor.b3,
            critic_W1=self.critic.W1, critic_b1=self.critic.b1,
            critic_W2=self.critic.W2, critic_b2=self.critic.b2,
            critic_W3=self.critic.W3, critic_b3=self.critic.b3,
        )
        print(f"💾 Model saved → {path}.npz")

    def load(self, path: str):
        """Load weights từ file .npz."""
        data = np.load(path)
        self.actor.set_params([
            data["actor_W1"],  data["actor_b1"],
            data["actor_W2"],  data["actor_b2"],
            data["actor_W3"],  data["actor_b3"],
        ])
        self.critic.set_params([
            data["critic_W1"], data["critic_b1"],
            data["critic_W2"], data["critic_b2"],
            data["critic_W3"], data["critic_b3"],
        ])
        print(f"✅ Model loaded ← {path}")

    # ── Private ───────────────────────────────────────────────────────

    def _compute_returns(self) -> List[float]:
        """Monte Carlo returns: G_t = r_t + γ*r_{t+1} + ..."""
        returns, G = [], 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _critic_gradients(self, returns: np.ndarray) -> List[np.ndarray]:
        """
        Gradient của MSE loss theo weights Critic.
        dL/dW = -2 * (return - V(s)) * dV/dW
        """
        grads_W1 = np.zeros_like(self.critic.W1)
        grads_b1 = np.zeros_like(self.critic.b1)
        grads_W2 = np.zeros_like(self.critic.W2)
        grads_b2 = np.zeros_like(self.critic.b2)
        grads_W3 = np.zeros_like(self.critic.W3)
        grads_b3 = np.zeros_like(self.critic.b3)

        n = len(self.states)
        for i, (state, G) in enumerate(zip(self.states, returns)):
            self.critic.forward(state)
            cache = self.critic._cache
            value = cache["out"][0]

            # MSE gradient
            delta = 2.0 * (value - G) / n   # dL/d(out)

            # Backprop layer 3
            grads_W3 += np.outer(cache["h2"], [delta])
            grads_b3 += delta

            # Backprop layer 2
            d_h2 = delta * self.critic.W3.flatten()
            d_h2 *= (cache["h2"] > 0)   # ReLU derivative
            grads_W2 += np.outer(cache["h1"], d_h2)
            grads_b2 += d_h2

            # Backprop layer 1
            d_h1 = d_h2 @ self.critic.W2.T
            d_h1 *= (cache["h1"] > 0)
            grads_W1 += np.outer(cache["x"], d_h1)
            grads_b1 += d_h1

        return [grads_W1, grads_b1, grads_W2, grads_b2, grads_W3, grads_b3]

    def _actor_gradients(
        self, advantages: np.ndarray
    ) -> Tuple[List[np.ndarray], float]:
        """
        Policy gradient với entropy regularization.
        ∇J(θ) = E[∇log π(a|s) * A(s,a)] + β * ∇H(π)
        """
        grads_W1 = np.zeros_like(self.actor.W1)
        grads_b1 = np.zeros_like(self.actor.b1)
        grads_W2 = np.zeros_like(self.actor.W2)
        grads_b2 = np.zeros_like(self.actor.b2)
        grads_W3 = np.zeros_like(self.actor.W3)
        grads_b3 = np.zeros_like(self.actor.b3)

        total_entropy = 0.0
        n = len(self.states)

        for i, (state, action, adv, valid) in enumerate(
            zip(self.states, self.actions, advantages, self.valid_actions_list)
        ):
            logits = self.actor.forward(state)
            cache  = self.actor._cache

            # Mask invalid actions
            mask      = np.full(self.action_dim, -1e9)
            mask[valid] = 0.0
            masked_logits = logits + mask
            probs = softmax(masked_logits)

            # Entropy H(π) = -Σ p*log(p)
            entropy        = -np.sum(probs * np.log(probs + 1e-8))
            total_entropy += entropy

            # Gradient của log π(a|s) * advantage
            d_logits = probs.copy()
            d_logits[action] -= 1.0      # ∂log(p_a)/∂logits
            d_logits = -adv * d_logits / n   # Policy gradient (ascent → descent)

            # Entropy gradient: ∂H/∂logits = -(log p + 1) * p → trừ để maximize
            d_entropy   = -(np.log(probs + 1e-8) + 1.0) * probs
            d_logits   -= self.entropy_coef * d_entropy / n

            # Backprop layer 3
            grads_W3 += np.outer(cache["h2"], d_logits)
            grads_b3 += d_logits

            # Layer 2
            d_h2  = d_logits @ self.actor.W3.T
            d_h2 *= (cache["h2"] > 0)
            grads_W2 += np.outer(cache["h1"], d_h2)
            grads_b2 += d_h2

            # Layer 1
            d_h1  = d_h2 @ self.actor.W2.T
            d_h1 *= (cache["h1"] > 0)
            grads_W1 += np.outer(cache["x"], d_h1)
            grads_b1 += d_h1

        grads = [grads_W1, grads_b1, grads_W2, grads_b2, grads_W3, grads_b3]
        return grads, total_entropy / n

    def _reset_trajectory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []
        self.valid_actions_list = []
