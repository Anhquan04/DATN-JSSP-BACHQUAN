"""
Training Script — A2C (NumPy) — RECIPE v3
==========================================
Cải tiến từ v2 sau khi phát hiện 3 vấn đề trên FT06:

  ❌ v2: Greedy=149 >>> Stochastic best=66    (policy collapse)
  ❌ v2: Mean stochastic ≈ Random              (chưa học được pattern tốt)
  ❌ v2: Best save dựa trên 1 episode lucky    (không stable)

  ✅ v3 fixes:
       1. lr_actor: 1e-4 → 3e-4 (Adam default, học nhanh hơn, ổn định)
       2. entropy_min: 0.005 → 0.01 (giữ exploration đủ để không stuck)
       3. normalize_adv: True → False (cho reward sparse như JSSP)
       4. Best policy = quick eval mỗi N episodes, KHÔNG dựa trên 1 episode
       5. GAE λ: 0.95 → 0.9 (ít variance hơn trong reward thưa)

Recipe v3 cho FT06:
    episodes=2500, hidden=256, lr_actor=3e-4
    eval_every=200 → check rolling performance định kỳ
    Save model chỉ khi rolling eval thực sự cải thiện
"""

import argparse, os, csv
import numpy as np
from collections import deque
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.job_shop_env            import JobShopEnv
from agent.a2c_numpy             import A2CAgentNumpy
from baselines.dispatching_rules import run_dispatching_rule
from data.instances              import get_instance, instance_info


# ── Schema CSV ────────────────────────────────────────────────────────────────
TRAINING_LOG_FIELDS = [
    "run_id", "trained_at", "instance", "n_episodes",
    "episode", "reward", "makespan", "avg_reward_100",
    "critic_loss", "entropy", "entropy_coef",
    "best_makespan", "is_best", "rolling_eval_ms",
]

EVALUATION_FIELDS = [
    "run_id", "trained_at", "instance",
    "algorithm", "run", "makespan", "idle_time", "utilization",
]

SCHEDULE_FIELDS = [
    "run_id", "trained_at", "instance",
    "algo", "job", "machine", "op_index", "start", "end", "duration",
]

DETERMINISTIC_BASELINES = {"fifo", "spt", "lpt", "edd"}


# ── Recipe v3: hyperparameters đã tune ────────────────────────────────────────
INSTANCE_RECIPES = {
    "3x3"  : {"episodes": 1000, "hidden": 128, "lr_actor": 3e-4, "eval_every": 100},
    "4x4"  : {"episodes": 1500, "hidden": 128, "lr_actor": 3e-4, "eval_every": 100},
    "5x5"  : {"episodes": 2000, "hidden": 256, "lr_actor": 3e-4, "eval_every": 200},
    "ft06" : {"episodes": 2500, "hidden": 256, "lr_actor": 3e-4, "eval_every": 200},  # ★
    "ft10" : {"episodes": 5000, "hidden": 256, "lr_actor": 3e-4, "eval_every": 500},
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _next_run_id(path: str) -> int:
    if not os.path.exists(path):
        return 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            ids = [int(row["run_id"])
                   for row in csv.DictReader(f)
                   if row.get("run_id", "").isdigit()]
        return (max(ids) + 1) if ids else 1
    except Exception:
        return 1


def _append_csv(rows, path, fields):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        # extrasaction='ignore' để các CSV cũ vẫn append được (bỏ qua field thiếu)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


def save_schedule_csv(schedule, algo, path, run_id, trained_at, instance):
    rows = [{
        "run_id": run_id, "trained_at": trained_at, "instance": instance,
        "algo": algo,
        "job": e["job"], "machine": e["machine"],
        "op_index": e.get("op_index", ""),
        "start": e["start"], "end": e["end"],
        "duration": e["end"] - e["start"],
    } for e in schedule]
    _append_csv(rows, path, SCHEDULE_FIELDS)


def _select_action_greedy(agent: A2CAgentNumpy, state: np.ndarray, valid) -> int:
    """Argmax thay vì sample → deterministic policy execution."""
    logits, _ = agent._actor_forward(state)
    mask = agent._to_mask(valid)
    masked = np.where(mask, logits, -1e9)
    return int(np.argmax(masked))


def _evaluate_episode(env, agent, greedy=False):
    """Chạy 1 episode và trả về (makespan, schedule, idle, util)."""
    state, _ = env.reset()
    done = False
    while not done:
        valid = env.get_valid_actions()
        if greedy:
            action = _select_action_greedy(agent, state, valid)
        else:
            action, _, _ = agent.select_action(state, valid)
        state, _, done, _, _ = env.step(action)
    return env.get_makespan(), env.schedule.copy(), env.get_idle_time(), env.get_utilization()


def _quick_eval(env, agent, n_runs: int = 5) -> float:
    """
    Quick eval — chạy n_runs lần stochastic + 1 lần greedy, trả về MEAN.

    ★ Quan trọng: dùng MEAN thay vì MIN để tránh "lucky bias".
    Một policy good phải có MEAN tốt, không phải vài lần lucky.
    """
    makespans = []
    for _ in range(n_runs):
        ms, _, _, _ = _evaluate_episode(env, agent, greedy=False)
        makespans.append(ms)
    ms_g, _, _, _ = _evaluate_episode(env, agent, greedy=True)
    makespans.append(ms_g)
    return float(np.mean(makespans))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING (Recipe v3)
# ══════════════════════════════════════════════════════════════════════════════
def train_numpy(
    instance_name : str   = "ft06",
    n_episodes    : int   = None,
    hidden_dim    : int   = None,
    lr_actor      : float = None,
    lr_critic     : float = 5e-4,
    gamma         : float = 0.99,
    gae_lambda    : float = 0.9,           # ★ v3: 0.95 → 0.9 (ít noise)
    entropy_coef  : float = 0.05,
    entropy_min   : float = 0.01,          # ★ v3: 0.005 → 0.01 (giữ explore)
    normalize_adv : bool  = False,         # ★ v3: tắt (gây noise cho JSSP)
    eval_every    : int   = None,
    eval_runs     : int   = 5,
    save_dir      : str   = "results",
    log_interval  : int   = 50,
    seed          : int   = 42,
    n_eval_runs   : int   = 30,
):
    """Train A2C với Recipe v3."""
    # ── Auto-load recipe ──────────────────────────────────────────────────────
    recipe = INSTANCE_RECIPES.get(instance_name, {})
    n_episodes  = n_episodes  or recipe.get("episodes", 2500)
    hidden_dim  = hidden_dim  or recipe.get("hidden",   256)
    lr_actor    = lr_actor    or recipe.get("lr_actor", 3e-4)
    eval_every  = eval_every  or recipe.get("eval_every", 200)

    # Entropy decay: giảm từ entropy_coef → entropy_min trong 80% episodes
    target_step = int(0.8 * n_episodes)
    entropy_decay = (entropy_min / entropy_coef) ** (1.0 / max(target_step, 1))

    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    log_path  = os.path.join(save_dir, f"training_log_{instance_name}.csv")
    eval_path = os.path.join(save_dir, f"evaluation_{instance_name}.csv")

    run_id     = _next_run_id(log_path)
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    jobs_data = get_instance(instance_name)
    info      = instance_info(jobs_data)
    env       = JobShopEnv(jobs_data)

    print("\n" + "═"*64)
    print(f"  Run ID    : #{run_id}  ({trained_at})  [RECIPE v3]")
    print(f"  Instance  : {instance_name}  "
          f"({info['n_jobs']} Jobs × {info['n_machines']} Machines)")
    print(f"  Episodes  : {n_episodes}  |  Lower bound ≈ {info['critical_path_lb']}")
    print(f"  Hidden    : {hidden_dim}  |  lr_actor: {lr_actor}  |  γ: {gamma}")
    print(f"  GAE λ     : {gae_lambda}  |  Entropy: {entropy_coef} → {entropy_min}")
    print(f"  Normalize Adv: {normalize_adv}  |  Eval every: {eval_every} eps")
    print("═"*64)

    agent = A2CAgentNumpy(
        state_dim     = env.observation_space.shape[0],
        action_dim    = env.action_space.n,
        hidden_dim    = hidden_dim,
        lr_actor      = lr_actor,
        lr_critic     = lr_critic,
        gamma         = gamma,
        gae_lambda    = gae_lambda,
        entropy_coef  = entropy_coef,
        entropy_min   = entropy_min,
        entropy_decay = entropy_decay,
        use_gae       = True,
        normalize_adv = normalize_adv,
    )

    # ── PHASE 1: Training ─────────────────────────────────────────────────────
    print(f"\n🚀 Training {n_episodes} episodes...\n")

    training_rows  = []
    reward_window  = deque(maxlen=100)
    best_eval_score = float("inf")          # ★ Dựa trên eval, không phải episode
    best_model_path = os.path.join(save_dir, f"model_{instance_name}")
    last_rolling   = -1.0

    for ep in range(1, n_episodes + 1):
        # Rollout
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            valid = env.get_valid_actions()
            action, log_prob, value = agent.select_action(state, valid)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, done,
                                   log_prob, value, valid)
            episode_reward += reward
            state = next_state

        losses = agent.update()
        makespan = env.get_makespan()
        reward_window.append(episode_reward)

        # ★ Eval check định kỳ thay vì save mỗi episode
        rolling_ms = -1.0
        is_best = False
        if ep % eval_every == 0 or ep == n_episodes:
            rolling_ms = _quick_eval(env, agent, n_runs=eval_runs)
            last_rolling = rolling_ms
            if rolling_ms < best_eval_score:
                best_eval_score = rolling_ms
                agent.save(best_model_path)
                is_best = True

        training_rows.append({
            "run_id"        : run_id,
            "trained_at"    : trained_at,
            "instance"      : instance_name,
            "n_episodes"    : n_episodes,
            "episode"       : ep,
            "reward"        : round(episode_reward, 4),
            "makespan"      : makespan,
            "avg_reward_100": round(float(np.mean(reward_window)), 4),
            "critic_loss"   : round(losses.get("critic_loss", 0), 6),
            "entropy"       : round(losses.get("entropy", 0), 6),
            "entropy_coef"  : round(losses.get("entropy_coef", 0), 6),
            "best_makespan" : round(best_eval_score, 2),
            "is_best"       : int(is_best),
            "rolling_eval_ms": round(rolling_ms, 2) if rolling_ms > 0 else -1,
        })

        if ep % log_interval == 0 or ep == 1:
            ent_c = losses.get("entropy_coef", 0)
            roll_str = f" | RollEval: {last_rolling:.1f}" if last_rolling > 0 else ""
            mark = " ★" if is_best else ""
            print(f"  Ep {ep:>5}/{n_episodes} | "
                  f"AvgR: {np.mean(reward_window):>8.1f} | "
                  f"MS: {makespan:>4.0f} | "
                  f"Best Eval: {best_eval_score:>6.1f}{roll_str}{mark} | "
                  f"H_coef: {ent_c:.4f}")

    # ── PHASE 2: Final Evaluation ────────────────────────────────────────────
    print(f"\n📊 Final Evaluation với BEST checkpoint (eval_score={best_eval_score:.1f})...")
    agent.load(best_model_path)

    eval_rows = []
    sched_map = {}

    # A2C Stochastic
    print(f"\n  🤖 A2C-Stochastic — {n_eval_runs} runs...")
    a2c_makespans = []
    best_a2c_ms, best_a2c_sched = float("inf"), None
    for run in range(n_eval_runs):
        ms, sched, idle, util = _evaluate_episode(env, agent, greedy=False)
        a2c_makespans.append(ms)
        eval_rows.append({
            "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
            "algorithm": "a2c", "run": run + 1, "makespan": ms,
            "idle_time": round(idle, 2), "utilization": round(util, 4),
        })
        if ms < best_a2c_ms:
            best_a2c_ms, best_a2c_sched = ms, sched
    sched_map["a2c"] = best_a2c_sched
    print(f"     mean={np.mean(a2c_makespans):.1f}  "
          f"std={np.std(a2c_makespans):.2f}  best={best_a2c_ms}")

    # A2C Greedy
    ms_g, sched_g, idle_g, util_g = _evaluate_episode(env, agent, greedy=True)
    eval_rows.append({
        "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
        "algorithm": "a2c_greedy", "run": 1, "makespan": ms_g,
        "idle_time": round(idle_g, 2), "utilization": round(util_g, 4),
    })
    if ms_g < best_a2c_ms:
        sched_map["a2c"] = sched_g
    print(f"  🎯 A2C-Greedy: makespan={ms_g}")

    # Baselines
    print(f"\n  📏 Deterministic baselines:")
    for algo in DETERMINISTIC_BASELINES:
        r = run_dispatching_rule(jobs_data, algo)
        eval_rows.append({
            "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
            "algorithm": algo, "run": 1, "makespan": r["makespan"],
            "idle_time": round(r["idle_time"], 2),
            "utilization": round(r["utilization"], 4),
        })
        sched_map[algo] = r["schedule"]
        print(f"     {algo.upper():<5}: makespan={r['makespan']}")

    # Random
    print(f"\n  🎲 Random — {n_eval_runs} runs...")
    random_ms = []
    for run in range(n_eval_runs):
        r = run_dispatching_rule(jobs_data, "random", seed=run)
        random_ms.append(r["makespan"])
        eval_rows.append({
            "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
            "algorithm": "random", "run": run + 1, "makespan": r["makespan"],
            "idle_time": round(r["idle_time"], 2),
            "utilization": round(r["utilization"], 4),
        })
    print(f"     mean={np.mean(random_ms):.1f}  std={np.std(random_ms):.2f}")

    # ── PHASE 3: Export ──────────────────────────────────────────────────────
    print(f"\n💾 Ghi CSV (Run #{run_id})...")
    _append_csv(training_rows, log_path,  TRAINING_LOG_FIELDS)
    _append_csv(eval_rows,     eval_path, EVALUATION_FIELDS)
    for algo, sched in sched_map.items():
        if sched:
            save_schedule_csv(
                sched, algo,
                os.path.join(save_dir, f"schedule_{instance_name}_{algo}.csv"),
                run_id=run_id, trained_at=trained_at, instance=instance_name,
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    gap_lb = ((best_a2c_ms - info['critical_path_lb']) /
              info['critical_path_lb'] * 100)
    print(f"\n{'═'*64}")
    print(f"  ✅ Run #{run_id} COMPLETE")
    print(f"  Best Eval Score (rolling): {best_eval_score:.1f}")
    print(f"  Best A2C Stochastic      : {best_a2c_ms}")
    print(f"  A2C Greedy               : {ms_g}")
    print(f"  Lower Bound              : {info['critical_path_lb']}")
    print(f"  Gap to LB (best stoch.)  : {gap_lb:+.1f}%")
    print(f"{'═'*64}")

    # ── Diagnostic warnings ──────────────────────────────────────────────────
    if ms_g > best_a2c_ms * 1.5:
        print("\n⚠️  CẢNH BÁO: Greedy >> Stochastic best → policy chưa converge ổn định")
        print("   → Thử: tăng episodes (--episodes 4000), hoặc giảm lr_actor xuống 1e-4")
    if np.mean(a2c_makespans) > np.mean(random_ms) * 0.95:
        print("\n⚠️  CẢNH BÁO: A2C mean ≈ Random mean → policy chưa học được pattern tốt")
        print("   → Thử: kiểm tra reward shaping; tăng entropy_min lên 0.02")
    if best_eval_score > best_a2c_ms * 1.3:
        print("\n💡 LƯU Ý: Best stochastic tốt hơn best eval rolling")
        print("   → Có thể tăng eval_runs lên 10 để check chính xác hơn")

    return agent, best_a2c_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2C cho JSSP")
    parser.add_argument("--instance",     type=str,   default="ft06")
    parser.add_argument("--episodes",     type=int,   default=None)
    parser.add_argument("--hidden",       type=int,   default=None)
    parser.add_argument("--lr_actor",     type=float, default=None)
    parser.add_argument("--lr_critic",    type=float, default=5e-4)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--gae",          type=float, default=0.9)
    parser.add_argument("--entropy_min",  type=float, default=0.01)
    parser.add_argument("--normalize_adv", action="store_true",
                        help="Bật advantage normalization (mặc định: TẮT)")
    parser.add_argument("--eval_every",   type=int,   default=None)
    parser.add_argument("--eval_runs",    type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_eval_runs",  type=int,   default=30)
    parser.add_argument("--save_dir",     type=str,   default="results")
    args = parser.parse_args()

    train_numpy(
        instance_name = args.instance,
        n_episodes    = args.episodes,
        hidden_dim    = args.hidden,
        lr_actor      = args.lr_actor,
        lr_critic     = args.lr_critic,
        gamma         = args.gamma,
        gae_lambda    = args.gae,
        entropy_min   = args.entropy_min,
        normalize_adv = args.normalize_adv,
        eval_every    = args.eval_every,
        eval_runs     = args.eval_runs,
        seed          = args.seed,
        n_eval_runs   = args.n_eval_runs,
        save_dir      = args.save_dir,
    )