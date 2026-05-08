"""
Training Script — A2C (NumPy, không cần PyTorch)
==================================================
Sau mỗi lần train, TỰ ĐỘNG GHI THÊM (append) vào CSV — không xóa data cũ.
Mỗi lần train được đánh dấu bằng run_id tự tăng + timestamp.

CSV files trong results/:
  training_log_{instance}.csv   ← Dữ liệu từng episode (tích lũy theo run_id)
  evaluation_{instance}.csv     ← So sánh A2C vs Baselines (tích lũy theo run_id)
  schedule_{instance}_{algo}.csv← Lịch sản xuất của run mới nhất

Cách chạy:
    python train_numpy.py --instance ft06 --episodes 2000
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
    "critic_loss", "entropy", "best_makespan", "is_best",
]

EVALUATION_FIELDS = [
    "run_id", "trained_at", "instance",
    "algorithm", "run", "makespan", "idle_time", "utilization",
]

SCHEDULE_FIELDS = [
    "run_id", "trained_at", "instance",
    "algo", "job", "machine", "op_index", "start", "end", "duration",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_run_id(path: str) -> int:
    """
    Tự động tạo run_id tăng dần.
    Đọc run_id lớn nhất trong file hiện tại rồi +1.
    Nếu file chưa tồn tại → run_id = 1.
    """
    if not os.path.exists(path):
        return 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            ids = [
                int(row["run_id"])
                for row in csv.DictReader(f)
                if row.get("run_id", "").isdigit()
            ]
        return (max(ids) + 1) if ids else 1
    except Exception:
        return 1


def save_training_log_csv(rows: list, path: str):
    """
    GHI THÊM log training vào CSV — giữ nguyên data cũ.

    Schema: run_id | trained_at | instance | n_episodes |
            episode | reward | makespan | avg_reward_100 |
            critic_loss | entropy | best_makespan | is_best
    """
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"  📄 training_log  {'[thêm vào]' if exists else '[tạo mới]'} → {path}")


def save_evaluation_csv(rows: list, path: str):
    """
    GHI THÊM kết quả evaluation vào CSV — giữ nguyên data cũ.

    Schema: run_id | trained_at | instance |
            algorithm | run | makespan | idle_time | utilization
    """
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVALUATION_FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"  📄 evaluation    {'[thêm vào]' if exists else '[tạo mới]'} → {path}")


def save_schedule_csv(schedule: list, algo: str, path: str,
                      run_id: int, trained_at: str, instance: str):
    """
    GHI THÊM schedule vào CSV — giữ nguyên data cũ.

    Schema: run_id | trained_at | instance |
            algo | job | machine | op_index | start | end | duration
    """
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEDULE_FIELDS)
        if not exists:
            w.writeheader()
        for entry in schedule:
            w.writerow({
                "run_id"    : run_id,
                "trained_at": trained_at,
                "instance"  : instance,
                "algo"      : algo,
                "job"       : entry["job"],
                "machine"   : entry["machine"],
                "op_index"  : entry.get("op_index", ""),
                "start"     : entry["start"],
                "end"       : entry["end"],
                "duration"  : entry["end"] - entry["start"],
            })
    print(f"  📄 schedule_{algo:<5} {'[thêm vào]' if exists else '[tạo mới]'} → {path}")


# ── Main Training Function ────────────────────────────────────────────────────

def train_numpy(
    instance_name : str   = "3x3",
    n_episodes    : int   = 500,
    hidden_dim    : int   = 128,
    lr_actor      : float = 3e-4,
    lr_critic     : float = 5e-4,
    gamma         : float = 0.99,
    entropy_coef  : float = 0.05,
    save_dir      : str   = "results",
    log_interval  : int   = 50,
    seed          : int   = 42,
    n_eval_runs   : int   = 30,
):
    """
    Train A2C và export kết quả ra CSV (append — giữ data cũ).

    Returns:
        agent, best_makespan
    """
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ── Metadata của run này ──────────────────────────────────────────────────
    log_path  = os.path.join(save_dir, f"training_log_{instance_name}.csv")
    eval_path = os.path.join(save_dir, f"evaluation_{instance_name}.csv")

    run_id     = _next_run_id(log_path)      # Tự tăng từ CSV hiện tại
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    jobs_data = get_instance(instance_name)
    info      = instance_info(jobs_data)
    env       = JobShopEnv(jobs_data)

    print("\n" + "═"*58)
    print(f"  Run ID    : #{run_id}  ({trained_at})")
    print(f"  Instance  : {instance_name}  "
          f"({info['n_jobs']} Jobs × {info['n_machines']} Machines)")
    print(f"  Episodes  : {n_episodes}  |  Lower bound ≈ {info['critical_path_lb']}")
    print("═"*58)

    agent = A2CAgentNumpy(
        state_dim    = env.observation_space.shape[0],
        action_dim   = env.action_space.n,
        hidden_dim   = hidden_dim,
        lr_actor     = lr_actor,
        lr_critic    = lr_critic,
        gamma        = gamma,
        entropy_coef = entropy_coef,
    )

    # ── PHASE 1: Training ─────────────────────────────────────────────────────
    print(f"\n🚀 Training...\n")

    training_rows = []
    best_makespan = float("inf")
    best_schedule = None
    reward_window = deque(maxlen=100)

    for ep in range(1, n_episodes + 1):
        state, _       = env.reset()
        episode_reward = 0.0
        done           = False

        while not done:
            valid                   = env.get_valid_actions()
            action, log_prob, value = agent.select_action(state, valid)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, done, log_prob, value, valid)
            episode_reward += reward
            state = next_state

        losses   = agent.update()
        makespan = env.get_makespan()
        reward_window.append(episode_reward)

        is_best = makespan < best_makespan
        if is_best:
            best_makespan = makespan
            best_schedule = env.schedule.copy()
            agent.save(os.path.join(save_dir, f"model_{instance_name}"))

        # Mỗi episode = 1 row trong CSV — gắn run_id và metadata
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
            "best_makespan" : best_makespan,
            "is_best"       : int(is_best),
        })

        if ep % log_interval == 0 or ep == 1:
            print(f"  Ep {ep:>5}/{n_episodes} | "
                  f"AvgR: {np.mean(reward_window):>8.1f} | "
                  f"MS: {makespan:>4.0f} (best={best_makespan}) | "
                  f"Loss: {losses.get('critic_loss', 0):>8.4f}")

    # ── PHASE 2: Evaluation ───────────────────────────────────────────────────
    print(f"\n📊 Evaluating ({n_eval_runs} runs/algo)...")

    eval_rows  = []
    sched_map  = {}

    # A2C
    best_a2c_ms, best_a2c_sched = float("inf"), None
    for run in range(n_eval_runs):
        state, _ = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            a, _, _ = agent.select_action(state, valid)
            state, _, done, _, _ = env.step(a)
        ms = env.get_makespan()
        eval_rows.append({
            "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
            "algorithm": "a2c", "run": run + 1, "makespan": ms,
            "idle_time": round(env.get_idle_time(), 2),
            "utilization": round(env.get_utilization(), 4),
        })
        if ms < best_a2c_ms:
            best_a2c_ms    = ms
            best_a2c_sched = env.schedule.copy()

    sched_map["a2c"] = best_a2c_sched
    print(f"  A2C  : mean={np.mean([r['makespan'] for r in eval_rows if r['algorithm']=='a2c']):.1f}"
          f"  best={best_a2c_ms}")

    # Baselines (deterministic → lặp n_eval_runs lần để cùng format)
    for algo in ("fifo", "spt", "lpt", "edd"):
        r = run_dispatching_rule(jobs_data, algo)
        for run in range(n_eval_runs):
            eval_rows.append({
                "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
                "algorithm": algo, "run": run + 1, "makespan": r["makespan"],
                "idle_time": round(r["idle_time"], 2),
                "utilization": round(r["utilization"], 4),
            })
        sched_map[algo] = r["schedule"]
        print(f"  {algo.upper():<5}: makespan={r['makespan']}")

    # Random (stochastic → chạy đủ n_eval_runs lần)
    for run in range(n_eval_runs):
        r = run_dispatching_rule(jobs_data, "random", seed=run)
        eval_rows.append({
            "run_id": run_id, "trained_at": trained_at, "instance": instance_name,
            "algorithm": "random", "run": run + 1, "makespan": r["makespan"],
            "idle_time": round(r["idle_time"], 2),
            "utilization": round(r["utilization"], 4),
        })

    # ── PHASE 3: Export CSV (append mode) ────────────────────────────────────
    print(f"\n💾 Ghi CSV (Run #{run_id})...")

    save_training_log_csv(training_rows, log_path)
    save_evaluation_csv(eval_rows, eval_path)

    for algo, sched in sched_map.items():
        if sched:
            save_schedule_csv(
                sched, algo,
                os.path.join(save_dir, f"schedule_{instance_name}_{algo}.csv"),
                run_id=run_id, trained_at=trained_at, instance=instance_name,
            )

    print(f"\n✅ Run #{run_id} hoàn thành!  Best makespan = {best_makespan}")
    return agent, best_makespan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2C cho JSSP — append CSV")
    parser.add_argument("--instance",  type=str,   default="ft06")
    parser.add_argument("--episodes",  type=int,   default=1000)
    parser.add_argument("--hidden",    type=int,   default=128)
    parser.add_argument("--lr_actor",  type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=5e-4)
    parser.add_argument("--gamma",     type=float, default=0.99)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--eval_runs", type=int,   default=30)
    parser.add_argument("--save_dir",  type=str,   default="results")
    args = parser.parse_args()

    train_numpy(
        instance_name = args.instance,
        n_episodes    = args.episodes,
        hidden_dim    = args.hidden,
        lr_actor      = args.lr_actor,
        lr_critic     = args.lr_critic,
        gamma         = args.gamma,
        seed          = args.seed,
        n_eval_runs   = args.eval_runs,
        save_dir      = args.save_dir,
    )
