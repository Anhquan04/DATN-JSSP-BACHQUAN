"""
Evaluate & Compare — A2C vs Baselines
======================================
Cách chạy:
    python evaluation/evaluate.py --instance ft06 --model_path results/model_ft06.npz
"""

import argparse
import os
import json
import numpy as np
import sys
from typing import Dict as Dict_type

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.job_shop_env  import JobShopEnv
from agent.a2c_numpy   import A2CAgentNumpy          # ← Dùng NumPy agent (.npz)
from baselines.dispatching_rules import evaluate_all_baselines
from data.instances    import get_instance, instance_info
from utils.visualization import (
    plot_learning_curves,
    plot_gantt,
    plot_comparison_boxplot,
    plot_comparison_bar,
    load_history,
)
from scipy import stats


# ── Helper: convert numpy types → Python native ───────────────────────────────
def _to_serializable(obj):
    """Đệ quy convert numpy types sang Python native để JSON không bị lỗi."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.bool_):     return bool(obj)
    return obj


def evaluate_a2c(agent: A2CAgentNumpy, jobs_data, n_runs: int = 30) -> Dict_type:
    """Đánh giá A2C agent trên n_runs lần để tính thống kê."""
    env           = JobShopEnv(jobs_data)
    makespans     = []
    best_schedule = None
    best_makespan = float("inf")

    for _ in range(n_runs):
        state, _ = env.reset()
        done = False
        while not done:
            valid        = env.get_valid_actions()
            action, _, _ = agent.select_action(state, valid)
            state, _, done, _, _ = env.step(action)

        ms = env.get_makespan()
        makespans.append(ms)
        if ms < best_makespan:
            best_makespan = ms
            best_schedule = env.schedule.copy()

    return {
        "mean"         : float(np.mean(makespans)),
        "std"          : float(np.std(makespans)),
        "min"          : int(np.min(makespans)),
        "max"          : int(np.max(makespans)),
        "all_makespans": [int(m) for m in makespans],
        "best_schedule": best_schedule,
        "idle_time"    : float(env.get_idle_time()),
        "utilization"  : float(env.get_utilization()),
    }


def run_full_evaluation(
    instance_name: str,
    model_path   : str,
    history_path : str = "",
    n_runs       : int = 30,
    output_dir   : str = "results",
):
    """Chạy toàn bộ evaluation và tạo charts."""
    os.makedirs(output_dir, exist_ok=True)

    jobs_data = get_instance(instance_name)
    info      = instance_info(jobs_data)
    print(f"\n🔍 Evaluating: {instance_name} | {info['n_jobs']}J × {info['n_machines']}M | LB={info['critical_path_lb']}\n")

    # Load agent
    env   = JobShopEnv(jobs_data)
    agent = A2CAgentNumpy(
        state_dim  = env.observation_space.shape[0],
        action_dim = env.action_space.n,
        hidden_dim = 128,
    )
    agent.load(model_path)

    # Evaluate
    print(f"🤖 Evaluating A2C ({n_runs} runs)...")
    a2c_results = evaluate_a2c(agent, jobs_data, n_runs=n_runs)

    print("📏 Evaluating baselines...")
    baseline_results = evaluate_all_baselines(jobs_data, n_runs=n_runs)

    all_results = {"A2C": a2c_results, **{k.upper(): v for k, v in baseline_results.items()}}

    # In bảng kết quả
    print("\n" + "═"*65)
    print("  EVALUATION RESULTS (Makespan)")
    print("═"*65)
    print(f"{'Algorithm':<12} {'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7} {'vs A2C':>8}")
    print("─"*65)
    a2c_mean = a2c_results["mean"]
    for algo, res in all_results.items():
        delta = ((res["mean"] - a2c_mean) / a2c_mean) * 100
        sign  = "+" if delta > 0 else ""
        print(f"{algo:<12} {res['mean']:>8.2f} {res['std']:>7.2f} "
              f"{res['min']:>7.0f} {res['max']:>7.0f} {sign}{delta:>7.1f}%")
    print("═"*65)

    # Wilcoxon test
    print("\n📊 Wilcoxon Signed-Rank Test (H1: A2C < Baseline, α=0.05):")
    print(f"  {'Algorithm':<10} {'p-value':>10} {'Kết luận':>25}")
    print("  " + "─"*48)
    for algo, res in all_results.items():
        if algo == "A2C":
            continue
        n  = min(len(a2c_results["all_makespans"]), len(res["all_makespans"]))
        try:
            _, p = stats.wilcoxon(
                a2c_results["all_makespans"][:n],
                res["all_makespans"][:n],
                alternative="less"
            )
        except Exception:
            p = 1.0
        sig = "✅ Tốt hơn (p<0.05)" if p < 0.05 else "❌ Không có ý nghĩa"
        print(f"  {algo:<10} {p:>10.4f} {sig:>25}")

    # Charts
    print("\n🎨 Generating charts...")

    if history_path and os.path.exists(history_path):
        history = load_history(history_path)
        plot_learning_curves(
            history,
            title     = f"A2C Learning Curves — {instance_name}",
            save_path = os.path.join(output_dir, f"learning_curves_{instance_name}.png"),
        )
    else:
        print("   ⚠️  Bỏ qua learning curves (không có history file)")

    if a2c_results["best_schedule"]:
        plot_gantt(
            schedule   = a2c_results["best_schedule"],
            n_machines = info["n_machines"],
            n_jobs     = info["n_jobs"],
            title      = f"Best A2C Schedule — {instance_name} (Makespan={a2c_results['min']})",
            save_path  = os.path.join(output_dir, f"gantt_{instance_name}.png"),
        )

    plot_comparison_boxplot(
        results   = {algo: res["all_makespans"] for algo, res in all_results.items()},
        title     = f"Makespan Distribution — {instance_name}",
        save_path = os.path.join(output_dir, f"boxplot_{instance_name}.png"),
    )

    plot_comparison_bar(
        results   = all_results,
        title     = f"Mean Makespan Comparison — {instance_name}",
        save_path = os.path.join(output_dir, f"barchart_{instance_name}.png"),
    )

    # Lưu JSON — dùng _to_serializable để tránh lỗi int64
    save_path = os.path.join(output_dir, f"eval_results_{instance_name}.json")
    serializable = {
        algo: _to_serializable({k: v for k, v in res.items() if k != "best_schedule"})
        for algo, res in all_results.items()
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Hoàn thành! Kết quả → {output_dir}/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate A2C vs Baselines")
    parser.add_argument("--instance",     type=str, default="ft06")
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--history_path", type=str, default="")
    parser.add_argument("--n_runs",       type=int, default=30)
    parser.add_argument("--output_dir",   type=str, default="results")
    args = parser.parse_args()

    run_full_evaluation(
        instance_name = args.instance,
        model_path    = args.model_path,
        history_path  = args.history_path,
        n_runs        = args.n_runs,
        output_dir    = args.output_dir,
    )