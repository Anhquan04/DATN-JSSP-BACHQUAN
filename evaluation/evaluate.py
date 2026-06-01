
import argparse
import os
import json
import numpy as np
import sys
from typing import Dict as Dict_type
from datetime import datetime, timezone
import csv
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.job_shop_env import JobShopEnv
from agent.a2c_numpy import A2CAgentNumpy
from baselines.dispatching_rules import evaluate_all_baselines
from data.instances import get_instance, instance_info
from utils.visualization import (
    plot_learning_curves,
    plot_gantt,
    plot_comparison_boxplot,
    plot_comparison_bar,
    load_history,
)
from scipy import stats


DETERMINISTIC_BASELINES = {"FIFO", "SPT", "LPT", "EDD"}


# ── Helper: Timestamp utilities 
def get_iso_timestamp() -> str:
    """
    Tạo ISO 8601 timestamp (UTC timezone aware).
    
    Ưu điểm:
    - Chuẩn quốc tế, dễ parse ở mọi nơi
    - Có timezone info → không bị lẫn múi giờ
    - Dễ sort lexicographically
    
    Ví dụ: "2025-05-13T15:30:45.123456+00:00"
    """
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


# ── Helper: CSV management 
def ensure_evaluation_csv_headers(csv_path: str):
    """
    Ensure CSV exists với headers đúng cho evaluation results.
    """
    headers = [
        "timestamp",
        "instance",
        "algorithm",
        "mean_makespan",
        "std_makespan",
        "min_makespan",
        "max_makespan",
        "is_deterministic",
        "vs_a2c_percent",
        "p_value",
        "statistical_significance",
    ]
    
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        print(f"✅ Tạo evaluation CSV mới: {csv_path}")
    else:
        print(f"📝 Append vào CSV tồn tại: {csv_path}")


def append_evaluation_record(
    csv_path: str,
    timestamp: str,
    instance: str,
    algorithm: str,
    mean_makespan: float,
    std_makespan: float,
    min_makespan: int,
    max_makespan: int,
    is_deterministic: bool,
    vs_a2c_percent: float,
    p_value: str = "-",
    statistical_significance: str = "-",
):
    """
    Append 1 dòng kết quả evaluation vào CSV.
    
    Args:
        csv_path: Đường dẫn CSV
        timestamp: Thời điểm evaluate (ISO 8601)
        instance: Tên instance (ft06, etc)
        algorithm: Tên algo (A2C, FIFO, RANDOM, etc)
        mean_makespan: Giá trị makespan trung bình
        std_makespan: Độ lệch chuẩn (hoặc "N/A" nếu deterministic)
        min_makespan: Giá trị min
        max_makespan: Giá trị max
        is_deterministic: Có phải deterministic không
        vs_a2c_percent: % chênh lệch so với A2C (0 nếu là A2C)
        p_value: P-value từ statistical test ("-" nếu không áp dụng)
        statistical_significance: Kết luận thống kê
    """
    ensure_evaluation_csv_headers(csv_path)
    
    headers = [
        "timestamp",
        "instance",
        "algorithm",
        "mean_makespan",
        "std_makespan",
        "min_makespan",
        "max_makespan",
        "is_deterministic",
        "vs_a2c_percent",
        "p_value",
        "statistical_significance",
    ]
    
    record = {
        "timestamp": timestamp,
        "instance": instance,
        "algorithm": algorithm,
        "mean_makespan": f"{mean_makespan:.2f}",
        "std_makespan": std_makespan,
        "min_makespan": min_makespan,
        "max_makespan": max_makespan,
        "is_deterministic": str(is_deterministic),
        "vs_a2c_percent": f"{vs_a2c_percent:.2f}%",
        "p_value": p_value,
        "statistical_significance": statistical_significance,
    }
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(record)
    except IOError as e:
        print(f"❌ Lỗi ghi CSV: {e}")


def ensure_training_csv_headers(csv_path: str):
    """Ensure CSV cho training records."""
    headers = ["instance", "model_name", "trained_at", "model_path", "n_runs"]
    
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        print(f"✅ Tạo training CSV mới: {csv_path}")


def append_training_record(
    csv_path: str,
    instance: str,
    model_name: str,
    trained_at: str,
    model_path: str,
    n_runs: int,
):
    """Append training record vào CSV."""
    ensure_training_csv_headers(csv_path)
    
    headers = ["instance", "model_name", "trained_at", "model_path", "n_runs"]
    record = {
        "instance": instance,
        "model_name": model_name,
        "trained_at": trained_at,
        "model_path": model_path,
        "n_runs": n_runs,
    }
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(record)
        print(f"✅ Ghi training record vào CSV: {model_name} @ {trained_at}")
    except IOError as e:
        print(f"❌ Lỗi ghi CSV: {e}")


# ── Helper: convert numpy types → Python native
def _to_serializable(obj):
    """Đệ quy convert numpy types sang Python native để JSON không bị lỗi."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def evaluate_a2c(agent: A2CAgentNumpy, jobs_data, n_runs: int = 30) -> Dict_type:
    """Đánh giá A2C agent trên n_runs lần để tính thống kê."""
    env = JobShopEnv(jobs_data)
    makespans = []
    best_schedule = None
    best_makespan = float("inf")

    for _ in range(n_runs):
        state, _ = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            action, _, _ = agent.select_action(state, valid)
            state, _, done, _, _ = env.step(action)

        ms = env.get_makespan()
        makespans.append(ms)
        if ms < best_makespan:
            best_makespan = ms
            best_schedule = env.schedule.copy()

    return {
        "mean": float(np.mean(makespans)),
        "std": float(np.std(makespans)),
        "min": int(np.min(makespans)),
        "max": int(np.max(makespans)),
        "all_makespans": [int(m) for m in makespans],
        "best_schedule": best_schedule,
        "idle_time": float(env.get_idle_time()),
        "utilization": float(env.get_utilization()),
        "is_deterministic": False,
    }


def _background_train(
    instance_name: str,
    n_episodes: int = 1000,
    results_dir: str = "results",
    training_csv_path: str = "results/training_records.csv",
):
    """
    Train A2C model in background và ghi trained_at vào CSV.
    """
    print(f"\n🚀 [BACKGROUND] Bắt đầu train A2C cho instance: {instance_name}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    jobs_data = get_instance(instance_name)
    info = instance_info(jobs_data)
    
    # Khởi tạo environment & agent
    env = JobShopEnv(jobs_data)
    agent = A2CAgentNumpy(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
    )
    
    # Training loop
    history = {"episode": [], "reward": [], "loss": []}
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid = env.get_valid_actions()
            action, log_prob, value = agent.select_action(state, valid)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        
        history["episode"].append(episode)
        history["reward"].append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"  Episode {episode + 1}/{n_episodes} ✓")
    
    # Lưu model VÀ ghi timestamp
    timestamp = get_iso_timestamp()
    model_name = f"model_{instance_name}"
    model_path = os.path.join(results_dir, f"{model_name}.npz")
    
    agent.save(model_path)
    print(f"✅ Model trained & saved: {model_path}")
    
    # Ghi vào training CSV
    append_training_record(
        csv_path=training_csv_path,
        instance=instance_name,
        model_name=model_name,
        trained_at=timestamp,
        model_path=model_path,
        n_runs=n_episodes,
    )
    
    # Lưu history
    history_path = os.path.join(results_dir, f"history_{instance_name}.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ [BACKGROUND] Hoàn thành train {instance_name} @ {timestamp}")


def run_full_evaluation(
    instance_name: str,
    model_path: str,
    history_path: str = "",
    n_runs: int = 30,
    output_dir: str = "results",
    eval_csv_path: str = "results/eval_results.csv",
):
    """
    Chạy toàn bộ evaluation và:
    1. Lưu kết quả vào CSV (dễ xem)
    2. Lưu kết quả vào JSON (để code xử lý)
    3. Tạo charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy timestamp hiện tại (dùng cho CSV)
    eval_timestamp = get_iso_timestamp()

    jobs_data = get_instance(instance_name)
    info = instance_info(jobs_data)
    print(f"\n🔍 Evaluating: {instance_name} | {info['n_jobs']}J × {info['n_machines']}M | LB={info['critical_path_lb']}\n")

    # Load agent
    env = JobShopEnv(jobs_data)
    agent = A2CAgentNumpy(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
    )
    agent.load(model_path)

    # Evaluate
    print(f"🤖 Evaluating A2C ({n_runs} runs)...")
    a2c_results = evaluate_a2c(agent, jobs_data, n_runs=n_runs)

    print("📏 Evaluating baselines...")
    baseline_results = evaluate_all_baselines(jobs_data, n_runs=n_runs)

    all_results = {"A2C": a2c_results, **{k.upper(): v for k, v in baseline_results.items()}}

    # In bảng kết quả console
    print("\n" + "═" * 72)
    print("  EVALUATION RESULTS (Makespan)")
    print("═" * 72)
    print(
        f"{'Algorithm':<12} {'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7} {'vs A2C':>8} {'Runs':>6}"
    )
    print("─" * 72)
    
    a2c_mean = a2c_results["mean"]
    
    for algo, res in all_results.items():
        delta = ((res["mean"] - a2c_mean) / a2c_mean) * 100
        sign = "+" if delta > 0 else ""
        n_samples = len(res["all_makespans"]) if res.get("all_makespans") else 1
        std_str = f"{res['std']:>7.2f}" if not res.get("is_deterministic") else "    N/A"
        min_str = f"{res['min']:>7.0f}"
        max_str = f"{res['max']:>7.0f}"
        print(
            f"{algo:<12} {res['mean']:>8.2f} {std_str} "
            f"{min_str} {max_str} {sign}{delta:>7.1f}% {n_samples:>6}"
        )
    print("═" * 72)

    # PHẦN QUAN TRỌNG: Tính statistical tests & ghi vào CSV
    
    print("\n📊 So sánh thống kê A2C vs Baselines (α=0.05):")
    print(f"  {'Algorithm':<10} {'Phương pháp':<30} {'p-value':>10} {'Kết luận':>22}")
    print("  " + "─" * 75)

    # Dictionary lưu p-values & kết luận để ghi CSV
    statistical_results = {}

    for algo, res in all_results.items():
        if algo == "A2C":
            # A2C vs A2C → 0% difference
            statistical_results[algo] = {
                "p_value": "-",
                "significance": "-",
                "delta": 0.0,
            }
            print(f"  {algo:<10} {'Baseline':<30} {'—':>10} {'(Baseline)':>22}")
            continue

        if res.get("is_deterministic", False) or algo in DETERMINISTIC_BASELINES:
            # Deterministic: so sánh trực tiếp mean
            baseline_val = res["mean"]
            delta = ((baseline_val - a2c_mean) / a2c_mean) * 100
            
            if a2c_mean < baseline_val:
                conclusion = "✅ A2C tốt hơn"
                sig = "A2C Better"
            elif a2c_mean == baseline_val:
                conclusion = "➖ Bằng nhau"
                sig = "Equal"
            else:
                conclusion = "❌ A2C kém hơn"
                sig = "A2C Worse"
            
            statistical_results[algo] = {
                "p_value": "-",
                "significance": sig,
                "delta": delta,
            }
            print(
                f"  {algo:<10} {'Direct Comparison (Det.)':<30} {'—':>10} {conclusion:>22}"
            )
        else:
            # Stochastic (Random): Wilcoxon Signed-Rank Test
            n = min(len(a2c_results["all_makespans"]), len(res["all_makespans"]))
            try:
                _, p = stats.wilcoxon(
                    a2c_results["all_makespans"][:n],
                    res["all_makespans"][:n],
                    alternative="less",
                )
            except Exception:
                p = 1.0
            
            delta = ((res["mean"] - a2c_mean) / a2c_mean) * 100
            sig = "A2C Better" if p < 0.05 else "No Significance"
            conclusion = "✅ Tốt hơn (p<0.05)" if p < 0.05 else "❌ Không có ý nghĩa"
            
            statistical_results[algo] = {
                "p_value": f"{p:.4f}",
                "significance": sig,
                "delta": delta,
            }
            print(
                f"  {algo:<10} {'Wilcoxon Signed-Rank':<30} {p:>10.4f} {conclusion:>22}"
            )

    # GHI VÀO CSV
    
    print(f"\n💾 Ghi kết quả vào CSV: {eval_csv_path}")
    ensure_evaluation_csv_headers(eval_csv_path)

    for algo, res in all_results.items():
        stats_info = statistical_results.get(algo, {})
        
        # Format std (N/A nếu deterministic)
        std_val = "N/A" if res.get("is_deterministic", False) else f"{res['std']:.2f}"
        
        append_evaluation_record(
            csv_path=eval_csv_path,
            timestamp=eval_timestamp,
            instance=instance_name,
            algorithm=algo,
            mean_makespan=res["mean"],
            std_makespan=std_val,
            min_makespan=res["min"],
            max_makespan=res["max"],
            is_deterministic=res.get("is_deterministic", False),
            vs_a2c_percent=stats_info.get("delta", 0.0),
            p_value=stats_info.get("p_value", "-"),
            statistical_significance=stats_info.get("significance", "-"),
        )

    # Charts
    print("\n🎨 Generating charts...")

    if history_path and os.path.exists(history_path):
        history = load_history(history_path)
        plot_learning_curves(
            history,
            title=f"A2C Learning Curves — {instance_name}",
            save_path=os.path.join(output_dir, f"learning_curves_{instance_name}.png"),
        )
    else:
        print("   ⚠️  Bỏ qua learning curves (không có history file)")

    if a2c_results["best_schedule"]:
        plot_gantt(
            schedule=a2c_results["best_schedule"],
            n_machines=info["n_machines"],
            n_jobs=info["n_jobs"],
            title=f"Best A2C Schedule — {instance_name} (Makespan={a2c_results['min']})",
            save_path=os.path.join(output_dir, f"gantt_{instance_name}.png"),
        )

    boxplot_data = {}
    for algo, res in all_results.items():
        samples = res.get("all_makespans", [])
        boxplot_data[algo] = samples

    plot_comparison_boxplot(
        results=boxplot_data,
        title=f"Makespan Distribution — {instance_name}\n(Deterministic: 1 điểm duy nhất)",
        save_path=os.path.join(output_dir, f"boxplot_{instance_name}.png"),
    )

    plot_comparison_bar(
        results=all_results,
        title=f"Mean Makespan Comparison — {instance_name}",
        save_path=os.path.join(output_dir, f"barchart_{instance_name}.png"),
    )

    # Lưu JSON (tương thích code cũ)
    save_path = os.path.join(output_dir, f"eval_results_{instance_name}.json")
    serializable = {
        algo: _to_serializable({k: v for k, v in res.items() if k != "best_schedule"})
        for algo, res in all_results.items()
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Hoàn thành!")
    print(f"   📊 CSV:  {eval_csv_path}")
    print(f"   📈 JSON: {save_path}")
    print(f"   📁 Dir:  {output_dir}/")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate A2C vs Baselines")
    parser.add_argument("--instance", type=str, default="ft06")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--history_path", type=str, default="")
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--eval_csv_path",
        type=str,
        default="results/eval_results.csv",
        help="CSV file to store evaluation results",
    )
    parser.add_argument(
        "--train_background",
        action="store_true",
        help="Train model in background, then evaluate",
    )
    parser.add_argument("--n_episodes", type=int, default=1000)
    args = parser.parse_args()

    if args.train_background:
        print(f"🔄 Mode: TRAIN + EVALUATE")
        _background_train(
            instance_name=args.instance,
            n_episodes=args.n_episodes,
            results_dir=args.output_dir,
        )
        
        model_path = os.path.join(args.output_dir, f"model_{args.instance}.npz")
        history_path = os.path.join(args.output_dir, f"history_{args.instance}.json")
        
        if os.path.exists(model_path):
            run_full_evaluation(
                instance_name=args.instance,
                model_path=model_path,
                history_path=history_path,
                n_runs=args.n_runs,
                output_dir=args.output_dir,
                eval_csv_path=args.eval_csv_path,
            )
        else:
            print(f"❌ Không tìm thấy model: {model_path}")
    else:
        if not args.model_path:
            print("❌ Thiếu --model_path. Ví dụ: --model_path results/model_ft06.npz")
            sys.exit(1)
        
        run_full_evaluation(
            instance_name=args.instance,
            model_path=args.model_path,
            history_path=args.history_path,
            n_runs=args.n_runs,
            output_dir=args.output_dir,
            eval_csv_path=args.eval_csv_path,
        )