"""
Baseline Algorithms — Dispatching Rules
=========================================
Cài đặt các quy tắc ưu tiên (Priority Dispatching Rules) để so sánh với A2C.

Các baseline:
  - FIFO : First In, First Out — chọn job đến trước
  - SPT  : Shortest Processing Time — chọn operation ngắn nhất
  - EDD  : Earliest Due Date — chọn job có deadline sớm nhất
  - LPT  : Longest Processing Time — chọn operation dài nhất
  - RANDOM: Ngẫu nhiên (lower bound để so sánh)
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.job_shop_env import JobShopEnv


def run_dispatching_rule(jobs_data, rule: str = "fifo", seed: int = 42) -> Dict:
    """
    Chạy một lần scheduling với quy tắc cho trước.

    Args:
        jobs_data: Dữ liệu JSSP
        rule     : 'fifo' | 'spt' | 'edd' | 'lpt' | 'random'
        seed     : Random seed (chỉ dùng cho 'random')

    Returns:
        dict: makespan, idle_time, utilization, schedule
    """
    env = JobShopEnv(jobs_data)
    state, _ = env.reset()
    rng = np.random.default_rng(seed)

    done = False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break

        action = _select_action(env, valid, rule, rng)
        _, _, done, _, info = env.step(action)

    return {
        "makespan"   : env.get_makespan(),
        "idle_time"  : env.get_idle_time(),
        "utilization": env.get_utilization(),
        "schedule"   : env.schedule,
        "rule"       : rule,
    }


def _select_action(env: JobShopEnv, valid: List[int], rule: str, rng) -> int:
    """Chọn action dựa trên quy tắc."""

    if rule == "fifo":
        # FIFO: ưu tiên job có index nhỏ nhất (job đến trước)
        return min(valid)

    elif rule == "spt":
        # SPT: ưu tiên operation có processing time ngắn nhất
        def spt_key(job_id):
            op_idx = env.job_op_index[job_id]
            _, proc_time = env.jobs_data[job_id][op_idx]
            return proc_time
        return min(valid, key=spt_key)

    elif rule == "lpt":
        # LPT: ưu tiên operation có processing time dài nhất
        def lpt_key(job_id):
            op_idx = env.job_op_index[job_id]
            _, proc_time = env.jobs_data[job_id][op_idx]
            return proc_time
        return max(valid, key=lpt_key)

    elif rule == "edd":
        # EDD: ưu tiên job có thời gian hoàn thành dự kiến sớm nhất
        # (Ước tính: job_available_at + tổng thời gian còn lại)
        def edd_key(job_id):
            op_idx = env.job_op_index[job_id]
            remaining = sum(
                t for _, t in env.jobs_data[job_id][op_idx:]
            )
            return env.job_available_at[job_id] + remaining
        return min(valid, key=edd_key)

    elif rule == "random":
        return int(rng.choice(valid))

    else:
        raise ValueError(f"Unknown rule: {rule}. Dùng: 'fifo', 'spt', 'lpt', 'edd', 'random'")


def evaluate_all_baselines(
    jobs_data,
    n_runs: int = 30
) -> Dict[str, Dict]:
    """
    Chạy tất cả baseline n_runs lần và tính thống kê.

    Note: FIFO/SPT/EDD/LPT là deterministic → 1 run thôi.
          RANDOM mới cần n_runs.

    Returns:
        results: {rule_name: {mean, std, min, max, all_makespans}}
    """
    results = {}
    rules = ["fifo", "spt", "lpt", "edd", "random"]

    for rule in rules:
        makespans = []

        # Deterministic rules: chỉ 1 run
        if rule in ["fifo", "spt", "lpt", "edd"]:
            r = run_dispatching_rule(jobs_data, rule)
            makespans = [r["makespan"]] * n_runs  # Lặp để thống kê cùng format
        else:
            # Random: chạy n_runs lần
            for seed in range(n_runs):
                r = run_dispatching_rule(jobs_data, rule, seed=seed)
                makespans.append(r["makespan"])

        results[rule] = {
            "mean"        : np.mean(makespans),
            "std"         : np.std(makespans),
            "min"         : np.min(makespans),
            "max"         : np.max(makespans),
            "all_makespans": makespans,
        }

    return results


def print_baseline_results(results: Dict):
    """In bảng kết quả đẹp ra console."""
    print("\n" + "═"*60)
    print("  BASELINE RESULTS")
    print("═"*60)
    print(f"{'Rule':<10} {'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7}")
    print("─"*60)
    for rule, stats in results.items():
        print(
            f"{rule.upper():<10} "
            f"{stats['mean']:>8.2f} "
            f"{stats['std']:>7.2f} "
            f"{stats['min']:>7.0f} "
            f"{stats['max']:>7.0f}"
        )
    print("═"*60)
