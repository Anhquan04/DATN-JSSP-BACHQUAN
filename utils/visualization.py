
import matplotlib
matplotlib.use("Agg")   # Chạy không cần display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional
import json
import os


# Palette màu đẹp cho biểu đồ
COLORS = {
    "a2c"    : "#2196F3",   # Blue
    "fifo"   : "#FF9800",   # Orange
    "spt"    : "#4CAF50",   # Green
    "lpt"    : "#9C27B0",   # Purple
    "edd"    : "#F44336",   # Red
    "random" : "#607D8B",   # Gray
    "ppo"    : "#00BCD4",   # Cyan
}


#  1. LEARNING CURVES

def plot_learning_curves(
    history: Dict,
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
    window: int = 50,
):
    """
    Vẽ reward và makespan theo episode.

    Args:
        history  : Dict từ training (keys: episode_rewards, makespans, ...)
        title    : Tiêu đề biểu đồ
        save_path: Đường dẫn lưu file ảnh
        window   : Cửa sổ moving average
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    episodes = range(1, len(history["episode_rewards"]) + 1)

    # ── Plot 1: Episode Reward 
    ax = axes[0, 0]
    rewards = history["episode_rewards"]
    ma = _moving_average(rewards, window)
    ax.plot(episodes, rewards, alpha=0.3, color=COLORS["a2c"], linewidth=0.8)
    ax.plot(range(window, len(rewards) + 1), ma,
            color=COLORS["a2c"], linewidth=2, label=f"MA({window})")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot 2: Makespan per Episode 
    ax = axes[0, 1]
    makespans = history["makespans"]
    ma_ms = _moving_average(makespans, window)
    ax.plot(episodes, makespans, alpha=0.3, color=COLORS["fifo"], linewidth=0.8)
    ax.plot(range(window, len(makespans) + 1), ma_ms,
            color=COLORS["fifo"], linewidth=2, label=f"MA({window})")
    ax.set_title("Makespan per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Makespan")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot 3: Actor & Critic Loss 
    ax = axes[1, 0]
    if history.get("actor_losses"):
        ax.plot(history["actor_losses"], label="Actor Loss",
                color=COLORS["a2c"], linewidth=1.2)
        ax.plot(history["critic_losses"], label="Critic Loss",
                color=COLORS["fifo"], linewidth=1.2)
        ax.set_title("Training Losses")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    # ── Plot 4: Entropy 
    ax = axes[1, 1]
    if history.get("entropies"):
        ax.plot(history["entropies"], color=COLORS["spt"], linewidth=1.2)
        ax.set_title("Policy Entropy (Exploration)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Entropy")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


#  2. GANTT CHART

def plot_gantt(
    schedule: List[Dict],
    n_machines: int,
    n_jobs: int,
    title: str = "Gantt Chart",
    save_path: Optional[str] = None,
):
    """
    Vẽ Gantt chart cho lịch sản xuất.

    Args:
        schedule  : List các entry {job, machine, start, end}
        n_machines: Số máy
        n_jobs    : Số jobs (dùng để tô màu)
        title     : Tiêu đề
        save_path : Đường dẫn lưu ảnh
    """
    # Tạo màu riêng cho mỗi job
    job_colors = plt.cm.Set3(np.linspace(0, 1, n_jobs))

    fig, ax = plt.subplots(figsize=(max(10, len(schedule) // 2), n_machines + 1))

    for entry in schedule:
        machine = entry["machine"]
        start   = entry["start"]
        duration = entry["end"] - entry["start"]
        job_id  = entry["job"]

        # Vẽ bar nằm ngang
        bar = ax.barh(
            y=machine,
            width=duration,
            left=start,
            color=job_colors[job_id],
            edgecolor="black",
            linewidth=0.8,
            height=0.6,
        )
        # Label Job ID ở giữa bar
        ax.text(
            start + duration / 2,
            machine,
            f"J{job_id}",
            ha="center", va="center",
            fontsize=8, fontweight="bold"
        )

    # Legend
    patches = [
        mpatches.Patch(color=job_colors[j], label=f"Job {j}")
        for j in range(n_jobs)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    # Formatting
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Makespan line
    makespan = max(e["end"] for e in schedule)
    ax.axvline(x=makespan, color="red", linestyle="--", linewidth=1.5,
               label=f"Makespan = {makespan}")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


#  3. COMPARISON CHARTS

def plot_comparison_boxplot(
    results: Dict[str, List[float]],
    title: str = "Makespan Comparison",
    save_path: Optional[str] = None,
):
    """
    Box plot so sánh makespan của các algorithm.

    Args:
        results  : {algorithm_name: [makespan_run1, makespan_run2, ...]}
        title    : Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(results.keys())
    data  = [results[n] for n in names]
    colors = [COLORS.get(n.lower(), "#999999") for n in names]

    bp = ax.boxplot(data, patch_artist=True, labels=names,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Makespan")
    ax.set_xlabel("Algorithm")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_comparison_bar(
    results: Dict[str, Dict],
    title: str = "Mean Makespan Comparison",
    save_path: Optional[str] = None,
):
    """
    Bar chart so sánh trung bình makespan.

    Args:
        results  : {algo_name: {mean, std, min, max}}
        title    : Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    names  = list(results.keys())
    means  = [results[n]["mean"] for n in names]
    stds   = [results[n]["std"] for n in names]
    colors = [COLORS.get(n.lower(), "#999999") for n in names]

    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor="black")

    # Ghi số lên đầu mỗi cột
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.1,
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Mean Makespan")
    ax.set_xlabel("Algorithm")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved: {save_path}")
    plt.close()


#  HELPER

def _moving_average(data: List[float], window: int) -> List[float]:
    """Tính moving average."""
    return [
        np.mean(data[max(0, i - window + 1): i + 1])
        for i in range(window - 1, len(data))
    ]


def load_history(path: str) -> Dict:
    """Load training history từ JSON."""
    with open(path) as f:
        return json.load(f)
