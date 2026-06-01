
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves(csv_path: str):
    """Load CSV và vẽ 3 biểu đồ learning curve."""
    
    # Load
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}\n")
    
    # Parse
    episodes = df['episode'].values
    rewards = df['reward'].values
    makespans = df['makespan'].values
    best_makespans = df['best_makespan'].values
    entropies = df['entropy'].values
    entropy_coefs = df['entropy_coef'].values
    rolling_evals = df['rolling_eval_ms'].values
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('A2C Learning Curves — Diagnostic Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Reward
    ax = axes[0, 0]
    ax.plot(episodes, rewards, 'b-', alpha=0.5, label='Raw reward')
    ax.plot(episodes, pd.Series(rewards).rolling(100).mean(), 'b-', linewidth=2, label='Moving avg (100)')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('1. Episode Reward Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Makespan (training episode vs best eval)
    ax = axes[0, 1]
    ax.scatter(episodes, makespans, s=10, alpha=0.3, label='Episode makespan', color='red')
    ax.plot(episodes, best_makespans, 'g-', linewidth=2, label='Best eval score')
    if np.any(rolling_evals > 0):
        eval_episodes = episodes[rolling_evals > 0]
        eval_scores = rolling_evals[rolling_evals > 0]
        ax.scatter(eval_episodes, eval_scores, s=80, alpha=0.6, color='purple', marker='*', label='Eval checkpoints')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Makespan')
    ax.set_title('2. Makespan Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Entropy Decay
    ax = axes[1, 0]
    ax.plot(episodes, entropy_coefs, 'r-', linewidth=2, label='Entropy coefficient')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy Coef')
    ax.set_title('3. Entropy Coefficient Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Statistics Summary
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    📊 TRAINING STATISTICS
    
    Total episodes: {len(df)}
    
    📈 Reward:
       Min: {rewards.min():.1f}
       Max: {rewards.max():.1f}
       Mean: {rewards.mean():.1f}
       Final 100-ep mean: {rewards[-100:].mean():.1f}
    
    🎯 Makespan:
       Best training: {makespans.min():.0f}
       Best eval: {best_makespans.min():.0f}
       Final: {makespans[-1]:.0f}
    
    🔀 Entropy:
       Start: {entropy_coefs[0]:.4f}
       End: {entropy_coefs[-1]:.4f}
       Decay rate: {entropy_coefs[-1]/entropy_coefs[0]:.2%}
    
    ⚡ Convergence:
       Episodes since last best: {len(df) - np.where(best_makespans == best_makespans.min())[0][-1]}
       Reward trend (last 200): {'↑ improving' if rewards[-200:].mean() > rewards[-400:-200].mean() else '→ flat' if abs(rewards[-200:].mean() - rewards[-400:-200].mean()) < 5 else '↓ declining'}
    """
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    out_path = Path(csv_path).parent / "diagnostic_learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved → {out_path}\n")
    
    # Print summary
    print("="*60)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Best makespan (training): {makespans.min():.0f}")
    print(f"Best eval score: {best_makespans.min():.0f}")
    print(f"Mean reward (last 100): {rewards[-100:].mean():.1f}")
    print(f"Entropy decay: {entropy_coefs[0]:.4f} → {entropy_coefs[-1]:.4f}")
    print(f"Episodes since improvement: {len(df) - np.where(best_makespans == best_makespans.min())[0][-1]}")
    
    # Diagnostic warnings
    print("\n🔎 DIAGNOSTIC CHECKS:")
    
    if rewards[-100:].mean() > rewards[-400:-200].mean():
        print("✅ Reward improving (good sign)")
    else:
        print("⚠️  Reward plateaued — policy may have converged early")
    
    if best_makespans[-1] < best_makespans[-100]:
        print("✅ Best eval improving throughout training")
    else:
        print("⚠️  Best eval plateaued — no new improvements recently")
    
    if entropy_coefs[-1] < entropy_coefs[0] / 5:
        print("✅ Entropy decayed appropriately")
    else:
        print("⚠️  Entropy still high — may need stronger decay")
    
    if makespans.min() < best_makespans.min() * 1.2:
        print("✅ Training episode best is competitive with eval best")
    else:
        print("⚠️  Training best >> eval best — likely overfit or luck")
    
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostic.py <csv_path>")
        print("Example: python diagnostic.py results/training_log_ft06.csv")
        sys.exit(1)
    
    plot_learning_curves(sys.argv[1])