import os
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_sft_convergence(state_file_path, output_png_path):
    with open(state_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    log_history = data.get("log_history", []) if isinstance(data, dict) else data

    train_steps, train_losses, learning_rates, grad_norms = [], [], [], []
    eval_steps, eval_losses, eval_perplexities = [], [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        
        # Training logs
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
            if "learning_rate" in entry:
                learning_rates.append(entry["learning_rate"])
            if "grad_norm" in entry:
                grad_norms.append(entry["grad_norm"])

        # Evaluation logs
        if "eval_loss" in entry:
            eval_steps.append(step)
            e_loss = entry["eval_loss"]
            eval_losses.append(e_loss)
            try:
                ppl = math.exp(min(e_loss, 20.0))
            except OverflowError:
                ppl = float("nan")
            eval_perplexities.append(ppl)

    # Styling setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.suptitle("Stage 7C: Supervised Fine-Tuning (SFT) Convergence & Optimization Diagnostics", fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(train_steps, train_losses, color='#1f77b4', marker='o', markersize=3, linewidth=1.8, label="Train Loss (Step)")
    if len(train_losses) >= 5:
        # Rolling EMA
        alpha = 0.3
        ema = [train_losses[0]]
        for val in train_losses[1:]:
            ema.append(alpha * val + (1 - alpha) * ema[-1])
        ax1.plot(train_steps, ema, color='#d62728', linestyle='--', linewidth=2.0, label="Smoothed Trend (EMA)")
    ax1.set_title("Training Loss Trajectory", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Optimizer Steps", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Loss & Perplexity
    ax2 = axes[0, 1]
    if eval_steps:
        color_loss = '#2ca02c'
        color_ppl = '#9467bd'
        ax2.plot(eval_steps, eval_losses, color=color_loss, marker='s', markersize=5, linewidth=2.0, label="Eval Loss")
        ax2.set_title("Generalization Validation Curve", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Optimizer Steps", fontsize=10)
        ax2.set_ylabel("Validation Loss", fontsize=10, color=color_loss)
        ax2.tick_params(axis='y', labelcolor=color_loss)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # Secondary Y axis for Perplexity
        ax2_sub = ax2.twinx()
        ax2_sub.plot(eval_steps, eval_perplexities, color=color_ppl, marker='^', markersize=5, linestyle=':', linewidth=1.8, label="Eval Perplexity")
        ax2_sub.set_ylabel("Perplexity (exp(Loss))", fontsize=10, color=color_ppl)
        ax2_sub.tick_params(axis='y', labelcolor=color_ppl)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_sub.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    # Panel 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    if learning_rates:
        ax3.plot(train_steps[:len(learning_rates)], [lr * 1e4 for lr in learning_rates], color='#ff7f0e', linewidth=2.0, label="Learning Rate")
        ax3.set_title("Learning Rate Schedule (Warmup + Cosine Decay)", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Optimizer Steps", fontsize=10)
        ax3.set_ylabel("Learning Rate (×10⁻⁴)", fontsize=10)
        ax3.legend(loc="upper right", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Gradient Norm Dynamics
    ax4 = axes[1, 1]
    if grad_norms:
        ax4.plot(train_steps[:len(grad_norms)], grad_norms, color='#8c564b', marker='.', markersize=4, linewidth=1.2, label="Gradient Norm")
        ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label="Nominal Threshold (1.0)")
        ax4.set_title("Gradient Norm Dynamics (Stability Check)", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Optimizer Steps", fontsize=10)
        ax4.set_ylabel("L2 Norm (||g||)", fontsize=10)
        ax4.legend(loc="upper right", frameon=True)
        ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()

    print("Plot successfully saved to:", output_png_path)

if __name__ == "__main__":
    state = r"C:\Users\DS\.gemini\antigravity\scratch\test_sft_plot\trainer_state.json"
    out = r"C:\Users\DS\.gemini\antigravity\scratch\test_sft_plot\sft_convergence_curves.png"
    plot_sft_convergence(state, out)
