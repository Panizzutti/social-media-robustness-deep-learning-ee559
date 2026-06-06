import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os

def plot_trends():
    out_dir = "/scratch/results/alpha_sweep"
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    
    f1_scores = []
    hate_flip_rates = []
    
    # Load the baseline (alpha 0) to compute flips against
    baseline_df = pd.read_csv(os.path.join(out_dir, "predictions_alpha_0.csv"))
    
    for alpha in alphas:
        csv_path = os.path.join(out_dir, f"predictions_alpha_{int(alpha*100)}.csv")
        df = pd.read_csv(csv_path)
        
        # Calculate Macro-F1
        macro_f1 = f1_score(df['true_label'], df['pred_label'], average='macro')
        f1_scores.append(macro_f1)
        
        # Calculate Misogynous -> Clean Flip Rate (Blindness)
        # Compare current predictions against the baseline predictions
        merged = pd.merge(baseline_df[['file_name', 'pred_label']], df[['file_name', 'pred_label']], on='file_name', suffixes=('_clean', '_obs'))
        originally_hate = merged[merged['pred_label_clean'] == 1]
        hate_to_clean = originally_hate[originally_hate['pred_label_obs'] == 0]
        
        flip_rate = len(hate_to_clean) / len(originally_hate) if len(originally_hate) > 0 else 0
        hate_flip_rates.append(flip_rate)

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot F1 Score
    color = 'tab:blue'
    ax1.set_xlabel('Emoji Opacity (Alpha)', fontsize=12)
    ax1.set_ylabel('Macro-F1 Score', color=color, fontsize=12)
    ax1.plot(alphas, f1_scores, marker='o', linewidth=2, color=color, label='Macro-F1 (Higher is better)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.4, 0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Flip Rate on a secondary Y-axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Hate Bypass Rate (Flip %)', color=color, fontsize=12)
    ax2.plot(alphas, [fr * 100 for fr in hate_flip_rates], marker='s', linewidth=2, color=color, linestyle='--', label='Hate Bypass Rate (Lower is safer)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    # Titles and formatting
    plt.title('MemeLens-VLM Robustness vs. Emoji Obfuscation Intensity', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save the plot
    plot_path = "/scratch/results/alpha_sweep/robustness_curve.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    plot_trends()