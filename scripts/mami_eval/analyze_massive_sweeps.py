import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(df_clean, df_obs):
    """
    Helper function to calculate F1, Accuracy, and Flip Rates 
    by joining obfuscated results with their clean baselines.
    """
    merged = pd.merge(
        df_obs[['file_name', 'true_label', 'pred_label']],
        df_clean[['file_name', 'pred_label']],
        on='file_name',
        suffixes=('_obs', '_clean')
    )
    
    if len(merged) == 0:
        return 0, 0, 0, 0

    acc = accuracy_score(merged['true_label'], merged['pred_label_obs'])
    f1 = f1_score(merged['true_label'], merged['pred_label_obs'], average='macro')
    
    orig_clean = merged[merged['pred_label_clean'] == 0]
    clean_to_misog = len(orig_clean[orig_clean['pred_label_obs'] == 1]) / len(orig_clean) if len(orig_clean) > 0 else 0
    
    orig_misog = merged[merged['pred_label_clean'] == 1]
    misog_to_clean = len(orig_misog[orig_misog['pred_label_obs'] == 0]) / len(orig_misog) if len(orig_misog) > 0 else 0
    
    return acc, f1, clean_to_misog, misog_to_clean

def report_detailed_metrics_at_alpha(df_obs, df_clean, alpha_target, base_f1):
    """
    Runs the deep-dive spatial and semantic analysis for a specific alpha level.
    """
    df_target = df_obs[df_obs['alpha'] == alpha_target]
    if len(df_target) == 0:
        print(f"\n[!] No data found for alpha = {alpha_target}. Skipping deep dive.")
        return

    print(f"\n>>> DEEP DIVE: OPACITY (ALPHA) = {alpha_target} <<<")
    print("This evaluates failures where human visibility differs from model visibility.")
    print("-" * 85)
    
    # --- Pattern Vulnerability ---
    print("1. PATTERN VULNERABILITY:")
    for pattern in df_target['pattern'].unique():
        subset = df_target[df_target['pattern'] == pattern]
        acc, f1, c2m, m2c = calculate_metrics(df_clean, subset)
        f1_drop = base_f1 - f1
        print(f"   Pattern: {pattern:<12} | F1 Drop: {f1_drop:+.4f} | Hallucinated Hate: {c2m*100:5.1f}% | Masked Hate: {m2c*100:5.1f}%")

    # --- Top 5 Disruptive Emojis ---
    print("\n2. TOP 5 SEMANTIC DISTRACTORS (Ranked by F1 Drop):")
    emoji_stats = []
    for emoji in df_target['emoji_name'].unique():
        subset = df_target[df_target['emoji_name'] == emoji]
        acc, f1, c2m, m2c = calculate_metrics(df_clean, subset)
        f1_drop = base_f1 - f1
        emoji_stats.append({
            'emoji': emoji,
            'f1_drop': f1_drop,
            'hallucination_rate': c2m * 100,
            'masked_rate': m2c * 100
        })
    
    # Sort by the largest drop in F1
    emoji_stats.sort(key=lambda x: x['f1_drop'], reverse=True)
    
    print(f"   {'Emoji Name':<15} | {'F1 Drop':<10} | {'Hallucinated Hate (C->M)':<26} | {'Masked Hate (M->C)':<20}")
    print("   " + "-" * 78)
    for stat in emoji_stats[:5]:
        print(f"   {stat['emoji']:<15} | {stat['f1_drop']:+.4f}   | {stat['hallucination_rate']:>10.1f}%                | {stat['masked_rate']:>10.1f}%")
    print("=" * 85)


def analyze_sweep(file_path, sweep_name):
    print(f"\n{'='*85}")
    print(f"  ANALYSIS REPORT: {sweep_name}")
    print(f"{'='*85}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Extract Clean Baseline (alpha == 0.0)
    df_clean = df[df['alpha'] == 0.0].drop_duplicates(subset=['file_name'])
    
    if len(df_clean) == 0:
        print("No baseline (alpha=0.0) data found. Cannot compute flip rates.")
        return
        
    base_acc = accuracy_score(df_clean['true_label'], df_clean['pred_label'])
    base_f1 = f1_score(df_clean['true_label'], df_clean['pred_label'], average='macro')
    
    print("--- 1. BASELINE PERFORMANCE (Clean Images) ---")
    print(f"Total Unique Images : {len(df_clean)}")
    print(f"Baseline Accuracy   : {base_acc:.4f}")
    print(f"Baseline Macro-F1   : {base_f1:.4f}")
    print("-" * 85)

    df_obs = df[df['alpha'] > 0.0]

    # Overall Severity Degradation
    print("--- 2. SEVERITY DEGRADATION (Aggregated by Alpha) ---")
    print(f"{'Alpha':<10} | {'Macro-F1':<10} | {'F1 Drop':<10} | {'Hallucinated (C->M)':<20} | {'Masked (M->C)':<20}")
    print("-" * 85)
    for alpha in sorted(df_obs['alpha'].unique()):
        subset = df_obs[df_obs['alpha'] == alpha]
        acc, f1, c2m, m2c = calculate_metrics(df_clean, subset)
        f1_drop = base_f1 - f1
        print(f"{alpha:<10.1f} | {f1:.4f}     | {f1_drop:+.4f}   | {c2m*100:>10.1f}%          | {m2c*100:>10.1f}%")
    print("-" * 85)

    # --- Run Detailed Metrics for Specific Opacities ---
    report_detailed_metrics_at_alpha(df_obs, df_clean, alpha_target=0.8, base_f1=base_f1)
    report_detailed_metrics_at_alpha(df_obs, df_clean, alpha_target=1.0, base_f1=base_f1)

def main():
    no_text_csv = "/scratch/results/mami_massive_sweep_results.csv"
    text_csv = "/scratch/results/mami_massive_sweep_with_text_results.csv"
    
    analyze_sweep(no_text_csv, "OCR COMPOUND ATTACK (No Text Injection)")
    analyze_sweep(text_csv, "PURE SEMANTIC ATTACK (With Text Injection)")

if __name__ == "__main__":
    main()
