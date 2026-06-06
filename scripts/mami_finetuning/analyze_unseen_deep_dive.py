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

def analyze_unseen_sweep(file_path, sweep_name):
    print(f"\n{'='*85}")
    print(f"  DEEP DIVE REPORT: {sweep_name}")
    print(f"{'='*85}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[!] File not found: {file_path}. Have you generated it yet?")
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
    
    if len(df_obs) == 0:
        print("No attacked (alpha=0.8) data found.")
        return

    # Overall Severity Degradation
    acc_ov, f1_ov, c2m_ov, m2c_ov = calculate_metrics(df_clean, df_obs)
    f1_drop_ov = base_f1 - f1_ov
    
    print("--- 2. OVERALL ATTACK DEGRADATION (Fixed Alpha = 0.8) ---")
    print(f"Attacked Macro-F1 : {f1_ov:.4f} (F1 Drop: {f1_drop_ov:+.4f})")
    print(f"Hallucinated Hate : {c2m_ov*100:.1f}% (Clean memes misclassified as Misogynous)")
    print(f"Masked Hate       : {m2c_ov*100:.1f}% (Misogynous memes misclassified as Clean)")
    print("-" * 85)

    # --- Pattern Vulnerability ---
    print("\n--- 3. UNSEEN PATTERN VULNERABILITY ---")
    print("Evaluates robustness across previously unencountered geometries.")
    for pattern in sorted(df_obs['pattern'].unique()):
        if pattern == "none": continue
        subset = df_obs[df_obs['pattern'] == pattern]
        acc, f1, c2m, m2c = calculate_metrics(df_clean, subset)
        f1_drop = base_f1 - f1
        print(f"   Pattern: {pattern:<22} | F1 Drop: {f1_drop:+.4f} | Hallucinated: {c2m*100:>5.1f}% | Masked: {m2c*100:>5.1f}%")

    # --- Top 5 Disruptive Emojis ---
    print("\n--- 4. TOP 5 SEMANTIC DISTRACTORS (Ranked by F1 Drop) ---")
    print("Evaluates which emojis most successfully hijacked the model's logic.")
    emoji_stats = []
    for emoji in df_obs['emoji_name'].unique():
        if emoji == "none": continue
        subset = df_obs[df_obs['emoji_name'] == emoji]
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
    print("=" * 85 + "\n")

def main():
    # File Paths
    base_no_text = "/scratch/results/mami_baseline_unseen_randomgeom_no_text_alpha08.csv"
    ft_no_text = "/scratch/results/mami_finetuned_unseen_randomgeom_no_text_alpha08.csv"
    
    base_with_text = "/scratch/results/mami_baseline_unseen_randomgeom_with_text_alpha08.csv"
    ft_with_text = "/scratch/results/mami_finetuned_unseen_randomgeom_with_text_alpha08.csv"

    # Run the deep dive sequentially for a full, continuous terminal report
    analyze_unseen_sweep(base_no_text, "BASELINE MODEL - NO TEXT (Unseen Geometries)")
    analyze_unseen_sweep(ft_no_text, "FINE-TUNED MODEL - NO TEXT (Unseen Geometries)")
    
    analyze_unseen_sweep(base_with_text, "BASELINE MODEL - WITH TEXT (Unseen Geometries)")
    analyze_unseen_sweep(ft_with_text, "FINE-TUNED MODEL - WITH TEXT (Unseen Geometries)")

if __name__ == "__main__":
    main()