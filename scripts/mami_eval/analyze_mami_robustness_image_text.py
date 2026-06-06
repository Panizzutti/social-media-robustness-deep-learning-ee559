import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def analyze_robustness():
    # Updated file names to reflect the transcription injection
    clean_csv = "/scratch/results/mami_reinstated_text_prompts_clean.csv"
    obs_csv = "/scratch/results/mami_reinstated_text_prompts.csv"

    print("Loading predictions...")
    df_clean = pd.read_csv(clean_csv)
    df_obs = pd.read_csv(obs_csv)

    # Rename columns to avoid confusion before merging
    df_clean = df_clean.rename(columns={"pred_label": "clean_pred"})
    df_obs = df_obs.rename(columns={"pred_label": "obs_pred"})

    # Merge on the file_name
    df_merged = pd.merge(df_clean[['file_name', 'true_label', 'clean_pred']], 
                         df_obs[['file_name', 'obs_pred']], 
                         on='file_name', how='inner')

    total_images = len(df_merged)
    
    # 1. Standard Metrics
    clean_f1 = f1_score(df_merged['true_label'], df_merged['clean_pred'], average='macro')
    obs_f1 = f1_score(df_merged['true_label'], df_merged['obs_pred'], average='macro')
    
    clean_acc = accuracy_score(df_merged['true_label'], df_merged['clean_pred'])
    obs_acc = accuracy_score(df_merged['true_label'], df_merged['obs_pred'])

    # 2. Flip Rate Analysis
    # Total flips (any change in prediction)
    flips = df_merged[df_merged['clean_pred'] != df_merged['obs_pred']]
    total_flip_rate = len(flips) / total_images

    # Hate to Non-Hate Flips (Model was blinded to the misogyny)
    originally_hate = df_merged[df_merged['clean_pred'] == 1]
    hate_to_clean = originally_hate[originally_hate['obs_pred'] == 0]
    hate_flip_rate = len(hate_to_clean) / len(originally_hate) if len(originally_hate) > 0 else 0

    # Non-Hate to Hate Flips (Emojis caused a hallucination)
    originally_clean = df_merged[df_merged['clean_pred'] == 0]
    clean_to_hate = originally_clean[originally_clean['obs_pred'] == 1]
    clean_flip_rate = len(clean_to_hate) / len(originally_clean) if len(originally_clean) > 0 else 0

    # 3. Print Final Report
    print("\n" + "="*55)
    print("MAMI ROBUSTNESS ANALYSIS (WITH TEXT TRANSCRIPTIONS)")
    print("="*55)
    print(f"Total Images Analyzed: {total_images}")
    print("-" * 55)
    print(f"Clean Macro-F1:       {clean_f1:.4f}")
    print(f"Obfuscated Macro-F1:  {obs_f1:.4f}")
    print(f"Macro-F1 Drop:        {clean_f1 - obs_f1:.4f}  <-- THE HEADLINE METRIC")
    print("-" * 55)
    print(f"Clean Accuracy:       {clean_acc:.4f}")
    print(f"Obfuscated Accuracy:  {obs_acc:.4f}")
    print(f"Accuracy Drop:        {clean_acc - obs_acc:.4f}")
    print("-" * 55)
    print("FLIP RATES (Vulnerability to Obfuscation):")
    print(f"Total Flip Rate:              {total_flip_rate*100:.1f}%")
    print(f"Misogynous -> Clean Flip:     {hate_flip_rate*100:.1f}%  (Failed to detect hate)")
    print(f"Clean -> Misogynous Flip:     {clean_flip_rate*100:.1f}%  (Hallucinated hate)")
    print("="*55)

if __name__ == "__main__":
    analyze_robustness()