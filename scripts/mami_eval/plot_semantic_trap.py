import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def calculate_flip_rates(df_clean, df_obs):
    """Calculates Hallucination (Clean->Misogynous) and Masking (Misogynous->Clean) rates."""
    merged = pd.merge(
        df_obs[['file_name', 'true_label', 'pred_label']],
        df_clean[['file_name', 'pred_label']],
        on='file_name',
        suffixes=('_obs', '_clean')
    )
    
    if len(merged) == 0:
        return 0, 0, 0
        
    f1 = f1_score(merged['true_label'], merged['pred_label_obs'], average='macro')

    orig_clean = merged[merged['pred_label_clean'] == 0]
    hallucination_rate = len(orig_clean[orig_clean['pred_label_obs'] == 1]) / len(orig_clean) * 100 if len(orig_clean) > 0 else 0
    
    orig_misog = merged[merged['pred_label_clean'] == 1]
    masking_rate = len(orig_misog[orig_misog['pred_label_obs'] == 0]) / len(orig_misog) * 100 if len(orig_misog) > 0 else 0
    
    return f1, hallucination_rate, masking_rate

def main():
    # 1. Load Data
    csv_path = "/scratch/results/mami_massive_sweep_with_text_results.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Isolate Baseline and Target Opacity
    df_clean = df[df['alpha'] == 0.0].drop_duplicates(subset=['file_name'])
    base_f1 = f1_score(df_clean['true_label'], df_clean['pred_label'], average='macro')
    
    alpha_target = 0.8
    df_target = df[df['alpha'] == alpha_target]

    # 3. Calculate metrics per emoji
    emoji_data = []
    for emoji in df_target['emoji_name'].unique():
        if emoji == 'none': 
            continue
            
        subset = df_target[df_target['emoji_name'] == emoji]
        f1, hallucination, masking = calculate_flip_rates(df_clean, subset)
        f1_drop = base_f1 - f1
        
        emoji_data.append({
            'Emoji': emoji.replace('_', ' ').title(), # Clean up names for the plot
            'F1 Drop': f1_drop,
            'Hallucinated Hate (Clean -> Misogynous)': hallucination,
            'Masked Hate (Misogynous -> Clean)': masking
        })

    # Convert to DataFrame and sort by Hallucination Rate to get the Top 5
    df_emojis = pd.DataFrame(emoji_data)
    df_top5 = df_emojis.sort_values(by='Hallucinated Hate (Clean -> Misogynous)', ascending=False).head(5)

    # Set Emoji as the index for easy pandas grouped bar plotting
    df_plot = df_top5.set_index('Emoji')[['Hallucinated Hate (Clean -> Misogynous)', 'Masked Hate (Misogynous -> Clean)']]

    # 4. Build the Plot using pure Matplotlib / Pandas
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Choose highly contrasting colors (Red for Hallucination, Grey for Masking)
    colors = ["#d62728", "#7f7f7f"]
    
    # Create the grouped bar chart directly from the dataframe
    df_plot.plot(kind='bar', ax=ax, color=colors, edgecolor="black", width=0.8, linewidth=1.5)

    # 5. Formatting & Labels
    plt.title("The Semantic Trap: Emoji-Specific Disruption Profiles\n(Text Injected, Alpha = 0.8)", 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Semantic Distractor (Emoji)", fontsize=14, fontweight='bold')
    plt.ylabel("Flip Rate (%)", fontsize=14, fontweight='bold')
    
    # Fix x-axis ticks to sit horizontally instead of turning 90 degrees
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Add a horizontal grid behind the bars for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # Ensure grid is *behind* bars
    
    # Format Y-axis to show percentage signs
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    
    # Add headroom for the legend
    max_val = df_plot.values.max()
    plt.ylim(0, max_val * 1.15) 

    # Clean up the legend
    plt.legend(title="", fontsize=12, loc='upper right')

    # Add data labels on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=11, fontweight='bold', 
                        xytext=(0, 5), textcoords='offset points')

    # 6. Save the Plot
    output_img = "/scratch/results/plot3_semantic_trap_alpha08.png"
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_img}")

if __name__ == "__main__":
    main()