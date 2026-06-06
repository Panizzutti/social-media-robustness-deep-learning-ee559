import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_BASELINE_NO_TEXT = "/scratch/results/mami_baseline_unseen_randomgeom_no_text_alpha08.csv"
DEFAULT_BASELINE_WITH_TEXT = "/scratch/results/mami_baseline_unseen_randomgeom_with_text_alpha08.csv"
DEFAULT_FINETUNED_NO_TEXT = "/scratch/results/mami_finetuned_unseen_randomgeom_no_text_alpha08.csv"
DEFAULT_FINETUNED_WITH_TEXT = "/scratch/results/mami_finetuned_unseen_randomgeom_with_text_alpha08.csv"

KEY_COLS = ["file_name", "emoji_name", "alpha"]


def macro_f1(df):
    return f1_score(df["true_label"], df["pred_label"], average="macro")


def acc(df):
    return accuracy_score(df["true_label"], df["pred_label"])


def metrics_against_clean(df):
    clean = df[df["alpha"] == 0.0].drop_duplicates(subset=["file_name"])
    attacked = df[df["alpha"] > 0.0]

    if len(clean) == 0:
        raise ValueError("No clean alpha=0.0 baseline rows found.")
    if len(attacked) == 0:
        raise ValueError("No attacked alpha>0.0 rows found.")

    clean_f1 = macro_f1(clean)
    clean_acc = acc(clean)

    rows = []
    for pattern, subset in attacked.groupby("pattern"):
        f1 = macro_f1(subset)
        accuracy = acc(subset)
        rows.append({
            "pattern": pattern,
            "n": len(subset),
            "macro_f1": f1,
            "f1_drop": clean_f1 - f1,
            "accuracy": accuracy,
            "acc_drop": clean_acc - accuracy,
        })

    overall_f1 = macro_f1(attacked)
    overall_acc = acc(attacked)
    overall = {
        "pattern": "ALL_UNSEEN_RANDOM_GEOMETRIES",
        "n": len(attacked),
        "macro_f1": overall_f1,
        "f1_drop": clean_f1 - overall_f1,
        "accuracy": overall_acc,
        "acc_drop": clean_acc - overall_acc,
    }

    return {
        "clean_n": len(clean),
        "attacked_n": len(attacked),
        "clean_macro_f1": clean_f1,
        "clean_accuracy": clean_acc,
        "overall": overall,
        "by_pattern": pd.DataFrame(rows).sort_values("pattern"),
    }


def verify_same_eval_set(base_df, ft_df):
    base_att = base_df[base_df["alpha"] > 0.0].copy()
    ft_att = ft_df[ft_df["alpha"] > 0.0].copy()

    cols = ["file_name", "emoji_name", "alpha", "pattern", "geometry_seed", "placement_seed"]
    missing = [c for c in cols if c not in base_att.columns or c not in ft_att.columns]
    if missing:
        return False, f"Cannot verify: missing columns {missing}"

    base_keys = base_att[cols].sort_values(cols).reset_index(drop=True)
    ft_keys = ft_att[cols].sort_values(cols).reset_index(drop=True)

    if len(base_keys) != len(ft_keys):
        return False, f"Different number of attacked rows: baseline={len(base_keys)}, fine_tuned={len(ft_keys)}"

    same = base_keys.equals(ft_keys)
    if same:
        return True, "Verified: baseline and fine-tuned attacked rows use the same file, emoji, alpha, pattern, geometry_seed, and placement_seed."

    diff_count = (base_keys != ft_keys).any(axis=1).sum()
    return False, f"Evaluation-set mismatch: {diff_count} rows differ after sorting."


def compare_pair(baseline_csv, finetuned_csv, label):
    print("\n" + "=" * 100)
    print(f"UNSEEN RANDOM GEOMETRY COMPARISON: {label}")
    print("=" * 100)

    base_df = pd.read_csv(baseline_csv)
    ft_df = pd.read_csv(finetuned_csv)

    ok, msg = verify_same_eval_set(base_df, ft_df)
    print(msg)

    base_m = metrics_against_clean(base_df)
    ft_m = metrics_against_clean(ft_df)

    summary = pd.DataFrame([
        {"model": "baseline", **base_m["overall"]},
        {"model": "fine_tuned", **ft_m["overall"]},
    ])

    print("\nOverall metrics:")
    print(summary[["model", "n", "macro_f1", "f1_drop", "accuracy", "acc_drop"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nClean baseline metrics:")
    print(f"baseline   clean_n={base_m['clean_n']}  clean_macro_f1={base_m['clean_macro_f1']:.4f}  clean_accuracy={base_m['clean_accuracy']:.4f}")
    print(f"fine_tuned clean_n={ft_m['clean_n']}  clean_macro_f1={ft_m['clean_macro_f1']:.4f}  clean_accuracy={ft_m['clean_accuracy']:.4f}")

    base_pat = base_m["by_pattern"].rename(columns={
        "macro_f1": "baseline_f1",
        "f1_drop": "baseline_drop",
        "accuracy": "baseline_acc",
        "acc_drop": "baseline_acc_drop",
        "n": "baseline_n",
    })
    ft_pat = ft_m["by_pattern"].rename(columns={
        "macro_f1": "finetuned_f1",
        "f1_drop": "finetuned_drop",
        "accuracy": "finetuned_acc",
        "acc_drop": "finetuned_acc_drop",
        "n": "finetuned_n",
    })

    merged = pd.merge(base_pat, ft_pat, on="pattern", how="outer")
    merged["drop_reduction"] = merged["baseline_drop"] - merged["finetuned_drop"]
    merged["f1_gain_on_attacked"] = merged["finetuned_f1"] - merged["baseline_f1"]

    print("\nBy unseen geometry:")
    cols = [
        "pattern", "baseline_n", "baseline_f1", "baseline_drop",
        "finetuned_f1", "finetuned_drop", "drop_reduction", "f1_gain_on_attacked"
    ]
    print(merged[cols].sort_values("pattern").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    overall_drop_reduction = base_m["overall"]["f1_drop"] - ft_m["overall"]["f1_drop"]
    overall_attacked_gain = ft_m["overall"]["macro_f1"] - base_m["overall"]["macro_f1"]

    print("\nHeadline:")
    print(f"Attacked Macro-F1 gain: {overall_attacked_gain:+.4f}")
    print(f"F1-drop reduction:     {overall_drop_reduction:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model on the same unseen random geometry evaluation set.")
    parser.add_argument("--baseline_no_text", default=DEFAULT_BASELINE_NO_TEXT)
    parser.add_argument("--finetuned_no_text", default=DEFAULT_FINETUNED_NO_TEXT)
    parser.add_argument("--baseline_with_text", default=DEFAULT_BASELINE_WITH_TEXT)
    parser.add_argument("--finetuned_with_text", default=DEFAULT_FINETUNED_WITH_TEXT)
    parser.add_argument("--mode", choices=["no_text", "with_text", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ["no_text", "both"]:
        compare_pair(args.baseline_no_text, args.finetuned_no_text, "NO TEXT")

    if args.mode in ["with_text", "both"]:
        compare_pair(args.baseline_with_text, args.finetuned_with_text, "WITH TEXT")


if __name__ == "__main__":
    main()
