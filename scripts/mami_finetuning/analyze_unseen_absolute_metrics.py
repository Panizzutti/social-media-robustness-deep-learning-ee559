import argparse
import csv
from pathlib import Path


DEFAULT_BASELINE_NO_TEXT = "results/mami_baseline_unseen_randomgeom_no_text_alpha08.csv"
DEFAULT_FINETUNED_NO_TEXT = "results/mami_finetuned_unseen_randomgeom_no_text_alpha08.csv"
DEFAULT_BASELINE_WITH_TEXT = "results/mami_baseline_unseen_randomgeom_with_text_alpha08.csv"
DEFAULT_FINETUNED_WITH_TEXT = "results/mami_finetuned_unseen_randomgeom_with_text_alpha08.csv"

LABELS = [0, 1]
CLASS_NAMES = {
    0: "clean/non-misogynous",
    1: "hate/misogynous",
}


def resolve_csv(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate

    scratch_candidate = Path("/scratch") / path
    if scratch_candidate.exists():
        return scratch_candidate

    raise FileNotFoundError(path)


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def read_predictions(path):
    with resolve_csv(path).open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["true_label"] = int(row["true_label"])
        row["pred_label"] = int(row["pred_label"])
        row["alpha"] = float(row["alpha"])

    return rows


def dedupe_by_file_name(rows):
    seen = set()
    deduped = []
    for row in rows:
        file_name = row.get("file_name")
        if file_name in seen:
            continue
        seen.add(file_name)
        deduped.append(row)
    return deduped


def f1_for_label(rows, label):
    tp = sum(1 for row in rows if row["true_label"] == label and row["pred_label"] == label)
    fp = sum(1 for row in rows if row["true_label"] != label and row["pred_label"] == label)
    fn = sum(1 for row in rows if row["true_label"] == label and row["pred_label"] != label)

    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    return safe_rate(2 * precision * recall, precision + recall)


def absolute_metrics(df):
    f1_clean = f1_for_label(df, 0)
    f1_hate = f1_for_label(df, 1)

    true_clean_n = sum(1 for row in df if row["true_label"] == 0)
    true_hate_n = sum(1 for row in df if row["true_label"] == 1)
    false_hate_n = sum(1 for row in df if row["true_label"] == 0 and row["pred_label"] == 1)
    masked_hate_n = sum(1 for row in df if row["true_label"] == 1 and row["pred_label"] == 0)
    correct_n = sum(1 for row in df if row["true_label"] == row["pred_label"])

    return {
        "n": len(df),
        "accuracy": safe_rate(correct_n, len(df)),
        "macro_f1": (f1_clean + f1_hate) / 2 if len(df) else 0.0,
        "f1_clean": f1_clean if len(df) else 0.0,
        "f1_hate": f1_hate if len(df) else 0.0,
        "true_clean_n": true_clean_n,
        "true_hate_n": true_hate_n,
        "false_hate_n": false_hate_n,
        "masked_hate_n": masked_hate_n,
        "false_hate_rate": safe_rate(false_hate_n, true_clean_n),
        "masked_hate_rate": safe_rate(masked_hate_n, true_hate_n),
    }


def print_metric_block(title, metrics):
    print(title)
    print(f"Rows                  : {metrics['n']}")
    print(f"Accuracy              : {metrics['accuracy']:.4f}")
    print(f"Macro-F1              : {metrics['macro_f1']:.4f}")
    print(f"F1 {CLASS_NAMES[0]:<20}: {metrics['f1_clean']:.4f}")
    print(f"F1 {CLASS_NAMES[1]:<20}: {metrics['f1_hate']:.4f}")
    print(
        "False hate detection  : "
        f"{metrics['false_hate_rate'] * 100:6.2f}% "
        f"({metrics['false_hate_n']}/{metrics['true_clean_n']} true clean samples)"
    )
    print(
        "Masked hate           : "
        f"{metrics['masked_hate_rate'] * 100:6.2f}% "
        f"({metrics['masked_hate_n']}/{metrics['true_hate_n']} true hate samples)"
    )


def metrics_by_group(df, group_col):
    rows = []
    groups = sorted({row[group_col] for row in df})
    for group in groups:
        if group == "none":
            continue
        subset = [row for row in df if row[group_col] == group]
        rows.append({group_col: group, **absolute_metrics(subset)})
    return rows


def print_group_table(title, grouped, group_col, sort_by="macro_f1", ascending=True, limit=None):
    if not grouped:
        print(f"\n{title}")
        print("No rows found.")
        return

    table = sorted(grouped, key=lambda row: row[sort_by], reverse=not ascending)
    if limit is not None:
        table = table[:limit]

    display_cols = [
        group_col,
        "n",
        "macro_f1",
        "f1_clean",
        "f1_hate",
        "false_hate_rate",
        "false_hate_n",
        "true_clean_n",
        "masked_hate_rate",
        "masked_hate_n",
        "true_hate_n",
    ]

    print(f"\n{title}")

    formatted_rows = []
    for row in table:
        formatted_rows.append({
            group_col: str(row[group_col]),
            "n": str(row["n"]),
            "macro_f1": f"{row['macro_f1']:.4f}",
            "f1_clean": f"{row['f1_clean']:.4f}",
            "f1_hate": f"{row['f1_hate']:.4f}",
            "false_hate_rate": f"{row['false_hate_rate'] * 100:.2f}%",
            "false_hate_n": str(row["false_hate_n"]),
            "true_clean_n": str(row["true_clean_n"]),
            "masked_hate_rate": f"{row['masked_hate_rate'] * 100:.2f}%",
            "masked_hate_n": str(row["masked_hate_n"]),
            "true_hate_n": str(row["true_hate_n"]),
        })

    widths = {
        col: max(len(col), *(len(row[col]) for row in formatted_rows))
        for col in display_cols
    }
    header = " ".join(col.ljust(widths[col]) for col in display_cols)
    print(header)
    print(" ".join("-" * widths[col] for col in display_cols))
    for row in formatted_rows:
        print(" ".join(row[col].ljust(widths[col]) for col in display_cols))


def analyze_file(file_path, model_name):
    print(f"\n{'=' * 105}")
    print(f"  ABSOLUTE UNSEEN REPORT: {model_name}")
    print(f"{'=' * 105}")

    try:
        df = read_predictions(file_path)
    except FileNotFoundError:
        print(f"[!] File not found: {file_path}")
        return

    required_cols = {"true_label", "pred_label", "alpha", "pattern", "emoji_name"}
    columns = set(df[0].keys()) if df else set()
    missing = sorted(required_cols - columns)
    if missing:
        print(f"[!] Missing required columns: {missing}")
        return

    df_clean = dedupe_by_file_name([row for row in df if row["alpha"] == 0.0])
    df_attacked = [row for row in df if row["alpha"] > 0.0]

    if df_clean:
        print_metric_block("--- 1. CLEAN IMAGES: ABSOLUTE PERFORMANCE ---", absolute_metrics(df_clean))
        print("-" * 105)

    if not df_attacked:
        print("No attacked alpha>0.0 rows found.")
        return

    print_metric_block("--- 2. ATTACKED IMAGES: ABSOLUTE PERFORMANCE ---", absolute_metrics(df_attacked))
    print("-" * 105)

    by_pattern = metrics_by_group(df_attacked, "pattern")
    print_group_table(
        "--- 3. UNSEEN PATTERN ABSOLUTE METRICS (worst Macro-F1 first) ---",
        by_pattern,
        "pattern",
    )

    by_emoji = metrics_by_group(df_attacked, "emoji_name")
    print_group_table(
        "--- 4. TOP 5 SEMANTIC DISTRACTORS (worst absolute Macro-F1) ---",
        by_emoji,
        "emoji_name",
        limit=5,
    )


def compare_pair(baseline_csv, finetuned_csv, label):
    print(f"\n{'#' * 105}")
    print(f"# {label}")
    print(f"{'#' * 105}")
    analyze_file(baseline_csv, "BASELINE MODEL")
    analyze_file(finetuned_csv, "FINE-TUNED MODEL")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report absolute MAMI unseen-geometry metrics: per-class F1, "
            "false hate detection, and masked hate rates. No clean-vs-attacked "
            "flip rates are computed."
        )
    )
    parser.add_argument("--baseline-no-text", default=DEFAULT_BASELINE_NO_TEXT)
    parser.add_argument("--finetuned-no-text", default=DEFAULT_FINETUNED_NO_TEXT)
    parser.add_argument("--baseline-with-text", default=DEFAULT_BASELINE_WITH_TEXT)
    parser.add_argument("--finetuned-with-text", default=DEFAULT_FINETUNED_WITH_TEXT)
    parser.add_argument("--mode", choices=["no_text", "with_text", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ["no_text", "both"]:
        compare_pair(args.baseline_no_text, args.finetuned_no_text, "NO TEXT")

    if args.mode in ["with_text", "both"]:
        compare_pair(args.baseline_with_text, args.finetuned_with_text, "WITH TEXT")


if __name__ == "__main__":
    main()
