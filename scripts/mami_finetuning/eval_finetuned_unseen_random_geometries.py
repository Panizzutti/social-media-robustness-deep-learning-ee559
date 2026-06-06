import os
import math
import csv
import argparse
import hashlib
import random

import torch
import torch.distributed as dist
import pandas as pd
from PIL import Image
import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# ==========================================
# Configuration
# ==========================================
TOP_20_EMOJIS = {
    "1f480": "skull", "1f600": "grinning", "1f602": "tears_of_joy", "1f609": "winking",
    "1f60a": "smiling_blush", "1f60e": "sunglasses", "1f620": "angry", "1f525": "fire",
    "1f921": "clown", "1f633": "flushed", "2764": "red_heart", "1f44d": "thumbs_up",
    "1f62d": "loudly_crying", "1f64f": "folded_hands", "1f914": "thinking", "1f4a9": "poop",
    "2728": "sparkles", "1f4af": "hundred", "1f644": "rolling_eyes", "1f389": "party_popper"
}

BASE_MODEL_ID = "QCRI/MemeLens-VLM"
ADAPTER_PATH = "/scratch/models/MemeLens_Robust_LoRA/final"

BASE_IMG_DIR = "/scratch/datasets/MAMI/test/"
LABELS_PATH = "/scratch/datasets/MAMI/test_labels.txt"
TEXTS_PATH = "/scratch/datasets/MAMI/test/Test.csv"
EMOJI_BASE_DIR = "/scratch/utils/emoticons/emoticons_png/512"
RESULTS_DIR = "/scratch/results"

FIXED_ALPHA = 0.8
GLOBAL_SEED = 42

UNSEEN_PATTERN_NAMES = [
    "single_center",
    "border_frame",
    "random_sparse",
    "random_dense",
    "random_scale_rotation",
]

SYSTEM_PROMPT = (
    "You are an automated classification API. You must output EXACTLY ONE WORD. "
    "Do not output any full sentences, explanations, or prefixes."
)

# ==========================================
# Deterministic randomness helpers
# ==========================================
def stable_seed(*parts) -> int:
    """Create a reproducible integer seed from strings/values."""
    text = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def deterministic_rng(*parts) -> random.Random:
    return random.Random(stable_seed(*parts))

# ==========================================
# Image helper functions
# ==========================================
def apply_alpha(emoji_img: Image.Image, alpha: float) -> Image.Image:
    emoji = emoji_img.convert("RGBA")
    r, g, b, a = emoji.split()
    a = a.point(lambda p: int(p * alpha))
    return Image.merge("RGBA", (r, g, b, a))


def resize_with_alpha(emoji_img: Image.Image, size: int, alpha: float) -> Image.Image:
    emoji = emoji_img.convert("RGBA").resize((size, size), resample=Image.Resampling.LANCZOS)
    return apply_alpha(emoji, alpha)


def paste_centered(overlay: Image.Image, emoji: Image.Image, cx: float, cy: float) -> None:
    x = int(round(cx - emoji.size[0] / 2))
    y = int(round(cy - emoji.size[1] / 2))
    overlay.paste(emoji, (x, y), emoji)


def random_center(rng: random.Random, w: int, h: int, size: int):
    # Keep the emoji center inside the image. If the image is tiny, fall back to center.
    if w <= size or h <= size:
        return w / 2, h / 2
    cx = rng.uniform(size / 2, w - size / 2)
    cy = rng.uniform(size / 2, h - size / 2)
    return cx, cy

# ==========================================
# Unseen geometry functions
# Each function accepts (base_img, emoji_img, alpha, rng)
# ==========================================
def apply_single_center(base_img: Image.Image, emoji_img: Image.Image, alpha: float, rng: random.Random) -> Image.Image:
    base = base_img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    size = max(1, int(min(w, h) * 0.35))
    emoji = resize_with_alpha(emoji_img, size, alpha)
    paste_centered(overlay, emoji, w / 2, h / 2)
    return Image.alpha_composite(base, overlay).convert("RGB")


def apply_border_frame(base_img: Image.Image, emoji_img: Image.Image, alpha: float, rng: random.Random) -> Image.Image:
    base = base_img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    size = max(1, int(min(w, h) * 0.14))
    emoji = resize_with_alpha(emoji_img, size, alpha)

    margin = size * 0.6
    top_y = margin
    bottom_y = h - margin
    left_x = margin
    right_x = w - margin

    # 4 along top, 4 along bottom, 2 along left side, 2 along right side.
    top_bottom_xs = [w * (i + 0.5) / 4 for i in range(4)]
    side_ys = [h * (i + 1) / 3 for i in range(2)]

    for cx in top_bottom_xs:
        paste_centered(overlay, emoji, cx, top_y)
        paste_centered(overlay, emoji, cx, bottom_y)
    for cy in side_ys:
        paste_centered(overlay, emoji, left_x, cy)
        paste_centered(overlay, emoji, right_x, cy)

    return Image.alpha_composite(base, overlay).convert("RGB")


def apply_random_sparse(base_img: Image.Image, emoji_img: Image.Image, alpha: float, rng: random.Random) -> Image.Image:
    base = base_img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    size = max(1, int(min(w, h) * 0.18))
    emoji = resize_with_alpha(emoji_img, size, alpha)

    for _ in range(5):
        cx, cy = random_center(rng, w, h, size)
        paste_centered(overlay, emoji, cx, cy)

    return Image.alpha_composite(base, overlay).convert("RGB")


def apply_random_dense(base_img: Image.Image, emoji_img: Image.Image, alpha: float, rng: random.Random) -> Image.Image:
    base = base_img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    size = max(1, int(min(w, h) * 0.22))
    emoji = resize_with_alpha(emoji_img, size, alpha)

    # Same emoji count as the old 4x4 grid, but with unseen random geometry.
    for _ in range(16):
        cx, cy = random_center(rng, w, h, size)
        paste_centered(overlay, emoji, cx, cy)

    return Image.alpha_composite(base, overlay).convert("RGB")


def apply_random_scale_rotation(base_img: Image.Image, emoji_img: Image.Image, alpha: float, rng: random.Random) -> Image.Image:
    base = base_img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    min_side = min(w, h)

    for _ in range(8):
        scale = rng.uniform(0.10, 0.30)
        size = max(1, int(min_side * scale))
        emoji = resize_with_alpha(emoji_img, size, alpha)
        angle = rng.uniform(-30.0, 30.0)
        emoji = emoji.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        cx, cy = random_center(rng, w, h, max(emoji.size))
        paste_centered(overlay, emoji, cx, cy)

    return Image.alpha_composite(base, overlay).convert("RGB")


PATTERN_FUNCS = {
    "single_center": apply_single_center,
    "border_frame": apply_border_frame,
    "random_sparse": apply_random_sparse,
    "random_dense": apply_random_dense,
    "random_scale_rotation": apply_random_scale_rotation,
}

# ==========================================
# Data and task construction
# ==========================================
def load_available_emojis():
    available = {}
    for hex_code, name in TOP_20_EMOJIS.items():
        candidates = [
            os.path.join(EMOJI_BASE_DIR, f"emoji_u{hex_code}.png"),
            os.path.join(EMOJI_BASE_DIR, f"emoji_u{hex_code.upper()}.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                available[name] = path
                break
    return available


def resolve_image_path(file_name: str) -> str:
    img_path = os.path.join(BASE_IMG_DIR, file_name)
    if os.path.exists(img_path):
        return img_path
    png_path = img_path.replace(".jpg", ".png")
    if os.path.exists(png_path):
        return png_path
    return img_path


def build_tasks(mami_df: pd.DataFrame, available_emojis: dict):
    tasks = []

    # One clean baseline per image.
    for _, row in mami_df.iterrows():
        tasks.append({
            "file_name": str(row["file_name"]),
            "true_label": int(row["misogynous"]),
            "text": str(row["Text Transcription"]) if pd.notna(row.get("Text Transcription", "")) else "",
            "emoji_name": "none",
            "emoji_path": "",
            "pattern_name": "none",
            "alpha": 0.0,
            "geometry_seed": -1,
            "placement_seed": -1,
        })

    # For each image and each emoji, assign exactly one unseen geometry deterministically.
    for _, row in mami_df.iterrows():
        file_name = str(row["file_name"])
        true_label = int(row["misogynous"])
        text = str(row["Text Transcription"]) if pd.notna(row.get("Text Transcription", "")) else ""

        for emoji_name, emoji_path in available_emojis.items():
            geom_seed = stable_seed(GLOBAL_SEED, file_name, emoji_name, "geometry")
            geom_rng = random.Random(geom_seed)
            pattern_name = geom_rng.choice(UNSEEN_PATTERN_NAMES)
            placement_seed = stable_seed(GLOBAL_SEED, file_name, emoji_name, pattern_name, "placement")

            tasks.append({
                "file_name": file_name,
                "true_label": true_label,
                "text": text,
                "emoji_name": emoji_name,
                "emoji_path": emoji_path,
                "pattern_name": pattern_name,
                "alpha": FIXED_ALPHA,
                "geometry_seed": geom_seed,
                "placement_seed": placement_seed,
            })

    return tasks


def make_instruction(task_text: str, include_text: bool) -> str:
    base_task = (
        "Task: Identify if this meme contains misogyny.\n"
        "Allowed Labels: 'misogynous' or 'clean'.\n"
        "Answer:"
    )
    if include_text:
        return f"Meme Text: '{task_text}'\n\n{base_task}"
    return base_task

# ==========================================
# Main distributed evaluation
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate unseen random emoji geometries at fixed alpha=0.8.")
    parser.add_argument("--mode", choices=["no_text", "with_text"], default="no_text",
                        help="Whether to inject OCR text transcription into the prompt.")
    parser.add_argument("--output_csv", default=None,
                        help="Optional output CSV path. If omitted, a default path is used.")
    args = parser.parse_args()

    include_text = args.mode == "with_text"
    default_name = (
        "mami_finetuned_unseen_randomgeom_with_text_alpha08.csv"
        if include_text else
        "mami_finetuned_unseen_randomgeom_no_text_alpha08.csv"
    )
    out_csv = args.output_csv or os.path.join(RESULTS_DIR, default_name)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    temp_csv = os.path.join(RESULTS_DIR, f"temp_unseen_randomgeom_{args.mode}_rank_{local_rank}.csv")

    with open(temp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "true_label", "emoji_name", "pattern", "alpha",
            "pred_label", "output_text", "mode", "geometry_seed", "placement_seed"
        ])

    if local_rank == 0:
        print("Loading MAMI evaluation metadata and available emojis...")

    texts_df = pd.read_csv(TEXTS_PATH, sep="\t")
    labels_df = pd.read_csv(
        LABELS_PATH,
        sep="\t",
        header=None,
        names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"],
    )
    mami_df = pd.merge(texts_df, labels_df, on="file_name", how="inner")

    available_emojis = load_available_emojis()
    if len(available_emojis) == 0:
        raise FileNotFoundError(f"No emoji PNG files found in {EMOJI_BASE_DIR}")

    all_tasks = build_tasks(mami_df, available_emojis)
    my_tasks = all_tasks[local_rank::world_size]

    if local_rank == 0:
        attacked = [t for t in all_tasks if t["alpha"] > 0.0]
        counts = pd.Series([t["pattern_name"] for t in attacked]).value_counts().sort_index()
        print(f"Mode: {args.mode}")
        print(f"Fixed alpha for attacked images: {FIXED_ALPHA}")
        print(f"Available emojis: {len(available_emojis)}")
        print(f"Total tasks: {len(all_tasks)} ({len(mami_df)} clean + {len(attacked)} attacked)")
        print("Randomly assigned unseen geometry counts:")
        print(counts.to_string())
        print(f"Each GPU is processing about {len(my_tasks)} tasks.")

    # Loading barrier to avoid concurrent cache/download issues.
    if local_rank != 0:
        dist.barrier()

    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        device_map={"": local_rank},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    if local_rank == 0:
        dist.barrier()

    # V100 compatibility: convert bfloat16 leftovers to float16.
    base_model.config.torch_dtype = torch.float16
    for _, param in base_model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for _, buffer in base_model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)

    if local_rank == 0:
        print("Attaching fine-tuned LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    pbar = tqdm.tqdm(total=len(my_tasks), disable=(local_rank != 0), desc=f"Inferencing {args.mode}")

    for task in my_tasks:
        img_path = resolve_image_path(task["file_name"])
        output_text = ""

        try:
            if task["alpha"] == 0.0:
                obfuscated_pil = Image.open(img_path).convert("RGB")
            else:
                base_img = Image.open(img_path).convert("RGBA")
                emoji_img = Image.open(task["emoji_path"]).convert("RGBA")
                placement_rng = random.Random(task["placement_seed"])
                pattern_func = PATTERN_FUNCS[task["pattern_name"]]
                obfuscated_pil = pattern_func(base_img, emoji_img, task["alpha"], placement_rng)

            dynamic_instruction = make_instruction(task["text"], include_text=include_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": obfuscated_pil},
                    {"type": "text", "text": dynamic_instruction},
                ]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(f"cuda:{local_rank}")

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
            pred_label = 1 if "misogyn" in output_text else 0

        except Exception as e:
            if local_rank == 0:
                print(f"Error on {task['file_name']} / {task['emoji_name']} / {task['pattern_name']}: {e}")
            pred_label = 0
            output_text = f"ERROR: {type(e).__name__}"

        with open(temp_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                task["file_name"],
                task["true_label"],
                task["emoji_name"],
                task["pattern_name"],
                task["alpha"],
                pred_label,
                output_text,
                args.mode,
                task["geometry_seed"],
                task["placement_seed"],
            ])

        pbar.update(1)

    dist.barrier()

    if local_rank == 0:
        print("All GPUs finished. Merging temporary files...")
        dfs = []
        for rank in range(world_size):
            rank_csv = os.path.join(RESULTS_DIR, f"temp_unseen_randomgeom_{args.mode}_rank_{rank}.csv")
            dfs.append(pd.read_csv(rank_csv))
            os.remove(rank_csv)

        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(out_csv, index=False)
        print(f"Saved results to: {out_csv}")
        print("Final pattern counts for attacked rows:")
        print(final_df[final_df["alpha"] > 0.0]["pattern"].value_counts().sort_index().to_string())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
