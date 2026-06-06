import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import tqdm
import os
import math


# ==========================================
# 1. The Top 20 Emoji Dictionary Mapping
# ==========================================
TOP_20_EMOJIS = {
    "1f480": "skull",
    "1f600": "grinning",
    "1f602": "tears_of_joy",
    "1f609": "winking",
    "1f60a": "smiling_blush",
    "1f60e": "sunglasses",
    "1f620": "angry",
    "1f525": "fire",
    "1f921": "clown",
    "1f633": "flushed", 

    # 10 Additional High-Usage / Distinct Shapes
    "2764": "red_heart",
    "1f44d": "thumbs_up",
    "1f62d": "loudly_crying",
    "1f64f": "folded_hands",
    "1f914": "thinking",
    "1f4a9": "poop",
    "2728": "sparkles",
    "1f4af": "hundred",
    "1f644": "rolling_eyes",
    "1f389": "party_popper"
}

# ==========================================
# 2. Pattern Generators
# ==========================================
def apply_4x4_grid(base_img, emoji_img, alpha):
    w, h = base_img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    min_side = min(w, h)
    e_size = int(min_side * 0.25)
    emoji = emoji_img.resize((e_size, e_size), resample=Image.Resampling.LANCZOS)
    
    r, g, b, a = emoji.split()
    a = a.point(lambda p: int(p * alpha))
    emoji = Image.merge("RGBA", (r, g, b, a))
    
    for i in range(4):
        for j in range(4):
            x = int((i + 0.5) * (w / 4) - e_size / 2)
            y = int((j + 0.5) * (h / 4) - e_size / 2)
            overlay.paste(emoji, (x, y), emoji)
            
    return Image.alpha_composite(base_img, overlay).convert("RGB")

def apply_circle_pattern(base_img, emoji_img, alpha):
    """Pastes 12 smaller emojis in a circle, leaving center and edges free."""
    w, h = base_img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    min_side = min(w, h)
    
    e_size = int(min_side * 0.15) # Smaller emojis for the circle
    emoji = emoji_img.resize((e_size, e_size), resample=Image.Resampling.LANCZOS)
    
    r, g, b, a = emoji.split()
    a = a.point(lambda p: int(p * alpha))
    emoji = Image.merge("RGBA", (r, g, b, a))
    
    cx, cy = w / 2, h / 2
    radius = min_side * 0.35 # 35% out from the center
    num_emojis = 12
    
    for i in range(num_emojis):
        theta = 2 * math.pi * i / num_emojis
        x = int(cx + radius * math.cos(theta) - e_size / 2)
        y = int(cy + radius * math.sin(theta) - e_size / 2)
        overlay.paste(emoji, (x, y), emoji)
        
    return Image.alpha_composite(base_img, overlay).convert("RGB")

# ==========================================
# 3. Main Execution Loop
# ==========================================
def main():
    # Setup Paths
    base_img_dir = "/scratch/datasets/MAMI/test/"
    labels_path = "/scratch/datasets/MAMI/test_labels.txt"
    emoji_base_dir = "/scratch/utils/emoticons/emoticons_png/512"
    out_csv = "/scratch/results/mami_massive_sweep_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Load Dataset
    mami_df = pd.read_csv(labels_path, sep="\t", header=None, 
                          names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])

    # Load previously completed rows to allow safe resuming
    completed_keys = set()
    if os.path.exists(out_csv):
        try:
            existing_df = pd.read_csv(out_csv)
            for _, row in existing_df.iterrows():
                # Unique identifier for a completed run
                key = f"{row['file_name']}_{row['emoji_name']}_{row['pattern']}_{row['alpha']}"
                completed_keys.add(key)
            print(f"Resuming job: Found {len(completed_keys)} previously completed inferences.")
        except Exception as e:
            print(f"Starting fresh CSV. (Could not read existing: {e})")
            with open(out_csv, "w") as f:
                f.write("file_name,true_label,emoji_name,pattern,alpha,pred_label\n")
    else:
        with open(out_csv, "w") as f:
            f.write("file_name,true_label,emoji_name,pattern,alpha,pred_label\n")

    # Load Model
    print("Loading MemeLens-VLM...")
    model_id = "QCRI/MemeLens-VLM"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    ).eval()

    system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."
    instruction_prompt = "Task: Identify if this meme contains misogyny.\nAllowed Labels: 'misogynous' or 'clean'.\nAnswer:"

    # Sweep Parameters
    alphas_to_test = [0.0, 0.5, 0.8, 1.0] # 0.0 is the clean baseline
    patterns = {"grid_4x4": apply_4x4_grid, "circle_12": apply_circle_pattern}

    # Build valid emoji paths
    available_emojis = {}
    for hex_code, name in TOP_20_EMOJIS.items():
        # Handle cases where your files might be uppercase or lowercase
        path1 = os.path.join(emoji_base_dir, f"emoji_u{hex_code}.png")
        path2 = os.path.join(emoji_base_dir, f"emoji_u{hex_code.upper()}.png")
        if os.path.exists(path1):
            available_emojis[name] = path1
        elif os.path.exists(path2):
            available_emojis[name] = path2
            
    print(f"Found {len(available_emojis)} out of 20 requested emojis on disk.")

    total_inferences = len(mami_df) * len(available_emojis) * len(patterns) * len(alphas_to_test)
    print(f"Total target inferences: {total_inferences}")

    pbar = tqdm.tqdm(total=total_inferences)
    pbar.update(len(completed_keys)) # Advance progress bar by what's already done

    # The Massive Loop
    for emoji_name, emoji_path in available_emojis.items():
        emoji_img = Image.open(emoji_path).convert("RGBA")
        
        for pattern_name, pattern_func in patterns.items():
            for alpha in alphas_to_test:
                
                # To optimize, we skip alpha 0.0 for multiple emojis/patterns since it's just a clean image.
                # We only need to compute alpha 0.0 once per image.
                is_clean_run = (alpha == 0.0)
                if is_clean_run and (emoji_name != list(available_emojis.keys())[0] or pattern_name != "grid_4x4"):
                    pbar.update(len(mami_df))
                    continue # Skip redundant clean runs

                for index, row in mami_df.iterrows():
                    orig_filename = str(row['file_name'])
                    true_label = int(row['misogynous'])
                    
                    # Check if already done (Resume logic)
                    key = f"{orig_filename}_{'none' if is_clean_run else emoji_name}_{'none' if is_clean_run else pattern_name}_{alpha}"
                    if key in completed_keys:
                        pbar.update(1)
                        continue

                    img_path = os.path.join(base_img_dir, orig_filename)
                    
                    try:
                        # 1. Obfuscate
                        if is_clean_run:
                            obfuscated_pil = Image.open(img_path).convert("RGB")
                        else:
                            obfuscated_pil = pattern_func(Image.open(img_path).convert("RGBA"), emoji_img, alpha)
                        
                        # 2. Setup Prompt
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "image", "image": obfuscated_pil},
                                {"type": "text", "text": instruction_prompt}
                            ]}
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        
                        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                                           padding=True, return_tensors="pt").to("cuda")
                        
                        # 3. Predict
                        with torch.no_grad():
                            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                        
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
                        
                        pred_label = 1 if "misogyn" in output_text else 0
                        
                    except Exception as e:
                        print(f"Error on {orig_filename}: {e}")
                        pred_label = 0 # Fallback
                        
                    # 4. Save to CSV immediately
                    out_emoji = 'none' if is_clean_run else emoji_name
                    out_pattern = 'none' if is_clean_run else pattern_name
                    
                    with open(out_csv, "a") as f:
                        f.write(f"{orig_filename},{true_label},{out_emoji},{out_pattern},{alpha},{pred_label}\n")
                    
                    completed_keys.add(key)
                    pbar.update(1)

if __name__ == "__main__":
    main()