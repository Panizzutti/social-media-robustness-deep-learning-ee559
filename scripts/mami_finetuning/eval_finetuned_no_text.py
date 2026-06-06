import os
import torch
import torch.distributed as dist
import pandas as pd
from PIL import Image
import math
import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel 

# ==========================================
# 1. Configuration & Emojis
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
# [CHANGE 1]: Distinct Output File
OUT_CSV = "/scratch/results/mami_finetuned_massive_no_text_results.csv"

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
    w, h = base_img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    min_side = min(w, h)
    
    e_size = int(min_side * 0.15) 
    emoji = emoji_img.resize((e_size, e_size), resample=Image.Resampling.LANCZOS)
    
    r, g, b, a = emoji.split()
    a = a.point(lambda p: int(p * alpha))
    emoji = Image.merge("RGBA", (r, g, b, a))
    
    cx, cy = w / 2, h / 2
    radius = min_side * 0.35 
    num_emojis = 12
    
    for i in range(num_emojis):
        theta = 2 * math.pi * i / num_emojis
        x = int(cx + radius * math.cos(theta) - e_size / 2)
        y = int(cy + radius * math.sin(theta) - e_size / 2)
        overlay.paste(emoji, (x, y), emoji)
        
    return Image.alpha_composite(base_img, overlay).convert("RGB")

# ==========================================
# 3. Main Multi-GPU Execution Loop
# ==========================================
def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    base_img_dir = "/scratch/datasets/MAMI/test/"
    labels_path = "/scratch/datasets/MAMI/test_labels.txt"
    texts_path = "/scratch/datasets/MAMI/test/Test.csv"
    emoji_base_dir = "/scratch/utils/emoticons/emoticons_png/512"
    
    temp_csv = f"/scratch/results/temp_no_text_rank_{local_rank}_results.csv"
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(temp_csv, "w") as f:
        f.write("file_name,true_label,emoji_name,pattern,alpha,pred_label\n")

    if local_rank == 0: print("Loading datasets and verifying emojis...")
    texts_df = pd.read_csv(texts_path, sep="\t") 
    labels_df = pd.read_csv(labels_path, sep="\t", header=None, 
                            names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])
    mami_df = pd.merge(texts_df, labels_df, on="file_name", how="inner")

    alphas_to_test = [0.0, 0.5, 0.8, 1.0] 
    patterns = {"grid_4x4": apply_4x4_grid, "circle_12": apply_circle_pattern}

    available_emojis = {}
    for hex_code, name in TOP_20_EMOJIS.items():
        for p in [os.path.join(emoji_base_dir, f"emoji_u{hex_code}.png"), os.path.join(emoji_base_dir, f"emoji_u{hex_code.upper()}.png")]:
            if os.path.exists(p):
                available_emojis[name] = p
                break

    all_tasks = []
    first_emoji = list(available_emojis.keys())[0]
    
    for index, row in mami_df.iterrows():
        for emoji_name, emoji_path in available_emojis.items():
            for pattern_name, pattern_func in patterns.items():
                for alpha in alphas_to_test:
                    if alpha == 0.0 and (emoji_name != first_emoji or pattern_name != "grid_4x4"):
                        continue
                    
                    all_tasks.append({
                        "file_name": str(row['file_name']),
                        "true_label": int(row['misogynous']),
                        "emoji_name": emoji_name,
                        "emoji_path": emoji_path,
                        "pattern_name": pattern_name,
                        "pattern_func": pattern_func,
                        "alpha": alpha
                    })

    my_tasks = all_tasks[local_rank::world_size]
    if local_rank == 0: 
        print(f"Total No-Text Tasks: {len(all_tasks)}. Each GPU is processing ~{len(my_tasks)} tasks.")

    if local_rank != 0: dist.barrier()
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, device_map={"": local_rank}, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    if local_rank == 0: dist.barrier()

    base_model.config.torch_dtype = torch.float16
    for name, param in base_model.named_parameters():
        if param.dtype == torch.bfloat16: param.data = param.data.to(torch.float16)

    if local_rank == 0: print("Attaching Fine-Tuned LoRA Adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."

    pbar = tqdm.tqdm(total=len(my_tasks), disable=(local_rank != 0), desc="Inferencing (No Text)")

    for task in my_tasks:
        img_path = os.path.join(base_img_dir, task["file_name"])
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            
        try:
            if task["alpha"] == 0.0:
                obfuscated_pil = Image.open(img_path).convert("RGB")
            else:
                emoji_img = Image.open(task["emoji_path"]).convert("RGBA")
                obfuscated_pil = task["pattern_func"](Image.open(img_path).convert("RGBA"), emoji_img, task["alpha"])
            
            # [CHANGE 2]: Pure Visual Prompt (No OCR Text Injected)
            dynamic_instruction = "Task: Identify if this meme contains misogyny.\nAllowed Labels: 'misogynous' or 'clean'.\nAnswer:"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": obfuscated_pil},
                    {"type": "text", "text": dynamic_instruction}
                ]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                               padding=True, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
            
            pred_label = 1 if "misogyn" in output_text else 0
            
        except Exception as e:
            if local_rank == 0: print(f"Error on {task['file_name']}: {e}")
            pred_label = 0 
            
        out_e = 'none' if task["alpha"] == 0.0 else task["emoji_name"]
        out_p = 'none' if task["alpha"] == 0.0 else task["pattern_name"]
        
        with open(temp_csv, "a") as f:
            f.write(f"{task['file_name']},{task['true_label']},{out_e},{out_p},{task['alpha']},{pred_label}\n")
        
        pbar.update(1)

    dist.barrier()
    
    if local_rank == 0:
        print("All GPUs finished. Merging temporary files into final No-Text CSV...")
        all_dfs = []
        for i in range(world_size):
            rank_csv = f"/scratch/results/temp_no_text_rank_{i}_results.csv"
            all_dfs.append(pd.read_csv(rank_csv))
            os.remove(rank_csv) 
            
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(OUT_CSV, index=False)
        print(f"✅ Master evaluation complete! Results saved to: {OUT_CSV}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()