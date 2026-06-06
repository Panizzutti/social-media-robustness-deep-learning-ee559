import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import tqdm
import os
import argparse

# 1. On-The-Fly Obfuscation Function
def apply_emoji_grid_on_the_fly(base_img_path, emoji_path, alpha):
    """Dynamically pastes a 4x4 emoji grid onto an image with a specific alpha."""
    base_img = Image.open(base_img_path).convert("RGBA")
    w, h = base_img.size
    
    # If alpha is 0, just return the clean image
    if alpha == 0.0:
        return base_img.convert("RGB")
        
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    emoji = Image.open(emoji_path).convert("RGBA")
    
    # Size emoji relative to the image
    min_side = min(w, h)
    e_size = int(min_side * 0.25) # 4x4 grid means roughly 25% of the screen
    emoji = emoji.resize((e_size, e_size), resample=Image.Resampling.LANCZOS)
    
    # Apply alpha transparency to the emoji
    r, g, b, a = emoji.split()
    a = a.point(lambda p: int(p * alpha))
    emoji = Image.merge("RGBA", (r, g, b, a))
    
    # Paste in a 4x4 grid pattern
    for i in range(4):
        for j in range(4):
            x = int((i + 0.5) * (w / 4) - e_size / 2)
            y = int((j + 0.5) * (h / 4) - e_size / 2)
            overlay.paste(emoji, (x, y), emoji)
            
    # Blend and convert back to RGB for the model
    return Image.alpha_composite(base_img, overlay).convert("RGB")


def main():
    # Setup Model
    model_id = "QCRI/MemeLens-VLM"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    ).eval()

    # Paths
    base_img_dir = "/scratch/datasets/MAMI/test/"
    labels_path = "/scratch/datasets/MAMI/test_labels.txt"
    # Make sure this points to your exact laughing emoji!
    emoji_path = "/scratch/utils/emoticons/emoticons_png/512/emoji_u1f602.png" 
    out_dir = "/scratch/results/alpha_sweep"
    os.makedirs(out_dir, exist_ok=True)

    mami_df = pd.read_csv(labels_path, sep="\t", header=None, 
                          names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])

    # Prompts
    system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."
    instruction_prompt = "Task: Identify if this meme contains misogyny.\nAllowed Labels: 'misogynous' or 'clean'.\nAnswer:"

    # The Alphas we want to test
    alphas_to_test = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    for alpha in alphas_to_test:
        print(f"\nStarting sweep for Alpha = {alpha}...")
        results = []
        
        for index, row in tqdm.tqdm(mami_df.iterrows(), total=len(mami_df)):
            orig_filename = str(row['file_name'])
            img_path = os.path.join(base_img_dir, orig_filename)
            true_label = int(row['misogynous'])
            
            try:
                # 1. Apply Obfuscation ON THE FLY
                obfuscated_pil = apply_emoji_grid_on_the_fly(img_path, emoji_path, alpha)
                
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
                pred_label = 0
                
            results.append({"file_name": orig_filename, "true_label": true_label, "pred_label": pred_label})
            
        # Save CSV for this specific alpha
        out_csv = os.path.join(out_dir, f"predictions_alpha_{int(alpha*100)}.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved Alpha {alpha} to {out_csv}")

if __name__ == "__main__":
    main()