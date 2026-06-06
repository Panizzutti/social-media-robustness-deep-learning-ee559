import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tqdm
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MemeLens on augmented MAMI datasets.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the augmented images folder.")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to save the prediction CSV.")
    parser.add_argument("--labels", type=str, default="/scratch/datasets/MAMI/test_labels.txt")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Model
    model_id = "QCRI/MemeLens-VLM"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # 2. Load Dataset
    print(f"Loading labels from {args.labels}")
    mami_df = pd.read_csv(
        args.labels, sep="\t", header=None, 
        names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"]
    )

    # 3. Prompting
    system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."
    instruction_prompt = "Task: Identify if this meme contains misogyny.\nAllowed Labels: 'misogynous' or 'clean'.\nAnswer:"

    def predict_meme(image_path):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction_prompt}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, 
            padding=True, return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
        
        if "clean" in output_text or "not" in output_text or "non" in output_text:
            return 0
        elif "misogyn" in output_text:
            return 1
        return 0 

    # 4. Run Evaluation
    results = []
    base_img_dir = Path(args.img_dir)
    
    print(f"Evaluating images from: {base_img_dir}")
    for index, row in tqdm.tqdm(mami_df.iterrows(), total=len(mami_df)):
        orig_filename = str(row['file_name'])
        
        # Smart resolver: The augmentation script might have changed .jpg to .png
        img_path = base_img_dir / orig_filename
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
            
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}. Defaulting to clean (0).")
            pred_label = 0
        else:
            try:
                pred_label = predict_meme(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                pred_label = 0
                
        true_label = int(row['misogynous'])
        results.append({
            "file_name": orig_filename,
            "true_label": true_label,
            "pred_label": pred_label
        })

    # 5. Save and Print Metrics
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    
    acc = accuracy_score(out_df['true_label'], out_df['pred_label'])
    macro_f1 = f1_score(out_df['true_label'], out_df['pred_label'], average='macro')
    
    print("\n==========================================")
    print(f"Results for: {args.img_dir}")
    print(f"Saved to: {args.out_csv}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()