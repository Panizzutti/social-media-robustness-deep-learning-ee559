import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import tqdm
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Path to images (clean or obfuscated)")
    parser.add_argument("--out_csv", type=str, required=True, help="Where to save predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Model
    model_id = "QCRI/MemeLens-VLM"
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    ).eval()

    # 2. Load Datasets and Merge
    texts_path = "/scratch/datasets/MAMI/test/Test.csv"
    labels_path = "/scratch/datasets/MAMI/test_labels.txt"
    
    print(f"Loading text transcriptions from {texts_path}...")
    texts_df = pd.read_csv(texts_path, sep="\t") 
    
    print(f"Loading ground truth labels from {labels_path}...")
    # Read labels with explicit column names since there is no header
    labels_df = pd.read_csv(labels_path, sep="\t", header=None, 
                            names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])
    
    # Merge on file_name so we have text and true labels in one row
    print("Merging datasets...")
    mami_df = pd.merge(texts_df, labels_df, on="file_name", how="inner")
    print(f"Total merged samples to process: {len(mami_df)}")

    # 3. Prompting
    system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."

    def predict_meme(image_path, text_transcription):
        image = Image.open(image_path).convert("RGB")
        
        # INJECT THE TEXT TRANSCRIPTION HERE
        instruction_prompt = f"Meme Text: '{text_transcription}'\n\nTask: Identify if this meme contains misogyny.\nAllowed Labels: 'misogynous' or 'clean'.\nAnswer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction_prompt}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                           padding=True, return_tensors="pt").to("cuda")
        
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
    
    print("Starting inference loop...")
    for index, row in tqdm.tqdm(mami_df.iterrows(), total=len(mami_df)):
        orig_filename = str(row['file_name'])
        
        # Resolve either jpg or png image paths.
        img_path = os.path.join(args.img_dir, orig_filename)
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            
        # Extract variables from merged row
        true_label = int(row['misogynous'])
        text_transcription = str(row['Text Transcription']) if pd.notna(row['Text Transcription']) else ""
        
        try:
            pred_label = predict_meme(img_path, text_transcription)
        except Exception:
            # If the image is completely missing or corrupted, default to 0
            pred_label = 0
            
        results.append({
            "file_name": orig_filename,
            "true_label": true_label,
            "pred_label": pred_label
        })

    # 5. Save
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"Saved predictions to: {args.out_csv}")

if __name__ == "__main__":
    main()
