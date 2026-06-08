import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report
import tqdm
import os

# 1. Setup Model
model_id = "QCRI/MemeLens-VLM"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, 
    dtype=torch.float16, 
    device_map="auto"
)
model.eval()

# 2. Load Dataset
labels_path = "/scratch/datasets/MAMI/test_labels.txt"
base_img_dir = "/scratch/datasets/MAMI/test/"
mami_df = pd.read_csv(labels_path, sep="\t", header=None, names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])

# 3. Prompting
system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."
instruction_prompt = "Task: Identify if this meme contains misogyny.\nAllowed Labels: 'misogynous' or 'clean'.\nAnswer:"

def predict_meme(image_path, debug=False):
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
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
    
    if debug:
        print(f"\n--- DEBUG {image_path} ---")
        print(f"RAW MODEL OUTPUT: '{output_text}'")
        print("---------------------------")
    
    if "clean" in output_text or "not" in output_text or "non" in output_text:
        return 0
    elif "misogyn" in output_text:
        return 1
    else:
        return 0

# 4. Run Evaluation
y_true = []
y_pred = []
results = []

for index, row in tqdm.tqdm(mami_df.iterrows(), total=len(mami_df)):
    orig_filename = str(row['file_name'])
    img_path = os.path.join(base_img_dir, orig_filename)
    
    debug_mode = True if index < 5 else False 
    
    try:
        pred_label = predict_meme(img_path, debug=debug_mode)
    except Exception as e:
        pred_label = 0
        
    true_label = int(row['misogynous'])
    
    y_true.append(true_label)
    y_pred.append(pred_label)
    
    results.append({
        "file_name": orig_filename,
        "true_label": true_label,
        "pred_label": pred_label
    })

# 5. Metrics & Saving
print(classification_report(y_true, y_pred, target_names=["non-misogynous", "misogynous"]))

out_csv = "/scratch/results/mami_clean_predictions.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
out_df = pd.DataFrame(results)
out_df.to_csv(out_csv, index=False)

print(f"\nSaved clean predictions to: {out_csv}")
