import torch
import json
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "QCRI/MemeLens-VLM" 
DATA_PATH = "/scratch/datasets/MAMI/train/mami_train_tiny.jsonl"
OUTPUT_DIR = "/scratch/models/MemeLens_Overfit_Check"

# ==========================================
# CUSTOM DATA COLLATOR (The Fix)
# ==========================================
class QwenCollator:
    """
    Loads images on the fly, processes tokens, and crucially: 
    Masks out the system and user prompts so the model ONLY learns the target label.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        prompt_texts = [ex["prompt_text"] for ex in examples]
        
        # 1. Load images dynamically (prevents Dataset serialization crashes)
        images = []
        for ex in examples:
            if ex["image_path"]:
                images.append(Image.open(ex["image_path"]).convert("RGB"))
            else:
                images.append(None)

        # 2. Let the Qwen Processor handle all complex tokenization and vision grids
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        
        # 3. Clone input_ids to create the target labels
        labels = batch["input_ids"].clone()
        
        # 4. MASKING THE PROMPT
        for i, prompt_text in enumerate(prompt_texts):
            # Tokenize just the prompt to find out exactly how many tokens it takes up
            prompt_inputs = self.processor(text=[prompt_text], padding=False, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # Set all prompt tokens to -100 so the optimizer ignores them
            labels[i, :prompt_len] = -100 
            
        batch["labels"] = labels
        return batch

# ==========================================
# DATA LOADING & FORMATTING
# ==========================================
def load_and_format_dataset(jsonl_path, processor):
    print("Loading and formatting dataset...")
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    hf_dataset = Dataset.from_list(data)
    
    def format_func(example):
        messages = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": [
                {"type": "image", "image": example["image_path"]},
                {"type": "text", "text": example["user_prompt"]}
            ]},
            {"role": "assistant", "content": example["target"]}
        ]
        
        # We isolate the prompt to tell our collator what to mask
        prompt_messages = [messages[0], messages[1]] 
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        return {
            "text": text,
            "prompt_text": prompt_text,
            "image_path": example["image_path"]
        }

    return hf_dataset.map(format_func, remove_columns=hf_dataset.column_names)

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    train_dataset = load_and_format_dataset(DATA_PATH, processor)
    print(f"Loaded {len(train_dataset)} examples for overfit check.")

    print("Loading Base Model in pure FP16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    )
    
    # [V100 FIX]: The Sledgehammer
    model.config.torch_dtype = torch.float16
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)
    
    # Target specific layers, including output layers for strict format learning
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj",
            "fc1", "fc2", "linear_q", "linear_k", "linear_v", "linear_o",
            "lm_head", "embed_tokens"
        ], 
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # V100 Stability Cast
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()

    # Native HF TrainingArguments (No TRL SFTConfig required)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_train_epochs=50, 
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no", 
        optim="adamw_torch", 
        fp16=True, 
        bf16=False, 
        report_to="none",
        remove_unused_columns=False, # Essential: preserves our custom data fields
    )

    # Native HF Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=QwenCollator(processor), # Inject our custom collator
    )

    print("Starting Overfit Training...")
    trainer.train()
    print("✅ Overfit Check Complete.")

if __name__ == "__main__":
    main()