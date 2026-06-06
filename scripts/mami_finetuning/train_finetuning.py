import os
import torch
import torch.distributed as dist
import json
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ==========================================
# PRODUCTION CONFIGURATION
# ==========================================
MODEL_ID = "QCRI/MemeLens-VLM" 
DATA_PATH = "/scratch/datasets/MAMI/train/mami_train_augmented.jsonl" 
OUTPUT_DIR = "/scratch/models/MemeLens_Robust_LoRA" 

# ==========================================
# CUSTOM DATA COLLATOR (Prompt Masking)
# ==========================================
class QwenCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        prompt_texts = [ex["prompt_text"] for ex in examples]
        
        images = []
        for ex in examples:
            if ex["image_path"]:
                images.append(Image.open(ex["image_path"]).convert("RGB"))
            else:
                images.append(None)

        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        
        for i, prompt_text in enumerate(prompt_texts):
            prompt_inputs = self.processor(text=[prompt_text], padding=False, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[i, :prompt_len] = -100 
            
        batch["labels"] = labels
        return batch

# ==========================================
# DATA LOADING & FORMATTING
# ==========================================
def load_and_format_dataset(jsonl_path, processor):
    # Only let Rank 0 print the loading message
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("Loading and formatting full dataset...")
        
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
        
        prompt_messages = [messages[0], messages[1]] 
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        return {
            "text": text,
            "prompt_text": prompt_text,
            "image_path": example["image_path"]
        }

    return hf_dataset.map(format_func, remove_columns=hf_dataset.column_names, desc="Formatting Data")

# ==========================================
# MAIN TRAINING LOOP (DDP ENABLED)
# ==========================================
def main():
    # 1. Initialize Multi-GPU Communication
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank) 

    # 2. THE TRAFFIC COP: Freeze GPUs 1-5 while GPU 0 downloads the model safely
    if local_rank != 0:
        dist.barrier()

    # 3. Process data and load model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    train_dataset = load_and_format_dataset(DATA_PATH, processor)
    
    if local_rank == 0:
        print(f"Loaded {len(train_dataset)} examples for production run.")

    print(f"Loading Base Model on GPU {local_rank} in pure FP16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map={"": local_rank}, # Lock to specific GPU
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    )
    
    # 4. Wake up GPUs 1-5 now that GPU 0 is finished downloading
    if local_rank == 0:
        dist.barrier()

    # =========================================================
    # [V100 FIX]: The Sledgehammer (Banish bfloat16)
    # =========================================================
    model.config.torch_dtype = torch.float16
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)
    
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
    
    # Optimizer Stability Cast
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if local_rank == 0:
        model.print_trainable_parameters()

    # Production Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_train_epochs=2,                
        learning_rate=2e-4,
        logging_steps=10,                  
        save_strategy="epoch",             
        save_total_limit=2,                
        optim="adamw_torch", 
        fp16=True, 
        bf16=False, 
        report_to="none",
        remove_unused_columns=False, 
        dataloader_num_workers=0,          
        ddp_find_unused_parameters=False, # Essential for LoRA + DDP
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=QwenCollator(processor),
    )

    if local_rank == 0:
        print("Starting Full Fine-Tuning Run across 6 GPUs...")
        
    trainer.train()
    
    # Ensure only the main GPU saves the final output to prevent file corruption
    if local_rank == 0:
        print("Saving Final Model Adapters...")
        trainer.save_model(f"{OUTPUT_DIR}/final")
        processor.save_pretrained(f"{OUTPUT_DIR}/final")
        print("✅ Full Fine-Tuning Complete.")

    # Final cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()