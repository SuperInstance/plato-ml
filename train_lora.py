#!/usr/bin/env python3
"""
PLATO-ML → LoRA Training Script for OCI GPU
Install dependencies: pip install torch transformers peft datasets accelerate
"""
import json, os, argparse
from pathlib import Path

def load_data(data_dir):
    data = []
    for jsonl in Path(data_dir).rglob("*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                if "instruction" in entry and "output" in entry:
                    data.append({
                        "instruction": entry["instruction"],
                        "input": entry.get("input", ""),
                        "output": entry["output"],
                        "type": entry.get("metadata", {}).get("type", "unknown"),
                        "priority": entry.get("metadata", {}).get("priority", "normal")
                    })
    return data

def format_for_training(entry):
    if entry["input"]:
        prompt = f"### Instruction:\n{entry['instruction']}\n\n### Input:\n{entry['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"
    return {"text": prompt + entry["output"]}

def train_lora(data_dir, output_dir, model_name="Qwen/Qwen2.5-7B", epochs=3, batch_size=4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    print(f"Loaded {len(data)} entries")
    
    # Prioritize high-priority entries
    high = [d for d in data if d["priority"] == "high"]
    normal = [d for d in data if d["priority"] != "high"]
    data = high * 3 + normal  # Oversample high priority 3x
    print(f"After priority weighting: {len(data)} entries")
    
    formatted = [format_for_training(d) for d in data]
    dataset = Dataset.from_list(formatted)
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        optim="adamw_torch"
    )
    
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=2048, padding=False)
    
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )
    
    print("Starting training...")
    trainer.train()
    
    adapter_path = os.path.join(output_dir, "cocapn-lora-adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
    print(f"Size: {sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fn in os.walk(adapter_path) for f in fn) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/.openclaw/workspace/training-data")
    parser.add_argument("--output_dir", default="./cocapn-lora-output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    try:
        train_lora(args.data_dir, args.output_dir, args.model, args.epochs, args.batch_size)
    except ImportError:
        print("\nERROR: Dependencies not installed. Install with:")
        print("  pip install torch transformers peft datasets accelerate")
        print("\nFalling back to data preparation only...")
        prepare_only(args.data_dir, args.output_dir)