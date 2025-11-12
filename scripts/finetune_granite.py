"""
Fine-tune IBM Granite model with LoRA for loan explanations
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from pathlib import Path
import json

# Paths
PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / "models" / "granite-lendsafe"
FINETUNED_DIR = PROJECT_DIR / "models" / "granite-finetuned"
DATA_DIR = PROJECT_DIR / "data" / "synthetic"

# Configuration (Optimized for 350M model on CPU)
MAX_LENGTH = 256  # Reduced for faster CPU training
BATCH_SIZE = 2    # Smaller batch for CPU
GRADIENT_ACCUMULATION_STEPS = 8  # Maintain effective batch size of 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100


def load_model_and_tokenizer():
    """Load base Granite model and tokenizer"""
    print("📥 Loading IBM Granite model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use CPU to ensure compatibility (350M model is small enough)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,  # FP32 for CPU
        device_map="cpu",  # Force CPU to avoid MPS issues
        low_cpu_mem_usage=True
    )

    print(f"✅ Model loaded: {model.num_parameters():,} parameters")

    return model, tokenizer


def configure_lora(model):
    """Configure LoRA for parameter-efficient fine-tuning"""
    print("\n🔧 Configuring LoRA...")

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                          # Rank
        lora_alpha=32,                 # Alpha scaling
        target_modules=[               # Modules to apply LoRA
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"✅ LoRA configured:")
    print(f"   Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"   Total params: {total_params:,}")
    print(f"   Memory savings: {100 - trainable_pct:.2f}% parameters frozen")

    return model


def prepare_dataset(tokenizer):
    """Load and prepare training dataset"""
    print("\n📊 Preparing dataset...")

    # Load JSONL dataset
    dataset = load_dataset(
        'json',
        data_files=str(DATA_DIR / "training_examples.jsonl"),
        split='train'
    )

    print(f"✅ Loaded {len(dataset)} examples")

    # Format prompt
    def format_prompt(example):
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        return {"text": prompt}

    # Apply formatting
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Split into train/validation (90/10)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"✅ Training samples: {len(split_dataset['train'])}")
    print(f"✅ Validation samples: {len(split_dataset['test'])}")

    return split_dataset


def train_model(model, tokenizer, dataset):
    """Fine-tune model with LoRA"""
    print("\n🚀 Starting fine-tuning...")

    # Create output directory
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(FINETUNED_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=False,                             # Disabled for CPU training
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        warmup_steps=WARMUP_STEPS,
        load_best_model_at_end=True,
        report_to="none",                       # Disable W&B, TensorBoard, etc.
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Train
    print("\n⏰ Training will take 30-60 minutes on M2 MacBook Air...")
    print("💡 Monitor training loss - should decrease steadily\n")

    trainer.train()

    print("\n✅ Fine-tuning complete!")

    # Save final model
    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)

    # Save training metrics
    metrics = trainer.state.log_history
    with open(FINETUNED_DIR / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Model saved to: {FINETUNED_DIR}")

    return trainer


def test_inference(model, tokenizer):
    """Test the fine-tuned model"""
    print("\n🧪 Testing fine-tuned model...")

    # Test prompt
    test_prompt = """### Instruction:
Explain why this loan application was approved.

### Input:
Credit Score: 720
Debt-to-Income Ratio: 28%
Loan Amount: $25,000
Annual Income: $85,000
Employment Length: 5 years
Delinquencies (2 yrs): 0
Credit Inquiries (6 mo): 1

### Response:
"""

    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    print("🤖 Generating explanation...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("="*60)
    print("TEST GENERATION:")
    print("="*60)
    print(response)
    print("="*60)


def main():
    """Main fine-tuning pipeline"""
    print("="*60)
    print("GRANITE MODEL FINE-TUNING WITH LORA")
    print("="*60)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Configure LoRA
    model = configure_lora(model)

    # Prepare dataset
    dataset = prepare_dataset(tokenizer)

    # Train model
    trainer = train_model(model, tokenizer, dataset)

    # Test inference
    test_inference(model, tokenizer)

    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"\nFine-tuned model saved to: {FINETUNED_DIR}")
    print("\nNext steps:")
    print("1. Evaluate with ROUGE/BERTScore metrics")
    print("2. Test on diverse loan scenarios")
    print("3. Compare with base model performance")


if __name__ == "__main__":
    main()
