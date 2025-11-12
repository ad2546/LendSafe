"""
Download and test IBM Granite 4.0 H 350M model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

# Model configuration
MODEL_ID = "ibm-granite/granite-4.0-h-350m"  # Smaller 350M model - fits in M2 8GB RAM
MODEL_DIR = Path(__file__).parent.parent / "models" / "granite-lendsafe"

def download_granite_model():
    """Download IBM Granite model from Hugging Face"""
    print(f"🔍 Downloading IBM Granite model: {MODEL_ID}")
    print(f"📁 Saving to: {MODEL_DIR}")

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download tokenizer
        print("\n📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(MODEL_DIR)
        print("✅ Tokenizer downloaded successfully")

        # Download model (using float16 for efficiency)
        print("\n📥 Downloading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(MODEL_DIR)
        print("✅ Model downloaded successfully")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise


def test_granite_inference(model, tokenizer):
    """Test basic inference with Granite"""
    print("\n🧪 Testing Granite inference...")

    # Test prompt
    test_prompt = """### Instruction:
Explain why a loan application was approved.

### Input:
Credit Score: 720
Debt-to-Income Ratio: 28%
Loan Amount: $25,000
Employment: 5 years

### Response:
"""

    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")

    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate
    print("🤖 Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "="*60)
    print("TEST GENERATION:")
    print("="*60)
    print(response)
    print("="*60)

    # Check memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"\n💾 GPU Memory Used: {memory_used:.2f} GB")

    print("\n✅ Inference test completed successfully!")


def main():
    """Main function"""
    print("=" * 60)
    print("IBM GRANITE MODEL DOWNLOAD & TEST")
    print("=" * 60)

    # Check if model already exists
    if (MODEL_DIR / "config.json").exists():
        print(f"\n⚠️  Model already exists at {MODEL_DIR}")
        print("Loading existing model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        # Download model
        model, tokenizer = download_granite_model()

    # Test inference
    test_granite_inference(model, tokenizer)

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Granite is ready for fine-tuning!")
    print("="*60)


if __name__ == "__main__":
    main()
