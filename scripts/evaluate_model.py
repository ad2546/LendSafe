"""
Evaluate fine-tuned Granite model with ROUGE and BERTScore metrics
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# Paths
PROJECT_DIR = Path(__file__).parent.parent
FINETUNED_DIR = PROJECT_DIR / "models" / "granite-finetuned"
DATA_DIR = PROJECT_DIR / "data" / "synthetic"
RESULTS_DIR = PROJECT_DIR / "models" / "evaluation_results"

def load_model():
    """Load fine-tuned model"""
    print("📥 Loading fine-tuned model...")

    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ Model loaded")
    return model, tokenizer


def load_test_data(n_samples=100):
    """Load test examples"""
    print(f"\n📊 Loading {n_samples} test examples...")

    with open(DATA_DIR / "training_examples.json", 'r') as f:
        all_examples = json.load(f)

    # Use last 100 as test set (not used in training)
    test_examples = all_examples[-n_samples:]

    print(f"✅ Loaded {len(test_examples)} test examples")
    return test_examples


def generate_prediction(model, tokenizer, example):
    """Generate explanation for a single example"""

    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the generated part (after the prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response after "### Response:"
    if "### Response:" in full_response:
        prediction = full_response.split("### Response:")[-1].strip()
    else:
        prediction = full_response.strip()

    return prediction


def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores"""
    print("\n📊 Calculating ROUGE scores...")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    results = {
        'rouge1': {
            'mean': sum(rouge1_scores) / len(rouge1_scores),
            'scores': rouge1_scores
        },
        'rouge2': {
            'mean': sum(rouge2_scores) / len(rouge2_scores),
            'scores': rouge2_scores
        },
        'rougeL': {
            'mean': sum(rougeL_scores) / len(rougeL_scores),
            'scores': rougeL_scores
        }
    }

    print(f"✅ ROUGE-1 F1: {results['rouge1']['mean']:.4f}")
    print(f"✅ ROUGE-2 F1: {results['rouge2']['mean']:.4f}")
    print(f"✅ ROUGE-L F1: {results['rougeL']['mean']:.4f}")

    return results


def calculate_bert_scores(predictions, references):
    """Calculate BERTScore"""
    print("\n📊 Calculating BERTScore...")

    P, R, F1 = bert_score(
        predictions,
        references,
        lang='en',
        verbose=False,
        device='cpu'  # Use CPU to avoid memory issues
    )

    results = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item(),
        'scores': {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    }

    print(f"✅ BERTScore Precision: {results['precision']:.4f}")
    print(f"✅ BERTScore Recall: {results['recall']:.4f}")
    print(f"✅ BERTScore F1: {results['f1']:.4f}")

    return results


def evaluate_model(model, tokenizer, test_examples, n_eval=50):
    """Run full evaluation"""
    print(f"\n🧪 Evaluating on {n_eval} examples...")

    predictions = []
    references = []

    # Generate predictions
    for i, example in enumerate(tqdm(test_examples[:n_eval], desc="Generating")):
        try:
            pred = generate_prediction(model, tokenizer, example)
            predictions.append(pred)
            references.append(example['output'])
        except Exception as e:
            print(f"\n⚠️  Error on example {i}: {e}")
            predictions.append("")
            references.append(example['output'])

    # Calculate metrics
    rouge_results = calculate_rouge_scores(predictions, references)
    bert_results = calculate_bert_scores(predictions, references)

    return {
        'rouge': rouge_results,
        'bertscore': bert_results,
        'predictions': predictions,
        'references': references
    }


def save_results(results, test_examples):
    """Save evaluation results"""
    print("\n💾 Saving results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics summary
    summary = {
        'rouge1_f1': results['rouge']['rouge1']['mean'],
        'rouge2_f1': results['rouge']['rouge2']['mean'],
        'rougeL_f1': results['rouge']['rougeL']['mean'],
        'bertscore_precision': results['bertscore']['precision'],
        'bertscore_recall': results['bertscore']['recall'],
        'bertscore_f1': results['bertscore']['f1'],
    }

    with open(RESULTS_DIR / "metrics_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    detailed_results = []
    for i, (pred, ref) in enumerate(zip(results['predictions'], results['references'])):
        detailed_results.append({
            'example_id': i,
            'prediction': pred,
            'reference': ref,
            'rouge1': results['rouge']['rouge1']['scores'][i],
            'rouge2': results['rouge']['rouge2']['scores'][i],
            'rougeL': results['rouge']['rougeL']['scores'][i],
            'bertscore_f1': results['bertscore']['scores']['f1'][i]
        })

    df = pd.DataFrame(detailed_results)
    df.to_csv(RESULTS_DIR / "detailed_results.csv", index=False)

    print(f"✅ Results saved to: {RESULTS_DIR}")


def display_sample_outputs(results, test_examples, n=3):
    """Display sample predictions"""
    print("\n" + "="*60)
    print(f"SAMPLE PREDICTIONS (showing {n} examples)")
    print("="*60)

    for i in range(min(n, len(results['predictions']))):
        print(f"\n--- Example {i+1} ({test_examples[i]['decision'].upper()}) ---")
        print(f"\nInput:\n{test_examples[i]['input']}")
        print(f"\nReference:\n{results['references'][i]}")
        print(f"\nPrediction:\n{results['predictions'][i]}")
        print(f"\nROUGE-L: {results['rouge']['rougeL']['scores'][i]:.4f}")
        print(f"BERTScore F1: {results['bertscore']['scores']['f1'][i]:.4f}")
        print("-"*60)


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("MODEL EVALUATION - ROUGE & BERTSCORE")
    print("="*60)

    # Load model
    model, tokenizer = load_model()

    # Load test data
    test_examples = load_test_data(n_samples=100)

    # Evaluate (use 50 examples for speed)
    results = evaluate_model(model, tokenizer, test_examples, n_eval=50)

    # Save results
    save_results(results, test_examples)

    # Display samples
    display_sample_outputs(results, test_examples, n=3)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"ROUGE-1 F1:    {results['rouge']['rouge1']['mean']:.4f}")
    print(f"ROUGE-2 F1:    {results['rouge']['rouge2']['mean']:.4f}")
    print(f"ROUGE-L F1:    {results['rouge']['rougeL']['mean']:.4f}")
    print(f"BERTScore F1:  {results['bertscore']['f1']:.4f}")
    print("="*60)

    # Performance assessment
    rougeL = results['rouge']['rougeL']['mean']
    bert_f1 = results['bertscore']['f1']

    print("\n✅ PERFORMANCE ASSESSMENT:")
    if rougeL >= 0.5 and bert_f1 >= 0.85:
        print("🎉 EXCELLENT - Model generates high-quality explanations")
    elif rougeL >= 0.4 and bert_f1 >= 0.80:
        print("👍 GOOD - Model generates acceptable explanations")
    elif rougeL >= 0.3:
        print("⚠️  FAIR - Model needs more training or data")
    else:
        print("❌ POOR - Model requires significant improvements")


if __name__ == "__main__":
    main()
