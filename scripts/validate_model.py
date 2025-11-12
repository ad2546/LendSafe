"""
LendSafe Model Validation Script
Tests the fine-tuned model and exports results
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime
from pathlib import Path
import time

# Test cases
TEST_CASES = [
    {
        "name": "Strong Approval",
        "credit_score": 750,
        "income": 85000,
        "dti": 15,
        "employment": 8,
        "home_ownership": "OWN",
        "loan_amount": 20000,
        "purpose": "home_improvement",
        "expected_decision": "APPROVED"
    },
    {
        "name": "Marginal Approval",
        "credit_score": 680,
        "income": 55000,
        "dti": 28,
        "employment": 3,
        "home_ownership": "RENT",
        "loan_amount": 15000,
        "purpose": "debt_consolidation",
        "expected_decision": "APPROVED"
    },
    {
        "name": "Review Required",
        "credit_score": 650,
        "income": 50000,
        "dti": 35,
        "employment": 2,
        "home_ownership": "RENT",
        "loan_amount": 18000,
        "purpose": "credit_card",
        "expected_decision": "REVIEW"
    },
    {
        "name": "Denial Case",
        "credit_score": 580,
        "income": 35000,
        "dti": 45,
        "employment": 1,
        "home_ownership": "RENT",
        "loan_amount": 12000,
        "purpose": "debt_consolidation",
        "expected_decision": "DENIED"
    }
]

def load_model():
    """Load the fine-tuned model"""
    print("🔧 Loading model...")

    base_model_id = "ibm-granite/granite-4.0-h-350m"
    adapter_repo = "notatharva0699/lendsafe-granite"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Load LoRA adapters
    print(f"📥 Loading LoRA adapters from {adapter_repo}...")
    model = PeftModel.from_pretrained(model, adapter_repo)
    model.eval()

    print("✅ Model loaded successfully!")
    return model, tokenizer

def calculate_risk_score(loan_data):
    """Calculate risk score"""
    score = 50

    # Credit score factor
    if loan_data['credit_score'] >= 750:
        score += 35
    elif loan_data['credit_score'] >= 700:
        score += 25
    elif loan_data['credit_score'] >= 650:
        score += 15
    elif loan_data['credit_score'] >= 600:
        score += 5
    else:
        score -= 10

    # DTI factor
    if loan_data['dti'] < 20:
        score += 10
    elif loan_data['dti'] > 40:
        score -= 15

    # Employment factor
    if loan_data['employment'] >= 5:
        score += 5
    elif loan_data['employment'] < 1:
        score -= 5

    score = max(0, min(100, score))

    # Determine decision
    if score >= 70:
        decision = "APPROVED"
    elif score >= 50:
        decision = "REVIEW"
    else:
        decision = "DENIED"

    return score, decision

def generate_explanation(model, tokenizer, loan_data, decision, risk_score):
    """Generate explanation using the model"""

    prompt = f"""### Instruction:
You are a loan officer AI assistant. Explain why this loan application was {decision}.
Provide clear, FCRA-compliant reasoning in 2-3 sentences.

### Input:
Applicant Profile:
- Credit Score: {loan_data['credit_score']}
- Annual Income: ${loan_data['income']:,}
- Employment: {loan_data['employment']} years
- Home Ownership: {loan_data['home_ownership']}

Loan Request:
- Amount: ${loan_data['loan_amount']:,}
- Purpose: {loan_data['purpose'].replace('_', ' ').title()}

Financial Metrics:
- Debt-to-Income Ratio: {loan_data['dti']}%
- Risk Score: {risk_score}/100

Decision: {decision}

### Response:
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    inference_time = time.time() - start_time

    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response
    if "### Response:" in full_text:
        explanation = full_text.split("### Response:")[-1].strip()
    else:
        explanation = full_text[len(prompt):].strip()

    return explanation, inference_time

def validate_model():
    """Run validation on test cases"""

    print("\n" + "="*70)
    print("🏦 LendSafe Model Validation")
    print("="*70 + "\n")

    # Load model
    model, tokenizer = load_model()

    # Run test cases
    results = []

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*70}")

        # Calculate risk score
        risk_score, decision = calculate_risk_score(test_case)

        print(f"\n📊 Risk Assessment:")
        print(f"   Risk Score: {risk_score}/100")
        print(f"   Decision: {decision}")
        print(f"   Expected: {test_case['expected_decision']}")

        decision_match = decision == test_case['expected_decision']
        print(f"   Match: {'✅' if decision_match else '❌'}")

        # Generate explanation
        print(f"\n🤖 Generating explanation...")
        explanation, inference_time = generate_explanation(
            model, tokenizer, test_case, decision, risk_score
        )

        print(f"   Inference time: {inference_time:.2f}s")
        print(f"\n📝 Explanation:")
        print(f"   {explanation}")

        # Store results
        results.append({
            "test_case": test_case['name'],
            "applicant_data": {
                "credit_score": test_case['credit_score'],
                "income": test_case['income'],
                "dti": test_case['dti'],
                "employment": test_case['employment'],
                "home_ownership": test_case['home_ownership'],
                "loan_amount": test_case['loan_amount'],
                "purpose": test_case['purpose']
            },
            "risk_score": risk_score,
            "decision": decision,
            "expected_decision": test_case['expected_decision'],
            "decision_match": decision_match,
            "explanation": explanation,
            "inference_time_seconds": round(inference_time, 2),
            "explanation_length": len(explanation)
        })

    # Export results
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"validation_{timestamp}.json"

    # Calculate summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['decision_match'])
    avg_inference_time = sum(r['inference_time_seconds'] for r in results) / total_tests
    avg_explanation_length = sum(r['explanation_length'] for r in results) / total_tests

    summary = {
        "validation_date": datetime.now().isoformat(),
        "model": {
            "base_model": "ibm-granite/granite-4.0-h-350m",
            "fine_tuned_model": "notatharva0699/lendsafe-granite",
            "adapter_type": "LoRA",
            "parameters": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05
            }
        },
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "accuracy": round(passed_tests / total_tests * 100, 2),
            "avg_inference_time_seconds": round(avg_inference_time, 2),
            "avg_explanation_length_chars": round(avg_explanation_length)
        },
        "test_results": results
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*70}")
    print("📊 Validation Summary")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Avg Inference Time: {avg_inference_time:.2f}s")
    print(f"Avg Explanation Length: {avg_explanation_length:.0f} characters")
    print(f"\n✅ Results saved to: {output_file}")
    print(f"{'='*70}\n")

    return summary

if __name__ == "__main__":
    try:
        summary = validate_model()
        print("\n🎉 Validation complete!")
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
