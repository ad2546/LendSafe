"""
Generate synthetic loan explanations for training data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Explanation templates
APPROVAL_TEMPLATES = [
    "Based on your {credit_score} credit score and {dti_ratio}% debt-to-income ratio, your loan application has been approved. Your strong credit history demonstrates reliable repayment behavior.",
    "We are pleased to approve your ${loan_amount:,} loan request. Your excellent credit score of {credit_score} and stable income of ${annual_income:,} per year meet our lending criteria.",
    "Your loan application is approved. Key factors include your {credit_score} FICO score, low debt-to-income ratio of {dti_ratio}%, and {employment_length} years of employment history.",
    "Congratulations! Your loan is approved based on strong creditworthiness indicators: {credit_score} credit score, ${annual_income:,} annual income, and minimal delinquencies.",
]

REJECTION_TEMPLATES = [
    "We regret to inform you that your loan application has been declined. Primary reasons include your credit score of {credit_score} (below our 640 minimum) and debt-to-income ratio of {dti_ratio}% (exceeds our 43% threshold).",
    "Your ${loan_amount:,} loan request cannot be approved at this time. Key adverse factors: credit score {credit_score} indicates elevated risk, and {delinq_2yrs} recent delinquencies raise concerns about repayment ability.",
    "Unfortunately, we cannot approve your application. Your {credit_score} credit score and {dti_ratio}% DTI ratio do not meet our current underwriting standards for a ${loan_amount:,} loan.",
    "Application declined. Adverse action reasons per FCRA Section 615: (1) Credit score {credit_score} below acceptable range, (2) DTI ratio {dti_ratio}% exceeds policy limits, (3) Recent credit inquiries indicate financial stress.",
]

def load_processed_data():
    """Load processed loan data"""
    print("📥 Loading processed loan data...")

    train_data = pd.read_csv(PROCESSED_DIR / "train.csv")

    print(f"✅ Loaded {len(train_data)} loan applications")

    return train_data


def generate_explanation(row, decision):
    """Generate explanation for a single loan decision"""

    # Select template
    if decision == 1:  # Approved
        template = np.random.choice(APPROVAL_TEMPLATES)
    else:  # Rejected
        template = np.random.choice(REJECTION_TEMPLATES)

    # Format explanation
    try:
        explanation = template.format(
            loan_amount=int(row.get('loan_amount', 0)),
            credit_score=int(row.get('credit_score', 0)),
            annual_income=int(row.get('annual_income', 0)),
            dti_ratio=round(row.get('dti_ratio', 0), 1),
            employment_length=int(row.get('employment_length', 0)),
            delinq_2yrs=int(row.get('delinq_2yrs', 0))
        )
    except KeyError as e:
        # Fallback for missing fields
        if decision == 1:
            explanation = f"Your loan application has been approved based on your creditworthiness."
        else:
            explanation = f"Your loan application has been declined due to credit risk factors."

    return explanation


def create_training_examples(data, n_samples=100):
    """Create training examples in instruction format"""
    print(f"\n🎯 Generating {n_samples} training examples...")

    # Sample data
    sampled = data.sample(n=min(n_samples, len(data)), random_state=42)

    training_examples = []

    for idx, row in sampled.iterrows():
        decision = row['approved']
        decision_text = "approved" if decision == 1 else "rejected"

        # Generate explanation
        explanation = generate_explanation(row, decision)

        # Create instruction format
        example = {
            "instruction": f"Explain why this loan application was {decision_text}.",
            "input": f"Credit Score: {int(row['credit_score'])}\n"
                    f"Debt-to-Income Ratio: {round(row['dti_ratio'], 1)}%\n"
                    f"Loan Amount: ${int(row['loan_amount']):,}\n"
                    f"Annual Income: ${int(row['annual_income']):,}\n"
                    f"Employment Length: {int(row['employment_length'])} years\n"
                    f"Delinquencies (2 yrs): {int(row['delinq_2yrs'])}\n"
                    f"Credit Inquiries (6 mo): {int(row['inquiries_6mon'])}",
            "output": explanation,
            "decision": decision_text
        }

        training_examples.append(example)

    print(f"✅ Generated {len(training_examples)} examples")
    print(f"   Approved: {sum(1 for e in training_examples if e['decision'] == 'approved')}")
    print(f"   Rejected: {sum(1 for e in training_examples if e['decision'] == 'rejected')}")

    return training_examples


def save_training_data(examples):
    """Save training examples"""
    print("\n💾 Saving training data...")

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(SYNTHETIC_DIR / "training_examples.json", 'w') as f:
        json.dump(examples, f, indent=2)

    # Save as JSONL for easy streaming
    with open(SYNTHETIC_DIR / "training_examples.jsonl", 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    # Save as CSV for analysis
    df = pd.DataFrame(examples)
    df.to_csv(SYNTHETIC_DIR / "training_examples.csv", index=False)

    print(f"✅ Saved to: {SYNTHETIC_DIR}")
    print(f"   - training_examples.json")
    print(f"   - training_examples.jsonl")
    print(f"   - training_examples.csv")


def show_sample_examples(examples, n=3):
    """Display sample training examples"""
    print("\n" + "="*60)
    print(f"SAMPLE TRAINING EXAMPLES (showing {n} of {len(examples)})")
    print("="*60)

    for i, example in enumerate(examples[:n]):
        print(f"\n--- Example {i+1} ({example['decision'].upper()}) ---")
        print(f"\n### Instruction:")
        print(example['instruction'])
        print(f"\n### Input:")
        print(example['input'])
        print(f"\n### Output:")
        print(example['output'])
        print("\n" + "-"*60)


def main():
    """Main pipeline"""
    print("="*60)
    print("SYNTHETIC LOAN EXPLANATION GENERATOR")
    print("="*60)

    # Load data
    data = load_processed_data()

    # Generate training examples (1000+ for production fine-tuning)
    examples = create_training_examples(data, n_samples=1500)

    # Save training data
    save_training_data(examples)

    # Show samples
    show_sample_examples(examples, n=3)

    print("\n" + "="*60)
    print("✅ TRAINING DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print("1. Review examples in {SYNTHETIC_DIR}")
    print("2. Use training_examples.jsonl for fine-tuning")
    print("3. Expand to 1000+ examples for production use")


if __name__ == "__main__":
    main()
