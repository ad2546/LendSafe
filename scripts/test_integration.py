"""
Integration test for LendSafe system
Tests the full pipeline: Risk Model -> LLM Explainer
"""

import sys
from pathlib import Path
import pickle
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_explainer import GraniteLoanExplainer


def load_risk_model():
    """Load the risk scoring model"""
    risk_model_path = Path("models/risk_model/loan_risk_model.pkl")
    if risk_model_path.exists():
        with open(risk_model_path, "rb") as f:
            return pickle.load(f)
    return None


def calculate_risk_score(loan_data, risk_model):
    """Calculate risk score"""
    if risk_model is None:
        print("⚠️  Risk model not found, using rule-based decision")
        return None

    features = [
        loan_data['credit_score'],
        loan_data['annual_income'],
        loan_data['loan_amount'],
        loan_data['dti'],
        loan_data['revol_util'],
        loan_data['emp_length'],
        loan_data['total_acc'],
        loan_data['inq_last_6mths'],
        1 if loan_data['home_ownership'] == 'OWN' else 0,
        1 if loan_data['home_ownership'] == 'RENT' else 0,
    ]

    df = pd.DataFrame([features], columns=[
        'credit_score', 'annual_inc', 'loan_amnt', 'dti', 'revol_util',
        'emp_length', 'total_acc', 'inq_last_6mths', 'home_own', 'home_rent'
    ])

    try:
        risk_prob = risk_model.predict_proba(df)[0][1]
        return risk_prob * 100
    except Exception as e:
        print(f"⚠️  Risk calculation error: {e}")
        return None


def test_full_pipeline():
    """Test the complete LendSafe pipeline"""

    print("\n" + "=" * 70)
    print("🧪 LENDSAFE INTEGRATION TEST")
    print("=" * 70)

    # Test cases
    test_cases = [
        {
            "name": "Strong Applicant (Expected: APPROVED)",
            "data": {
                'credit_score': 750,
                'annual_income': 85000,
                'emp_length': 10,
                'home_ownership': 'OWN',
                'loan_amount': 20000,
                'purpose': 'debt_consolidation',
                'int_rate': 8.5,
                'term': 36,
                'dti': 12.0,
                'revol_util': 35.0,
                'total_acc': 15,
                'inq_last_6mths': 0
            }
        },
        {
            "name": "Risky Applicant (Expected: DENIED)",
            "data": {
                'credit_score': 580,
                'annual_income': 35000,
                'emp_length': 1,
                'home_ownership': 'RENT',
                'loan_amount': 30000,
                'purpose': 'credit_card',
                'int_rate': 18.5,
                'term': 60,
                'dti': 42.0,
                'revol_util': 95.0,
                'total_acc': 8,
                'inq_last_6mths': 5
            }
        },
        {
            "name": "Borderline Applicant (Expected: APPROVED with conditions)",
            "data": {
                'credit_score': 680,
                'annual_income': 55000,
                'emp_length': 5,
                'home_ownership': 'RENT',
                'loan_amount': 15000,
                'purpose': 'home_improvement',
                'int_rate': 12.5,
                'term': 36,
                'dti': 28.5,
                'revol_util': 65.0,
                'total_acc': 12,
                'inq_last_6mths': 2
            }
        }
    ]

    # Load models
    print("\n📦 Loading models...")
    print("-" * 70)

    try:
        risk_model = load_risk_model()
        if risk_model:
            print("✅ Risk scoring model loaded")
        else:
            print("⚠️  Risk model not found, using rule-based decisions")

        explainer = GraniteLoanExplainer()
        print("✅ LLM explainer loaded (IBM Granite 350M)")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "=" * 70)
        print(f"TEST CASE {i}: {test_case['name']}")
        print("=" * 70)

        loan_data = test_case['data']

        # Display key metrics
        print("\n📊 Application Summary:")
        print(f"   Credit Score: {loan_data['credit_score']}")
        print(f"   Annual Income: ${loan_data['annual_income']:,}")
        print(f"   Loan Amount: ${loan_data['loan_amount']:,}")
        print(f"   DTI Ratio: {loan_data['dti']}%")
        print(f"   Revolving Utilization: {loan_data['revol_util']}%")
        print(f"   Employment Length: {loan_data['emp_length']} years")

        # Calculate risk score
        print("\n🎯 Risk Assessment:")
        risk_score = calculate_risk_score(loan_data, risk_model)

        if risk_score is not None:
            print(f"   Risk Score: {risk_score:.2f}%")
            decision = "DENIED" if risk_score > 50 else "APPROVED"
        else:
            # Simple rule-based fallback
            if loan_data['credit_score'] < 620 or loan_data['dti'] > 40 or loan_data['revol_util'] > 90:
                decision = "DENIED"
            else:
                decision = "APPROVED"

        print(f"   Decision: {decision}")

        # Generate explanation
        print("\n🤖 AI Explanation:")
        print("-" * 70)

        try:
            explanation = explainer.explain_decision(
                loan_data,
                decision,
                risk_score,
                max_new_tokens=200,
                temperature=0.7
            )
            print(explanation)

        except Exception as e:
            print(f"❌ Error generating explanation: {e}")

        print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETED")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Open the web interface in your browser")
    print("3. Test with different loan applications")
    print("4. Generate PDF reports for adverse actions")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    test_full_pipeline()
