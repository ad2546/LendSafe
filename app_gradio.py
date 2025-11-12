"""
LendSafe - Gradio Version
AI-Powered Loan Decision Explainer with Fine-tuned IBM Granite 350M
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None

def load_model():
    """Load the fine-tuned IBM Granite model with LoRA adapters"""
    global model, tokenizer

    if model is not None:
        return True

    try:
        base_model_id = "ibm-granite/granite-4.0-h-350m"
        adapter_repo = "notatharva0699/lendsafe-granite"

        logger.info(f"Loading base model: {base_model_id}")

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

        # Load LoRA adapters directly from HF
        logger.info(f"Loading LoRA adapters from: {adapter_repo}")
        model = PeftModel.from_pretrained(model, adapter_repo)
        model.eval()

        logger.info("✅ Model loaded successfully with fine-tuned adapters!")
        return True

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def calculate_risk_score(credit_score, income, loan_amount, dti_ratio, employment_length):
    """Calculate risk score using rule-based logic"""

    # Base score from credit score
    if credit_score >= 750:
        score = 85
    elif credit_score >= 700:
        score = 75
    elif credit_score >= 650:
        score = 65
    elif credit_score >= 600:
        score = 55
    else:
        score = 45

    # Adjust for DTI
    if dti_ratio < 20:
        score += 10
    elif dti_ratio > 40:
        score -= 15

    # Adjust for loan-to-income ratio
    lti_ratio = (loan_amount / income) * 100
    if lti_ratio < 20:
        score += 5
    elif lti_ratio > 50:
        score -= 10

    # Adjust for employment
    if employment_length >= 5:
        score += 5
    elif employment_length < 1:
        score -= 5

    # Clamp between 0-100
    score = max(0, min(100, score))

    # Determine decision
    if score >= 70:
        decision = "APPROVED"
    elif score >= 50:
        decision = "REVIEW"
    else:
        decision = "DENIED"

    return score, decision

def generate_explanation(loan_data, decision, risk_score):
    """Generate AI explanation using fine-tuned Granite model"""

    if model is None:
        return "❌ Model not loaded. Please wait for initialization."

    # Format prompt
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
- Purpose: {loan_data['purpose']}
- Term: {loan_data['term']} months
- Interest Rate: {loan_data['rate']}%

Financial Metrics:
- Debt-to-Income Ratio: {loan_data['dti']}%
- Risk Score: {risk_score}/100

Decision: {decision}

### Response:
"""

    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
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

        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after "### Response:"
        if "### Response:" in full_text:
            explanation = full_text.split("### Response:")[-1].strip()
        else:
            explanation = full_text[len(prompt):].strip()

        return explanation

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"❌ Error generating explanation: {str(e)}"

def analyze_loan(credit_score, income, employment, home_ownership,
                 loan_amount, purpose, term, rate, dti, revol_util,
                 total_accounts, inquiries):
    """Main function to analyze loan application"""

    # Ensure model is loaded
    if model is None:
        load_success = load_model()
        if not load_success:
            return "❌ Failed to load model", "", "", ""

    # Calculate risk score
    risk_score, decision = calculate_risk_score(
        credit_score, income, loan_amount, dti, employment
    )

    # Prepare loan data
    loan_data = {
        'credit_score': credit_score,
        'income': income,
        'employment': employment,
        'home_ownership': home_ownership,
        'loan_amount': loan_amount,
        'purpose': purpose.replace('_', ' ').title(),
        'term': term,
        'rate': rate,
        'dti': dti
    }

    # Generate explanation
    explanation = generate_explanation(loan_data, decision, risk_score)

    # Format outputs
    decision_color = {
        "APPROVED": "🟢",
        "REVIEW": "🟡",
        "DENIED": "🔴"
    }

    decision_output = f"{decision_color[decision]} **{decision}**"
    risk_output = f"**Risk Score:** {risk_score}/100"

    # Format metrics
    metrics = f"""
**Key Metrics:**
- Credit Score: {credit_score}
- DTI Ratio: {dti}%
- Loan-to-Income: {(loan_amount/income)*100:.1f}%
- Employment: {employment} years
"""

    return decision_output, risk_output, explanation, metrics

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.gr-box {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
.gr-button-primary {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: none;
}
#decision_output {
    font-size: 24px;
    font-weight: bold;
    padding: 20px;
    text-align: center;
}
#explanation_output {
    font-size: 16px;
    line-height: 1.6;
    padding: 20px;
    background: #f9fafb;
    border-radius: 8px;
}
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css, title="LendSafe - AI Loan Explainer") as demo:
    gr.Markdown("""
    # 🏦 LendSafe
    ### AI-Powered Loan Decision Explainer | Fine-tuned IBM Granite 350M

    Get instant, FCRA-compliant explanations for loan decisions using fine-tuned AI.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📋 Loan Application")

            gr.Markdown("### Applicant Information")
            credit_score = gr.Slider(300, 850, value=680, step=1, label="Credit Score")
            income = gr.Number(value=55000, label="Annual Income ($)")
            employment = gr.Slider(0, 40, value=3, step=0.5, label="Employment Length (years)")
            home_ownership = gr.Radio(["RENT", "OWN", "MORTGAGE"], value="RENT", label="Home Ownership")

            gr.Markdown("### Loan Details")
            loan_amount = gr.Number(value=15000, label="Loan Amount ($)")
            purpose = gr.Dropdown(
                ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
                 "small_business", "car", "medical", "moving", "vacation", "house", "other"],
                value="debt_consolidation",
                label="Loan Purpose"
            )
            term = gr.Radio([36, 60], value=36, label="Loan Term (months)")
            rate = gr.Slider(5, 30, value=12.5, step=0.1, label="Interest Rate (%)")

            gr.Markdown("### Financial Metrics")
            dti = gr.Slider(0, 50, value=18.5, step=0.1, label="Debt-to-Income Ratio (%)")
            revol_util = gr.Slider(0, 100, value=65, step=1, label="Revolving Utilization (%)")
            total_accounts = gr.Slider(0, 50, value=12, step=1, label="Total Credit Accounts")
            inquiries = gr.Slider(0, 10, value=1, step=1, label="Credit Inquiries (last 6 months)")

            analyze_btn = gr.Button("🔍 Analyze Application", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("## 📊 Decision & Explanation")

            decision_output = gr.Markdown(elem_id="decision_output")
            risk_output = gr.Markdown()
            metrics_output = gr.Markdown()

            gr.Markdown("### 🤖 AI Explanation")
            explanation_output = gr.Markdown(elem_id="explanation_output")

    # Example scenarios
    gr.Markdown("## 📌 Quick Examples")
    gr.Examples(
        examples=[
            [750, 85000, 8, "OWN", 20000, "home_improvement", 36, 8.5, 15, 30, 18, 0],
            [620, 45000, 2, "RENT", 12000, "debt_consolidation", 60, 18.5, 35, 80, 8, 3],
            [580, 35000, 1, "RENT", 8000, "credit_card", 36, 22.0, 42, 95, 5, 5]
        ],
        inputs=[credit_score, income, employment, home_ownership, loan_amount, purpose,
                term, rate, dti, revol_util, total_accounts, inquiries],
        label="Try these scenarios"
    )

    # Connect button
    analyze_btn.click(
        fn=analyze_loan,
        inputs=[credit_score, income, employment, home_ownership, loan_amount, purpose,
                term, rate, dti, revol_util, total_accounts, inquiries],
        outputs=[decision_output, risk_output, explanation_output, metrics_output]
    )

    gr.Markdown("""
    ---
    **Built with:**
    - 🤖 IBM Granite 4.0 H 350M (Fine-tuned with LoRA)
    - 🚀 Gradio for the interface
    - 🔒 100% local processing (privacy-first)

    [GitHub](https://github.com/ad2546/LendSafe) | [Model on HF](https://huggingface.co/notatharva0699/lendsafe-granite)
    """)

# Load model on startup
logger.info("Initializing model...")
load_model()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
