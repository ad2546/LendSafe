"""
LendSafe LLM Explainer - Fine-tuned IBM Granite 350M
Generates FCRA-compliant loan decision explanations
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraniteLoanExplainer:
    """
    Loads fine-tuned IBM Granite 4.0 H 350M model and generates
    loan decision explanations with regulatory compliance.
    """

    def __init__(
        self,
        base_model_path: str = "ibm-granite/granite-4.0-h-350m",
        adapter_path: str = "models/granite-finetuned",
        device: str = "auto"
    ):
        """
        Initialize the loan explainer.

        Args:
            base_model_path: Path to base Granite model
            adapter_path: Path to fine-tuned LoRA adapters
            device: Device to run inference on ('auto', 'cpu', 'mps', 'cuda')
        """
        logger.info(f"Loading base model: {base_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon (MPS) acceleration")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA GPU")
            else:
                device = "cpu"
                logger.info("Using CPU")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else None,
            low_cpu_mem_usage=True
        )

        # Move to MPS if needed
        if device == "mps":
            self.model = self.model.to("mps")

        # Load LoRA adapters
        logger.info(f"Loading fine-tuned adapters from: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self.device = device
        logger.info("Model loaded successfully!")

    def format_prompt(
        self,
        loan_data: Dict,
        decision: str,
        risk_score: Optional[float] = None
    ) -> str:
        """
        Format loan application data into the instruction prompt.

        Args:
            loan_data: Dictionary containing loan application data
            decision: 'APPROVED' or 'DENIED'
            risk_score: Optional risk score (0-100)

        Returns:
            Formatted prompt string
        """
        prompt = f"""### Instruction:
You are a loan decision explainer. Provide a clear, regulatory-compliant explanation for why this loan was {decision}.

### Input:
Applicant Information:
- Credit Score: {loan_data.get('credit_score', 'N/A')}
- Annual Income: ${loan_data.get('annual_income', 0):,.2f}
- Employment Length: {loan_data.get('emp_length', 'N/A')} years
- Home Ownership: {loan_data.get('home_ownership', 'N/A')}

Loan Details:
- Loan Amount: ${loan_data.get('loan_amount', 0):,.2f}
- Purpose: {loan_data.get('purpose', 'N/A')}
- Interest Rate: {loan_data.get('int_rate', 0):.2f}%
- Term: {loan_data.get('term', 'N/A')} months

Financial Metrics:
- Debt-to-Income Ratio: {loan_data.get('dti', 0):.2f}%
- Revolving Utilization: {loan_data.get('revol_util', 0):.2f}%
- Total Credit Lines: {loan_data.get('total_acc', 0)}
- Recent Inquiries: {loan_data.get('inq_last_6mths', 0)}
"""

        if risk_score is not None:
            prompt += f"- Risk Score: {risk_score:.1f}/100\n"

        prompt += f"\nDecision: {decision}\n\n### Response:\n"

        return prompt

    def explain_decision(
        self,
        loan_data: Dict,
        decision: str,
        risk_score: Optional[float] = None,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate explanation for loan decision.

        Args:
            loan_data: Dictionary containing loan application data
            decision: 'APPROVED' or 'DENIED'
            risk_score: Optional risk score
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter

        Returns:
            Generated explanation text
        """
        # Format prompt
        prompt = self.format_prompt(loan_data, decision, risk_score)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Move to device
        if self.device == "mps" or self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode and extract response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (after ### Response:)
        if "### Response:" in full_output:
            explanation = full_output.split("### Response:")[-1].strip()
        else:
            explanation = full_output.strip()

        return explanation

    def batch_explain(
        self,
        loan_data_list: list,
        decisions: list,
        risk_scores: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Generate explanations for multiple loan applications.

        Args:
            loan_data_list: List of loan data dictionaries
            decisions: List of decisions ('APPROVED' or 'DENIED')
            risk_scores: Optional list of risk scores
            **kwargs: Additional generation parameters

        Returns:
            List of explanation strings
        """
        if risk_scores is None:
            risk_scores = [None] * len(loan_data_list)

        explanations = []
        for loan_data, decision, risk_score in zip(loan_data_list, decisions, risk_scores):
            explanation = self.explain_decision(
                loan_data,
                decision,
                risk_score,
                **kwargs
            )
            explanations.append(explanation)

        return explanations


# Example usage
if __name__ == "__main__":
    # Test the explainer
    explainer = GraniteLoanExplainer()

    # Sample loan application
    sample_loan = {
        'credit_score': 680,
        'annual_income': 55000,
        'emp_length': 5,
        'home_ownership': 'RENT',
        'loan_amount': 15000,
        'purpose': 'debt_consolidation',
        'int_rate': 12.5,
        'term': 36,
        'dti': 18.5,
        'revol_util': 65.0,
        'total_acc': 12,
        'inq_last_6mths': 1
    }

    # Generate explanation
    print("=" * 60)
    print("LOAN DECISION EXPLANATION")
    print("=" * 60)

    explanation = explainer.explain_decision(
        sample_loan,
        decision="APPROVED",
        risk_score=42.5
    )

    print(explanation)
    print("=" * 60)
