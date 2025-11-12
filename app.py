"""
LendSafe - Local Loan Decision Explainer
Streamlit App for Demo and Testing
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.llm_explainer import GraniteLoanExplainer

# Page configuration
st.set_page_config(
    page_title="LendSafe - Loan Decision Explainer",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS - Dark Theme
st.markdown("""
    <style>
    /* Global dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Main content area */
    .main .block-container {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #64b5f6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #fafafa !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
    }
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }

    /* Explanation card - dark with light text */
    .metric-card {
        background-color: #1e2127;
        color: #fafafa !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        line-height: 1.8;
        border: 1px solid #333;
    }
    .metric-card p {
        color: #fafafa !important;
        margin: 0.5rem 0;
    }

    /* Decision colors */
    .approved {
        color: #66bb6a !important;
        font-weight: bold;
    }
    .denied {
        color: #ef5350 !important;
        font-weight: bold;
    }

    /* Form inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #444 !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #1976d2 !important;
        color: #fafafa !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #2196f3 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: #1e2127 !important;
        color: #fafafa !important;
    }

    /* Dividers */
    hr {
        border-color: #333 !important;
    }

    /* Markdown text */
    .stMarkdown {
        color: #fafafa !important;
    }

    /* Forms */
    [data-testid="stForm"] {
        background-color: #1a1d24;
        border: 1px solid #333;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_explainer():
    """Load the fine-tuned model (cached)"""
    import os
    from pathlib import Path

    adapter_path = Path("models/granite-finetuned")
    base_model_path = "ibm-granite/granite-4.0-h-350m"

    # Check if we need to download model from Hugging Face
    # Try multiple sources: secrets, env var, or hardcoded default for HF Spaces
    hf_repo = None
    try:
        hf_repo = st.secrets.get("HF_MODEL_REPO")
    except:
        pass

    if not hf_repo:
        hf_repo = os.getenv("HF_MODEL_REPO")

    # If still not found and we're on HF Spaces, use the model from same user
    if not hf_repo and os.path.exists("/app"):  # HF Spaces indicator
        hf_repo = "notatharva0699/lendsafe-granite"
        st.info("🤖 Using default model: notatharva0699/lendsafe-granite")

    # If adapter doesn't exist locally and HF repo is configured
    if not adapter_path.exists() and hf_repo:
        with st.spinner("📥 Downloading fine-tuned model from Hugging Face (first time only, ~2-3 min)..."):
            try:
                from huggingface_hub import snapshot_download
                import time

                adapter_path.mkdir(parents=True, exist_ok=True)

                # Download with progress tracking
                downloaded_path = snapshot_download(
                    repo_id=hf_repo,
                    local_dir=str(adapter_path),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

                # Wait a moment for files to be written
                time.sleep(2)

                # List what was actually downloaded (for debugging)
                downloaded_files = list(adapter_path.glob("*"))
                logger.info(f"Downloaded {len(downloaded_files)} files to {adapter_path}")

                # Verify download succeeded by checking for key files
                required_files = ["adapter_model.safetensors", "adapter_config.json"]
                missing_files = [f for f in required_files if not (adapter_path / f).exists()]

                if not missing_files:
                    st.success(f"✅ Model downloaded successfully! ({len(downloaded_files)} files)")
                else:
                    raise FileNotFoundError(f"Missing files after download: {missing_files}")

            except Exception as e:
                st.error(f"❌ Error downloading model from Hugging Face: {e}")
                st.info("Falling back to base model (no fine-tuning)")
                # Clean up partial download
                import shutil
                if adapter_path.exists():
                    shutil.rmtree(adapter_path, ignore_errors=True)
                adapter_path = None
    elif not adapter_path.exists():
        st.warning("⚠️ Fine-tuned model not found. Using base model only.")
        st.info("To use the fine-tuned model, set HF_MODEL_REPO in Streamlit secrets.")
        adapter_path = None

    # Verify adapter_path has required files if it exists
    if adapter_path and adapter_path.exists():
        required_files = ["adapter_model.safetensors", "adapter_config.json"]
        missing_files = [f for f in required_files if not (adapter_path / f).exists()]

        if missing_files:
            st.warning(f"⚠️ Adapter files incomplete. Missing: {missing_files}. Using base model only.")
            adapter_path = None

    # Load model
    with st.spinner("Loading AI model... (this may take 30-60 seconds on first run)"):
        try:
            explainer = GraniteLoanExplainer(
                base_model_path=base_model_path,
                adapter_path=str(adapter_path) if adapter_path else None
            )
            return explainer
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("Please check the model files or try redeploying the app.")
            return None


@st.cache_resource
def load_risk_model():
    """Load the risk scoring model (cached)"""
    risk_model_path = Path("models/risk_model/loan_risk_model.pkl")
    if risk_model_path.exists():
        with open(risk_model_path, "rb") as f:
            model = pickle.load(f)
        return model
    return None


def calculate_risk_score(loan_data, risk_model):
    """Calculate risk score using the trained model"""
    if risk_model is None:
        return None

    # Extract features in the order expected by the model
    features = [
        loan_data['credit_score'],
        loan_data['annual_income'],
        loan_data['loan_amount'],
        loan_data['dti'],
        loan_data['revol_util'],
        loan_data['emp_length'],
        loan_data['total_acc'],
        loan_data['inq_last_6mths'],
        1 if loan_data['home_ownership'] == 'OWN' else 0,  # home_own
        1 if loan_data['home_ownership'] == 'RENT' else 0,  # home_rent
    ]

    # Predict probability of default
    df = pd.DataFrame([features], columns=[
        'credit_score', 'annual_inc', 'loan_amnt', 'dti', 'revol_util',
        'emp_length', 'total_acc', 'inq_last_6mths', 'home_own', 'home_rent'
    ])

    try:
        risk_prob = risk_model.predict_proba(df)[0][1]  # Probability of default
        return risk_prob * 100  # Convert to percentage
    except:
        return None


def main():
    # Header
    st.markdown('<div class="main-header">🏦 LendSafe</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Loan Decision Explainer | 100% Local & Private</div>',
        unsafe_allow_html=True
    )

    # Load models
    try:
        explainer = load_explainer()
        risk_model = load_risk_model()

        # Check if explainer loaded successfully
        if explainer is None:
            st.error("❌ Failed to load AI model. The app cannot generate explanations.")
            st.info("""
            **The model failed to load. This could be because:**

            1. **Streamlit Cloud limitations**: The base Granite model (~700MB) may be too large for the free tier
            2. **Missing dependencies**: Check that all required packages are installed
            3. **No HF_MODEL_REPO configured**: Add your Hugging Face model repo in Streamlit secrets

            **To fix this:**
            - Upload your model to Hugging Face: `hf upload notatharva0699/lendsafe-granite models/granite-finetuned/`
            - Add secret in Streamlit Cloud: `HF_MODEL_REPO = "notatharva0699/lendsafe-granite"`
            - Or try a smaller model variant
            """)
            return

        st.success("✅ AI models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Check the error logs for details. The app cannot continue without the AI model.")
        return

    # Sidebar - Info
    with st.sidebar:
        st.header("About LendSafe")
        st.markdown("""
        **LendSafe** uses fine-tuned IBM Granite AI to generate:
        - FCRA-compliant explanations
        - Adverse action notices
        - Human-readable reasoning

        **Privacy First:**
        - 100% local processing
        - No data leaves your machine
        - No API costs

        **Model:**
        - IBM Granite 4.0 H 350M
        - Fine-tuned with LoRA
        - <2GB RAM usage
        """)

        st.divider()

        st.header("Quick Examples")
        if st.button("Load Good Application"):
            st.session_state.example = "good"
        if st.button("Load Risky Application"):
            st.session_state.example = "risky"
        if st.button("Load Denied Application"):
            st.session_state.example = "denied"

    # Main content - two columns
    col1, col2 = st.columns([1, 1])

    # Default values
    defaults = {
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

    # Handle examples
    if 'example' in st.session_state:
        if st.session_state.example == "good":
            defaults.update({
                'credit_score': 750,
                'annual_income': 85000,
                'emp_length': 10,
                'home_ownership': 'OWN',
                'loan_amount': 20000,
                'dti': 12.0,
                'revol_util': 35.0,
                'inq_last_6mths': 0
            })
        elif st.session_state.example == "risky":
            defaults.update({
                'credit_score': 640,
                'annual_income': 42000,
                'emp_length': 2,
                'home_ownership': 'RENT',
                'loan_amount': 25000,
                'dti': 32.0,
                'revol_util': 85.0,
                'inq_last_6mths': 3
            })
        elif st.session_state.example == "denied":
            defaults.update({
                'credit_score': 580,
                'annual_income': 35000,
                'emp_length': 1,
                'home_ownership': 'RENT',
                'loan_amount': 30000,
                'dti': 42.0,
                'revol_util': 95.0,
                'inq_last_6mths': 5
            })
        del st.session_state.example

    # Left column - Input form
    with col1:
        st.header("📋 Loan Application")

        with st.form("loan_form"):
            st.subheader("Applicant Information")

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=defaults['credit_score'],
                help="FICO score (300-850)"
            )

            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                value=defaults['annual_income'],
                step=1000,
                help="Gross annual income"
            )

            emp_length = st.slider(
                "Employment Length (years)",
                min_value=0,
                max_value=40,
                value=defaults['emp_length']
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
                index=['RENT', 'OWN', 'MORTGAGE', 'OTHER'].index(defaults['home_ownership'])
            )

            st.subheader("Loan Details")

            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=100000,
                value=defaults['loan_amount'],
                step=1000
            )

            purpose = st.selectbox(
                "Loan Purpose",
                options=[
                    'debt_consolidation',
                    'credit_card',
                    'home_improvement',
                    'major_purchase',
                    'small_business',
                    'car',
                    'medical',
                    'moving',
                    'vacation',
                    'other'
                ],
                index=0
            )

            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=5.0,
                max_value=35.0,
                value=defaults['int_rate'],
                step=0.5
            )

            term = st.selectbox(
                "Loan Term (months)",
                options=[36, 60],
                index=0 if defaults['term'] == 36 else 1
            )

            st.subheader("Financial Metrics")

            dti = st.number_input(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=defaults['dti'],
                step=0.5,
                help="Monthly debt payments / Monthly income"
            )

            revol_util = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0,
                max_value=100.0,
                value=defaults['revol_util'],
                step=1.0,
                help="Credit card balance / Credit limit"
            )

            total_acc = st.number_input(
                "Total Credit Accounts",
                min_value=0,
                max_value=100,
                value=defaults['total_acc']
            )

            inq_last_6mths = st.number_input(
                "Credit Inquiries (last 6 months)",
                min_value=0,
                max_value=20,
                value=defaults['inq_last_6mths']
            )

            submitted = st.form_submit_button("🔍 Analyze Application", use_container_width=True)

    # Right column - Results
    with col2:
        st.header("📊 Decision & Explanation")

        if submitted:
            # Prepare loan data
            loan_data = {
                'credit_score': credit_score,
                'annual_income': annual_income,
                'emp_length': emp_length,
                'home_ownership': home_ownership,
                'loan_amount': loan_amount,
                'purpose': purpose,
                'int_rate': int_rate,
                'term': term,
                'dti': dti,
                'revol_util': revol_util,
                'total_acc': total_acc,
                'inq_last_6mths': inq_last_6mths
            }

            # Calculate risk score
            with st.spinner("Calculating risk score..."):
                risk_score = calculate_risk_score(loan_data, risk_model)

            # Determine decision
            if risk_score is not None:
                decision = "DENIED" if risk_score > 50 else "APPROVED"
            else:
                # Simple rule-based decision if model not available
                if credit_score < 620 or dti > 40 or revol_util > 90:
                    decision = "DENIED"
                else:
                    decision = "APPROVED"

            # Display risk score
            if risk_score is not None:
                st.metric(
                    label="Risk Score",
                    value=f"{risk_score:.1f}%",
                    delta="High Risk" if risk_score > 50 else "Low Risk",
                    delta_color="inverse"
                )

            # Display decision
            decision_class = "approved" if decision == "APPROVED" else "denied"
            st.markdown(
                f'<h2 class="{decision_class}">Decision: {decision}</h2>',
                unsafe_allow_html=True
            )

            st.divider()

            # Generate explanation
            with st.spinner("Generating AI explanation... (5-10 seconds)"):
                explanation = explainer.explain_decision(
                    loan_data,
                    decision,
                    risk_score
                )

            # Display explanation
            st.subheader("📝 Explanation")
            st.markdown(f"""
            <div class="metric-card">
            {explanation}
            </div>
            """, unsafe_allow_html=True)

            # Additional metrics
            st.divider()
            st.subheader("Key Metrics")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Credit Score", credit_score)
                st.metric("DTI Ratio", f"{dti:.1f}%")

            with metric_col2:
                st.metric("Income", f"${annual_income:,}")
                st.metric("Revolving Util", f"{revol_util:.1f}%")

            with metric_col3:
                st.metric("Loan Amount", f"${loan_amount:,}")
                st.metric("Employment", f"{emp_length} years")

        else:
            st.info("👈 Fill out the loan application form and click 'Analyze Application'")
            st.markdown("""
            ### What you'll get:
            - **Risk Score**: ML-powered default probability
            - **Decision**: APPROVED or DENIED
            - **AI Explanation**: Clear reasoning for the decision
            - **FCRA Compliance**: Regulatory-compliant language

            ### Try the examples:
            Click the example buttons in the sidebar to load pre-filled applications!
            """)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with IBM Granite 4.0 H 350M | 100% Local Processing | No Data Leaves Your Device<br>
    <a href="https://github.com/yourusername/lendsafe">GitHub</a> |
    <a href="https://huggingface.co/ibm-granite">IBM Granite</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
