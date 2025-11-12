# 🏦 LendSafe - AI-Powered Loan Decision Explainer

> **FCRA-compliant loan explanations powered by fine-tuned IBM Granite 4.0 H 350M**
> 100% local processing • Zero API costs • Privacy-first architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite%204.0-052FAD.svg)](https://huggingface.co/ibm-granite)

---

## 🎯 Overview

**LendSafe** is an AI-powered loan decision explanation system that generates regulatory-compliant adverse action notices and human-readable explanations for loan approvals and rejections. Built to address the critical need for explainable AI in financial services while maintaining complete data privacy.

### Key Features

- ✅ **FCRA/ECOA Compliant**: Generates explanations meeting regulatory requirements
- 🔒 **100% Local Processing**: No data leaves your machine
- 💰 **Zero API Costs**: Runs entirely offline on commodity hardware
- ⚡ **Fast Inference**: <10 seconds per loan application
- 🪶 **Lightweight**: <2GB RAM on M2 MacBook Air
- 🏢 **Enterprise-Grade**: Built on IBM Granite 4.0 H 350M
- 📊 **Interactive UI**: Professional Streamlit interface

---

## 🎬 Demo

```bash
# Launch the app
streamlit run app.py
```

![LendSafe Demo](docs/demo-screenshot.png)

### Example Output

**Input**: Credit score 680, $55K income, $15K loan request
**Decision**: APPROVED
**AI Explanation**:
> "Thank you for providing the information requested in your application. Based on your detailed account and that you have demonstrated capability to manage monthly payments within our acceptable range (25% of income), we have granted approval to proceed with approving your loan.
>
> The specific criteria accepted include:
> 1. Your credit score of 680 places you at an excellent standing.
> 2. Your annual income of $55,000 is reasonable given current market conditions.
> 3. Your five-year employment history indicates stable capacity to repay loans comfortably..."

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│       Streamlit Web Interface           │
│           (app.py)                      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      LLM Explainer Module               │
│    (src/llm_explainer.py)               │
│                                          │
│  ┌────────────┐      ┌──────────┐      │
│  │  Granite   │◄─────┤   LoRA   │      │
│  │   350M     │      │ Adapters │      │
│  └────────────┘      └──────────┘      │
│                                          │
│  Device: MPS/CUDA/CPU                   │
│  Memory: <2GB RAM                       │
│  Speed:  5-10 seconds                   │
└─────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | IBM Granite 4.0 H 350M | Base language model (350M params) |
| **Fine-tuning** | PEFT/LoRA (r=8) | Parameter-efficient training (644KB adapters) |
| **Frontend** | Streamlit | Interactive web interface |
| **ML Framework** | PyTorch + Transformers | Model inference |
| **Risk Scoring** | Scikit-learn/XGBoost | Credit risk assessment |
| **Package Manager** | uv | Fast dependency management |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 8GB RAM minimum (16GB recommended)
- macOS (Apple Silicon), Linux, or Windows

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/lendsafe.git
cd lendsafe

# 2. Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Download the base model (optional - will auto-download)
python scripts/download_model.py

# 6. Run the app
streamlit run app.py
```

Open your browser to **http://localhost:8501**

---

## 📦 Getting the Fine-Tuned Model

The fine-tuned model weights are **not included** in this repository due to size (>1GB). You have two options:

### Option 1: Download Pre-trained Model (Recommended)

```bash
# Download from Hugging Face or Google Drive
# [Link will be provided after public release]
wget <model-url> -O models/granite-finetuned.zip
unzip models/granite-finetuned.zip -d models/
```

### Option 2: Train Your Own Model

```bash
# 1. Generate training data
python scripts/generate_synthetic_explanations.py

# 2. Fine-tune the model (requires GPU, or use Google Colab)
python scripts/finetune_granite.py

# Or use the provided Colab notebook
# Upload LendSafe_Finetune_Colab.ipynb to Google Colab
```

See [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md) for detailed training guide.

---

## 💻 Usage

### Web Interface

1. **Launch the app**: `streamlit run app.py`
2. **Fill out loan application** with applicant details
3. **Click "Analyze Application"** to generate explanation
4. **Review decision** and AI-generated reasoning

### Programmatic Usage

```python
from src.llm_explainer import GraniteLoanExplainer

# Initialize explainer
explainer = GraniteLoanExplainer(
    base_model_path="ibm-granite/granite-4.0-h-350m",
    adapter_path="models/granite-finetuned"
)

# Prepare loan data
loan_data = {
    'credit_score': 680,
    'annual_income': 55000,
    'loan_amount': 15000,
    'dti': 18.5,
    'revol_util': 65.0,
    'emp_length': 5,
    'home_ownership': 'RENT',
    # ... other fields
}

# Generate explanation
explanation = explainer.explain_decision(
    loan_data,
    decision="APPROVED",  # or "DENIED"
    risk_score=42.5       # optional risk score
)

print(explanation)
```

---

## 🧪 Testing

```bash
# Run integration tests
python scripts/test_integration.py

# Test model directly
python src/llm_explainer.py

# Run all tests
pytest tests/
```

---

## 📊 Performance

### Model Specifications

| Metric | Value |
|--------|-------|
| Base Model | IBM Granite 4.0 H 350M |
| Parameters | 350M (base) + 0.9M (LoRA) |
| Model Size | 700MB (base) + 644KB (adapters) |
| Memory Usage | <2GB RAM |
| Inference Time | 5-10 seconds/application |
| Device Support | CPU, Apple Silicon (MPS), CUDA |

### Benchmarks (M2 MacBook Air)

- **First Load**: 30-60 seconds (model loading)
- **Subsequent Inferences**: 5-10 seconds
- **Throughput**: ~6-12 applications/minute
- **Memory**: 1.8GB peak usage

---

## 🎓 Why This Project?

### Business Problem

Financial institutions face regulatory requirements to explain AI-driven credit decisions under FCRA and ECOA. Current solutions:
- Cost $10K+/month in API fees
- Send sensitive data to third parties
- Lack customization for specific use cases

### LendSafe Solution

- **$0 operational cost** (no APIs)
- **Complete data privacy** (100% local)
- **Customizable** (fine-tune on your data)
- **Regulatory compliant** (FCRA/ECOA language)

### Target Users

- Credit unions and community banks
- Fintech lenders building transparent platforms
- Model risk management teams
- Financial institutions migrating to AI decisioning

---

## 🗺️ Roadmap

### Current (v1.0)
- ✅ Fine-tuned IBM Granite 350M model
- ✅ Streamlit web interface
- ✅ Basic loan decision explanations
- ✅ Risk scoring integration

### Coming Soon (v1.1)
- [ ] RAG system with FCRA/ECOA regulations
- [ ] PDF adverse action notice generator
- [ ] Regulatory citation system
- [ ] Batch processing interface

### Future (v2.0)
- [ ] Multi-language support
- [ ] FastAPI REST endpoints
- [ ] Docker containerization
- [ ] Fairness/bias monitoring
- [ ] Counterfactual explanations
- [ ] Voice-based explanations

---

## 📚 Documentation

- [Quick Start Guide](QUICK_START.md) - Get up and running in 2 minutes
- [Usage Guide](USAGE_GUIDE.md) - Comprehensive user manual
- [Model Integration](MODEL_INTEGRATION_COMPLETE.md) - Technical details
- [Colab Training](COLAB_INSTRUCTIONS.md) - Fine-tuning guide
- [Project Overview](CLAUDE.md) - Full project documentation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install dev dependencies
uv pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IBM Research** for the Granite 4.0 H model series
- **Hugging Face** for PEFT/LoRA implementation
- **Streamlit** for the web framework
- **Lending Club** for the public loan dataset

---

## 📞 Contact

**Atharva Deshmukh**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/lendsafe&type=Date)](https://star-history.com/#yourusername/lendsafe&Date)

---

## 📈 Project Stats

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/lendsafe)
![GitHub issues](https://img.shields.io/github/issues/yourusername/lendsafe)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/lendsafe)
![GitHub stars](https://img.shields.io/github/stars/yourusername/lendsafe?style=social)

---

**Built with 💙 using IBM Granite AI**
**Privacy-First • Zero-Cost • Regulatory-Compliant**

---

## 🔗 Related Projects

- [IBM Granite Models](https://huggingface.co/ibm-granite)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Fair Credit Reporting Act](https://www.ftc.gov/enforcement/statutes/fair-credit-reporting-act)

---

**Made for banks and lenders who value privacy and compliance** 🏦
