# LendSafe: Local Loan Decision Explainer

## 🎯 Project Overview

**LendSafe** is an AI-powered loan decision explanation system that runs entirely locally, providing regulatory-compliant adverse action notices and human-readable explanations for loan approvals/rejections. Built to address the critical need for explainable AI in financial services while maintaining data privacy.

### Why This Project Matters
- **Regulatory Compliance**: Addresses FCRA/ECOA requirements for explaining credit decisions
- **Privacy-First**: 100% local processing - no data leaves the institution
- **Cost-Effective**: No API costs, runs on commodity hardware (M2 MacBook Air)
- **Explainable AI**: Every decision backed by clear reasoning and regulatory citations

### Target Use Cases
- Credit unions and community banks needing affordable compliance solutions
- Fintech lenders building transparent lending platforms
- Model risk management teams requiring explainable models
- Financial institutions migrating to AI-powered decisioning

---

## 🏗️ Architecture

### System Components
```
┌─────────────────────────────────────────────────────────────┐
│                    LendSafe Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │   Streamlit  │ ◄─────► │  Loan Decision  │              │
│  │   Frontend   │         │     Engine      │              │
│  └──────────────┘         └────────┬────────┘              │
│                                     │                        │
│                                     ▼                        │
│                          ┌──────────────────┐               │
│                          │  IBM Granite     │               │
│                          │  4.0 H 350M      │               │
│                          │  (Fine-tuned)    │               │
│                          └────────┬─────────┘               │
│                                   │                          │
│                    ┌──────────────┼──────────────┐          │
│                    ▼              ▼              ▼          │
│          ┌─────────────┐  ┌──────────┐  ┌──────────────┐  │
│          │  Risk Score │  │   RAG    │  │  Adverse     │  │
│          │  Calculator │  │  System  │  │  Action      │  │
│          │             │  │          │  │  Generator   │  │
│          └─────────────┘  └────┬─────┘  └──────────────┘  │
│                                 │                            │
│                                 ▼                            │
│                        ┌─────────────────┐                  │
│                        │   ChromaDB      │                  │
│                        │ (Regulatory     │                  │
│                        │  Knowledge)     │                  │
│                        └─────────────────┘                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core Components (All Free & Open Source)

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **LLM** | IBM Granite 4.0 H 350M | Tiny (350M params), enterprise-focused, <2GB RAM |
| **Fine-tuning** | Hugging Face PEFT (LoRA) | Parameter-efficient, Mac-friendly |
| **Vector DB** | ChromaDB | Local storage, no cloud dependency |
| **Framework** | LangChain | Standard RAG implementation |
| **Frontend** | Streamlit | Rapid prototyping, free deployment |
| **Risk Model** | Scikit-learn / XGBoost | Industry-standard, interpretable |
| **Package Manager** | uv | 10-100x faster than pip, Rust-based |
| **Deployment** | Streamlit Cloud | Free tier, easy sharing |

### Why IBM Granite 4.0 H 350M?

1. **Tiny but Mighty**: 350M parameters vs 3B+ in other models
   - <2GB RAM on M2 MacBook Air
   - Inference: <500ms per request
   - No GPU needed

2. **Enterprise-Grade**: Built by IBM Research
   - Pre-trained on financial/business documents
   - Better regulatory language understanding
   - Apache 2.0 license

3. **Finance-Optimized**: Handles structured data naturally

### Why uv Package Manager?

- **Speed**: 10-100x faster than pip
- **Reliability**: Rust-based, deterministic
- **Modern**: Drop-in pip replacement
- **Zero-config**: Works out of the box

---

## 📋 4-Week Development Roadmap

### Week 1: Foundation & Data Pipeline
**Deliverables:**
- [ ] Development environment configured with uv
- [ ] Lending Club dataset cleaned
- [ ] Baseline XGBoost risk model trained
- [ ] IBM Granite 4.0 H 350M tested
- [ ] 100 synthetic loan explanations

**Setup:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv venv
source .venv/bin/activate

# Install dependencies (blazing fast!)
uv pip install torch transformers accelerate peft
uv pip install pandas scikit-learn xgboost
uv pip install langchain chromadb streamlit
```

### Week 2: LLM Fine-tuning
**Deliverables:**
- [ ] 1000+ training examples
- [ ] Fine-tuned Granite with LoRA
- [ ] Evaluation metrics (ROUGE, BERTScore)

**Expected Performance on M2:**
- Memory: ~6-8GB RAM
- Training: 2-3 hours for 1000 examples
- Inference: <500ms per application

### Week 3: RAG System & Compliance
**Deliverables:**
- [ ] ChromaDB with FCRA/ECOA regulations
- [ ] RAG pipeline with citations
- [ ] Adverse action notice generator (PDF)

### Week 4: Integration & Demo
**Deliverables:**
- [ ] Streamlit dashboard
- [ ] Demo video (3-5 minutes)
- [ ] GitHub repo + README
- [ ] Medium article draft

---

## 🚀 Quick Start
```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone https://github.com/yourusername/lendsafe.git
cd lendsafe
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Download Granite
python scripts/download_model.py

# 5. Run app
streamlit run app.py
```

---

## 💻 Key Code Examples

### Fine-tuning Script
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "ibm-granite/granite-4.0-h-350m"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Apply LoRA (only 0.15% params trained!)
model = get_peft_model(model, lora_config)
```

### Inference
```python
class GraniteLoanExplainer:
    def explain_decision(self, loan_data, decision):
        prompt = f"""### Instruction:
Explain why this loan was {decision}.

### Input:
Credit Score: {loan_data['credit_score']}
DTI: {loan_data['dti_ratio']}%
Amount: ${loan_data['loan_amount']:,}

### Response:
"""
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0])
```

---

## 🎯 Pitch Strategy

### Elevator Pitch
*"LendSafe uses IBM's enterprise-grade Granite AI to generate FCRA-compliant loan explanations entirely on your infrastructure. Zero API costs, complete privacy, runs on a laptop. Built for credit unions who need compliance without cloud dependency."*

### Target Companies
1. **Navy Federal Credit Union** - Follow up on your interview!
2. **Upstart, SoFi, Affirm** - AI-native lenders
3. **Regional banks** - Model risk teams
4. **Credit bureaus** - White-label solution

### Demo Script (5 min)
1. **Problem** (30s): $X billion in regulatory fines
2. **Solution** (30s): Show architecture
3. **Live Demo** (3 min): Upload → Explain → PDF in <3 seconds
4. **Tech** (30s): "IBM Granite + M2 MacBook + <2GB RAM"
5. **Value** (30s): $0/inference vs $10K+/month APIs
6. **CTA** (30s): "Built in 4 weeks with $0 budget"

---

## 📊 Success Metrics

### Technical
- ✅ <2GB RAM usage
- ✅ <3 seconds inference
- ✅ 90%+ ROUGE-L score
- ✅ 100% FCRA compliance

### Business
- 🎯 5+ company pilots
- 🎯 2+ interviews from demo
- 🎯 1000+ GitHub stars

---

## 📦 Project Structure
```
lendsafe/
├── .venv/                      # uv virtual environment
├── pyproject.toml              # Modern Python config
├── requirements.txt
├── app.py                      # Streamlit app
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
│   ├── risk_model/
│   └── granite-lendsafe/       # Fine-tuned weights
├── chromadb/                   # Vector DB
├── scripts/
│   ├── download_model.py
│   ├── finetune_granite.py
│   └── build_chroma_db.py
└── src/
    ├── llm_explainer.py
    ├── rag_pipeline.py
    └── notice_generator.py
```

---

## 🔐 Regulatory Compliance

### FCRA Requirements ✅
- Section 615: Adverse action notices
- Section 609: Disclosure of information
- Section 623: Accuracy and completeness

### ECOA Requirements ✅
- Regulation B: Notification
- 12 CFR 1002.9: Specific reasons
- Plain language: 8th-grade level

---

## 🚧 Future Enhancements

**Phase 2:**
- Multi-language support
- Voice explanations
- Counterfactual analysis

**Phase 3:**
- API access
- Multi-tenant support
- Kubernetes deployment

**Phase 4:**
- Drift detection
- Fairness metrics
- Causal inference

---

## 📚 Resources

- [IBM Granite Models](https://huggingface.co/ibm-granite)
- [uv Documentation](https://github.com/astral-sh/uv)
- [PEFT Guide](https://huggingface.co/docs/peft)
- [FCRA Text](https://www.ftc.gov/enforcement/statutes/fair-credit-reporting-act)

---

## ⚡ Quick Start Checklist

**Week 1:**
- [ ] Install uv
- [ ] Download Lending Club data
- [ ] Test Granite model

**Week 2:**
- [ ] Generate training data
- [ ] Fine-tune with LoRA

**Week 3:**
- [ ] Build ChromaDB
- [ ] Implement RAG

**Week 4:**
- [ ] Streamlit app
- [ ] Demo video
- [ ] GitHub launch

---

**Built with:**
- Zero dollars 💰
- M2 MacBook Air 💻
- 4 weeks ⏰
- IBM Granite 🤖

**Perfect for:**
- Navy Federal follow-up interview
- Fintech job applications
- Portfolio differentiation
- Learning regulatory ML

---

*Ready to build LendSafe and land that job? Let's go! 🚀*

**File saved to:** `/Users/atharvadeshmukh/Desktop/lendsafe_claude.md`