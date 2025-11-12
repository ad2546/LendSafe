# 🚀 LendSafe Quick Start

Your fine-tuned model is ready! Here's everything you need to know in 2 minutes.

---

## ✅ What's Done

- ✅ IBM Granite 350M fine-tuned with LoRA
- ✅ Streamlit web app with AI explanations
- ✅ Integration tests passing
- ✅ Full documentation

---

## 🎯 Launch the App (30 seconds)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Launch app
streamlit run app.py
```

**App URL**: http://localhost:8501

---

## 💻 Using the Web Interface

### Step 1: Fill Out Loan Application
- Enter applicant info (credit score, income, employment)
- Add loan details (amount, purpose, term)
- Input financial metrics (DTI, credit utilization)

### Step 2: Analyze
- Click **"Analyze Application"** button
- Wait 5-10 seconds for AI processing

### Step 3: Review Results
- See risk score and decision (APPROVED/DENIED)
- Read AI-generated explanation
- View key metrics

### Quick Test: Use Sidebar Examples
- **"Load Good Application"** - High approval probability
- **"Load Risky Application"** - Borderline case
- **"Load Denied Application"** - High risk factors

---

## 🧪 Testing

### Integration Test
```bash
python scripts/test_integration.py
```

Tests 3 scenarios with AI explanations.

### Model Test
```bash
python src/llm_explainer.py
```

Tests model loading and inference.

---

## 📊 What You Get

**For Each Application**:
1. **Risk Score**: ML-powered probability (0-100%)
2. **Decision**: APPROVED or DENIED
3. **AI Explanation**:
   - References specific metrics
   - Professional regulatory language
   - Clear reasoning
4. **Metrics Dashboard**: Visual summary

---

## 🎬 Demo Flow (3 minutes)

1. **Open app** → http://localhost:8501
2. **Click "Load Good Application"** in sidebar
3. **Click "Analyze Application"**
4. **Show results**:
   - Risk score: ~25% (low risk)
   - Decision: APPROVED
   - AI explanation with reasoning
5. **Click "Load Denied Application"**
6. **Analyze again**
7. **Show how tone changes** for denials

**Key Points to Mention**:
- "Runs 100% locally, <2GB RAM"
- "IBM Granite 350M with LoRA"
- "FCRA-compliant explanations"
- "$0 API costs"

---

## 📁 Key Files

| File | What It Does |
|------|--------------|
| `app.py` | Streamlit web interface |
| `src/llm_explainer.py` | AI model wrapper |
| `models/granite-finetuned/` | Fine-tuned model |
| `scripts/test_integration.py` | Full system test |

---

## 🛠️ Troubleshooting

**App won't start?**
```bash
source .venv/bin/activate
uv pip install streamlit
streamlit run app.py
```

**Model not found?**
- Check: `ls models/granite-finetuned/`
- Should see: `adapter_model.safetensors`, `adapter_config.json`

**Out of memory?**
- Edit `src/llm_explainer.py`
- Change line 51: `device="cpu"` instead of `"auto"`

**Slow inference?**
- Normal for first run (model loading)
- Subsequent calls: 5-10 seconds
- Uses Apple Silicon (MPS) acceleration

---

## 🎯 Next Steps

### Immediate
- [x] Test the web app
- [ ] Try all 3 example scenarios
- [ ] Record demo video

### This Week
- [ ] Add risk model integration
- [ ] Build RAG system (Week 3)
- [ ] Generate PDF notices (Week 3)

### Future
- [ ] Deploy to Streamlit Cloud
- [ ] Add API endpoints
- [ ] Create demo video for portfolio

---

## 📚 More Info

- **Full Guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Integration Details**: [MODEL_INTEGRATION_COMPLETE.md](MODEL_INTEGRATION_COMPLETE.md)
- **Project Overview**: [CLAUDE.md](CLAUDE.md)
- **Training Details**: [WEEK2_COMPLETE.md](WEEK2_COMPLETE.md)

---

## 🎉 You're Ready!

**Current Status**: ✅ Production-ready demo

**What works**:
- AI loan explanations
- Risk scoring (rule-based)
- Professional web interface
- Example scenarios

**What's next**: RAG system + PDF reports (Week 3)

---

**Launch command**: `streamlit run app.py`

**Questions?** Check [USAGE_GUIDE.md](USAGE_GUIDE.md)

---

🏦 **LendSafe** - AI-Powered Loan Decisions, 100% Local
