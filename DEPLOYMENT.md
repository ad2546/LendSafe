# 🚀 LendSafe Deployment Guide

Complete guide for deploying LendSafe to GitHub and various platforms.

---

## 📋 Pre-Deployment Checklist

### Required Files
- [x] README.md (updated for GitHub)
- [x] requirements.txt (dependencies)
- [x] .gitignore (excludes large files)
- [x] LICENSE (MIT recommended)
- [x] app.py (main application)
- [x] src/ (source code)
- [x] scripts/ (utility scripts)

### Optional but Recommended
- [ ] docs/ (screenshots, diagrams)
- [ ] tests/ (unit tests)
- [ ] .github/workflows/ (CI/CD)
- [ ] Dockerfile (containerization)

---

## 🎯 Deployment Options

### Option 1: GitHub Repository (Recommended First Step)

Perfect for:
- Portfolio showcase
- Code sharing
- Collaboration
- Version control

**Note**: Models will NOT be included (too large for git)

### Option 2: Streamlit Community Cloud (Free Hosting)

Perfect for:
- Live demos
- Public sharing
- Quick deployment

**Limitations**:
- 1GB storage limit (model must be downloaded at runtime)
- 1GB RAM limit (may need model optimization)

### Option 3: Docker + Cloud VPS

Perfect for:
- Production deployments
- Full control
- Enterprise use

**Cost**: $5-20/month (DigitalOcean, AWS EC2, etc.)

---

## 📦 Option 1: Deploy to GitHub

### Step 1: Prepare Repository

```bash
# Navigate to project directory
cd /Users/atharvadeshmukh/LendSafe

# Check current status
git status

# Initialize git (if not already done)
git init

# Add all files (respects .gitignore)
git add .

# Create first commit
git commit -m "Initial commit: LendSafe AI Loan Explainer

- Fine-tuned IBM Granite 350M with LoRA
- Streamlit web interface with dark theme
- FCRA-compliant explanation generation
- Integration tests and documentation
- <2GB RAM, 5-10s inference time"
```

### Step 2: Create GitHub Repository

**Via GitHub Website:**
1. Go to https://github.com/new
2. Repository name: `LendSafe` (or `lendsafe`)
3. Description: "AI-powered loan decision explainer with FCRA-compliant explanations. Built with fine-tuned IBM Granite 350M + Streamlit."
4. Choose: **Public** (for portfolio) or **Private**
5. **DO NOT** initialize with README (we have our own)
6. Click "Create repository"

**Via GitHub CLI (if installed):**
```bash
gh repo create LendSafe --public --description "AI-powered FCRA-compliant loan explainer"
```

### Step 3: Connect and Push

```bash
# Add remote repository (replace with your username)
git remote add origin https://github.com/YOUR_USERNAME/LendSafe.git

# Rename branch to main (GitHub standard)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 4: Post-Deployment Setup

**Add Topics (GitHub Website):**
- Click "Add topics" on repo page
- Add: `ai`, `machine-learning`, `fintech`, `explainable-ai`, `ibm-granite`, `streamlit`, `lending`, `fcra-compliance`, `privacy-first`

**Create Releases:**
```bash
# Tag the first release
git tag -a v1.0.0 -m "LendSafe v1.0.0 - Initial Release"
git push origin v1.0.0
```

**Update Repository Settings:**
1. Settings → General → Features
2. Enable: Issues, Discussions (optional)
3. Settings → Pages (for GitHub Pages docs - optional)

---

## ☁️ Option 2: Streamlit Community Cloud

### Prerequisites
- GitHub repository (from Option 1)
- Streamlit account (free)

### Step 1: Optimize for Streamlit Cloud

**Create `.streamlit/config.toml`:**
```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#1976d2"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d24"
textColor = "#fafafa"
```

**Create `packages.txt` (system dependencies):**
```
build-essential
```

**Update `requirements.txt` (add version pins):**
```
torch==2.1.0
transformers==4.35.0
streamlit==1.29.0
# ... rest of dependencies
```

### Step 2: Deploy

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `YOUR_USERNAME/LendSafe`
   - Branch: `main`
   - Main file path: `app.py`
5. Advanced settings:
   - Python version: `3.12`
6. Click "Deploy!"

### Step 3: Handle Large Models

**Issue**: Fine-tuned model >1GB won't fit

**Solution 1: Download at Runtime**
Add to `app.py` (before loading model):
```python
import os
from huggingface_hub import snapshot_download

MODEL_PATH = "models/granite-finetuned"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only, ~5 min)..."):
            snapshot_download(
                repo_id="YOUR_HF_USERNAME/lendsafe-granite",
                local_dir=MODEL_PATH
            )
    return MODEL_PATH
```

**Solution 2: Use Hugging Face Model Hub**
1. Upload model to Hugging Face
2. Load directly from hub:
```python
model = PeftModel.from_pretrained(
    base_model,
    "YOUR_HF_USERNAME/lendsafe-granite"
)
```

**Solution 3: Use Smaller Model**
- Use base Granite 350M without fine-tuning
- Or fine-tune with even lower rank (r=4)

---

## 🐳 Option 3: Docker Deployment

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Test Locally

```bash
# Build image
docker build -t lendsafe:latest .

# Run container
docker run -p 8501:8501 lendsafe:latest

# Test at http://localhost:8501
```

### Step 3: Deploy to Cloud

**DigitalOcean App Platform:**
```bash
# Install doctl
brew install doctl

# Authenticate
doctl auth init

# Deploy
doctl apps create --spec .do/app.yaml
```

**AWS EC2:**
```bash
# SSH to instance
ssh ubuntu@YOUR_EC2_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Pull and run image
docker pull YOUR_DOCKERHUB_USER/lendsafe:latest
docker run -d -p 8501:8501 YOUR_DOCKERHUB_USER/lendsafe:latest
```

---

## 🔐 Environment Variables

If you add API keys or secrets later:

**Create `.env` file (NOT committed to git):**
```bash
HUGGINGFACE_TOKEN=your_token_here
DATABASE_URL=your_db_url
```

**Load in app.py:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
```

**For Streamlit Cloud:**
- Settings → Secrets
- Add secrets in TOML format

---

## 📊 Analytics (Optional)

### Add Google Analytics

Add to `app.py`:
```python
# At the top, after imports
GA_TRACKING_ID = "G-XXXXXXXXXX"

st.markdown(f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_TRACKING_ID}');
    </script>
""", unsafe_allow_html=True)
```

---

## 🚀 Recommended Deployment Flow

### For Portfolio/Demo:
1. **GitHub** (code + documentation)
2. **Streamlit Cloud** (live demo)
3. **LinkedIn/Twitter** (share links)

### For Production:
1. **GitHub** (code + CI/CD)
2. **Docker** (containerization)
3. **AWS/GCP/Azure** (scalable hosting)
4. **Monitoring** (Datadog, New Relic)

---

## 📝 Post-Deployment Checklist

### GitHub
- [ ] Repository is public (or private if needed)
- [ ] README.md is clear and complete
- [ ] Topics are added
- [ ] Release v1.0.0 is tagged
- [ ] Issues are enabled

### Documentation
- [ ] README has installation instructions
- [ ] Screenshots/GIFs are added
- [ ] Architecture diagram is included
- [ ] Contributing guide is present

### Code Quality
- [ ] .gitignore excludes large files
- [ ] Requirements.txt is up to date
- [ ] Code is formatted (black, isort)
- [ ] Tests are passing

### Demo
- [ ] Streamlit app is deployed
- [ ] URL is added to GitHub README
- [ ] App loads in <60 seconds
- [ ] Examples work correctly

---

## 🐛 Common Issues

### Issue: Git push fails (file too large)
```
remote: error: File models/granite-finetuned/adapter_model.safetensors is 644.00 MB
```

**Solution:**
```bash
# Check .gitignore includes large files
cat .gitignore | grep safetensors

# If file was already committed, remove from git
git rm --cached models/granite-finetuned/*.safetensors
git commit -m "Remove large model files"
git push
```

### Issue: Streamlit Cloud out of memory
```
MemoryError: Unable to allocate X MB
```

**Solution:**
- Use smaller model (Granite 350M instead of 3B)
- Reduce batch size
- Load model lazily with @st.cache_resource

### Issue: Docker build takes too long
```
Building wheel for torch...
```

**Solution:**
Use pre-built wheels:
```dockerfile
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 📞 Support

If you encounter issues:
1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. Search [GitHub Issues](https://github.com/YOUR_USERNAME/LendSafe/issues)
3. Create new issue with:
   - Error message
   - Steps to reproduce
   - Environment (OS, Python version)

---

**Ready to Deploy?** Start with GitHub (Option 1)!

```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/LendSafe.git
git push -u origin main
```

**Then share your repo URL!** 🎉
