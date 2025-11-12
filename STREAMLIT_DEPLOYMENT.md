# 🚀 Deploy LendSafe to Streamlit Cloud

Complete guide to deploy your app to Streamlit Community Cloud (FREE).

---

## 🎯 Overview

**Streamlit Cloud** offers:
- ✅ Free hosting for public GitHub repos
- ✅ Automatic deployments on git push
- ✅ HTTPS domain (yourapp.streamlit.app)
- ✅ Easy to setup (5 minutes)

**Limitations**:
- 1GB storage (model files must be downloaded at runtime)
- 1GB RAM (our app uses <2GB, should work)
- CPU only (no GPU)

---

## 📋 Prerequisites

- ✅ GitHub repository (already done!)
- ✅ Streamlit Cloud account (free)
- ✅ Model hosted somewhere accessible (we'll handle this)

---

## 🚀 Step 1: Prepare for Streamlit Cloud

### Option A: Download Model at Runtime (Recommended)

We need to modify `app.py` to download the model on first run since the model files are too large for git.

Let me create a helper script:

**Create `utils/model_downloader.py`:**
```python
"""
Model downloader for Streamlit Cloud deployment
Downloads fine-tuned model from external source on first run
"""

import os
import streamlit as st
from pathlib import Path
import gdown  # Google Drive downloader

MODEL_DIR = Path("models/granite-finetuned")
MODEL_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "chat_template.jinja"
]

def check_model_exists():
    """Check if model is already downloaded"""
    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)

@st.cache_resource
def download_model_from_gdrive(gdrive_url):
    """
    Download model from Google Drive

    Args:
        gdrive_url: Google Drive sharing link or file ID
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with st.spinner("📥 Downloading model (first time only, ~2-3 min)..."):
        try:
            # Download zip file
            output_zip = "models/model.zip"
            gdown.download(gdrive_url, output_zip, quiet=False, fuzzy=True)

            # Extract
            import zipfile
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall("models/")

            # Cleanup
            os.remove(output_zip)

            st.success("✅ Model downloaded successfully!")
            return True

        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.info("Please check the Google Drive link or contact support.")
            return False

@st.cache_resource
def download_model_from_hf(hf_repo_id):
    """
    Download model from Hugging Face Hub

    Args:
        hf_repo_id: Hugging Face repository ID (e.g., "username/model-name")
    """
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with st.spinner("📥 Downloading model from Hugging Face (first time only)..."):
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False
            )
            st.success("✅ Model downloaded successfully!")
            return True

        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return False

def ensure_model_available():
    """
    Ensure model is available, download if needed
    Returns True if model is ready, False otherwise
    """
    if check_model_exists():
        return True

    st.warning("⚠️ Model not found. Downloading...")

    # Option 1: Try Hugging Face (if you uploaded there)
    HF_REPO_ID = os.getenv("HF_MODEL_REPO")  # Set in Streamlit secrets
    if HF_REPO_ID:
        return download_model_from_hf(HF_REPO_ID)

    # Option 2: Try Google Drive
    GDRIVE_URL = os.getenv("GDRIVE_MODEL_URL")  # Set in Streamlit secrets
    if GDRIVE_URL:
        return download_model_from_gdrive(GDRIVE_URL)

    # If neither configured
    st.error("""
    ❌ Model download not configured.

    Please either:
    1. Upload model to Hugging Face and set HF_MODEL_REPO secret
    2. Upload model to Google Drive and set GDRIVE_MODEL_URL secret

    See STREAMLIT_DEPLOYMENT.md for instructions.
    """)
    return False
```

### Option B: Use Smaller Base Model (Fallback)

If model download fails, fall back to base model (no fine-tuning):

```python
# In app.py, modify load_explainer()
@st.cache_resource
def load_explainer():
    """Load the fine-tuned model (cached)"""

    # Try to ensure model is available
    from utils.model_downloader import ensure_model_available

    if not ensure_model_available():
        st.warning("⚠️ Using base model (not fine-tuned). Explanations may be less accurate.")
        # Use base model without adapters
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-4.0-h-350m",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-h-350m")
        # Create explainer with base model
        # ... (simplified explainer logic)
    else:
        # Normal fine-tuned model loading
        explainer = GraniteLoanExplainer(
            base_model_path="ibm-granite/granite-4.0-h-350m",
            adapter_path="models/granite-finetuned"
        )
        return explainer
```

---

## 🔧 Step 2: Update Dependencies

**Add to `requirements.txt`:**
```txt
# Existing dependencies
torch>=2.0.0
transformers>=4.35.0
streamlit>=1.29.0
# ... rest of your dependencies

# Add for Streamlit Cloud deployment
gdown>=4.7.1  # Google Drive downloader
huggingface-hub>=0.19.0  # Hugging Face model download
```

**Create `packages.txt` (system dependencies):**
```
build-essential
```

---

## 🎨 Step 3: Configure Streamlit

**Create `.streamlit/config.toml`:**
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
base = "dark"
primaryColor = "#1976d2"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d24"
textColor = "#fafafa"
font = "sans serif"

[client]
showErrorDetails = true
```

---

## 📤 Step 4: Upload Model (Choose One Option)

### Option A: Hugging Face Hub (Recommended)

**1. Create Hugging Face account**: https://huggingface.co/join

**2. Create new model repository**:
- Go to https://huggingface.co/new
- Name: `lendsafe-granite`
- Type: Model
- License: MIT
- Click "Create model repository"

**3. Upload model**:
```bash
# Install huggingface-cli
pip install huggingface-hub

# Login
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens

# Upload model
huggingface-cli upload ad2546/lendsafe-granite models/granite-finetuned/ --repo-type model
```

**4. Get repository ID**: `ad2546/lendsafe-granite`

### Option B: Google Drive

**1. Zip the model**:
```bash
cd models
zip -r granite-finetuned.zip granite-finetuned/
```

**2. Upload to Google Drive**:
- Go to https://drive.google.com
- Upload `granite-finetuned.zip`
- Right-click → Share → Anyone with the link
- Copy the sharing link

**3. Extract file ID from link**:
```
Link: https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing
File ID: 1ABC123xyz
```

---

## 🌐 Step 5: Deploy to Streamlit Cloud

### 1. Sign Up for Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "Sign up"
3. **Sign in with GitHub** (recommended)
4. Authorize Streamlit to access your repositories

### 2. Create New App

1. Click "New app" button
2. Fill in details:
   - **Repository**: `ad2546/LendSafe`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click "Advanced settings"

### 3. Configure Secrets

In Advanced settings, add your model source:

**If using Hugging Face**:
```toml
# .streamlit/secrets.toml format
HF_MODEL_REPO = "ad2546/lendsafe-granite"
```

**If using Google Drive**:
```toml
GDRIVE_MODEL_URL = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"
```

### 4. Deploy!

1. Click "Deploy!"
2. Wait 5-10 minutes for first deployment
3. Your app will be at: `https://lendsafe.streamlit.app`

---

## 🔍 Monitoring Deployment

### Watch Build Logs

- Streamlit Cloud shows real-time logs
- Look for:
  - ✅ Dependencies installed
  - ✅ Model downloading (first time)
  - ✅ App starting
  - ✅ "Your app is live!"

### Common Issues

**Issue 1: Out of Memory**
```
MemoryError: Unable to allocate X GB
```
**Solution**: Use CPU-optimized model loading:
```python
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-4.0-h-350m",
    torch_dtype=torch.float32,  # Use float32 instead of float16
    device_map="cpu",
    low_cpu_mem_usage=True
)
```

**Issue 2: Model Download Timeout**
```
Timeout downloading model
```
**Solution**: Increase timeout in downloader or use smaller chunks

**Issue 3: Module Not Found**
```
ModuleNotFoundError: No module named 'X'
```
**Solution**: Add missing package to `requirements.txt`

---

## 🎯 Post-Deployment

### Update GitHub README

Add deployment badge to README.md:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lendsafe.streamlit.app)
```

### Test Your Deployed App

1. Visit `https://lendsafe.streamlit.app` (or your custom URL)
2. Test all 3 example scenarios
3. Check explanation generation
4. Verify dark theme loads correctly

### Share Your Live Demo

Update your posts with live link:
- LinkedIn: "Check out the live demo: lendsafe.streamlit.app"
- GitHub README: Add "Live Demo" button
- Resume: Include deployed URL

---

## 📊 App Settings

In Streamlit Cloud dashboard:

### Custom Domain (Optional)
- Settings → Domains
- Add custom domain (requires DNS setup)

### Analytics
- Settings → Analytics
- View usage stats

### Auto-Deploy
- ✅ Already enabled by default
- Push to GitHub → Auto-deploys

---

## 🔄 Updating Your Deployed App

```bash
# Make changes locally
# Edit app.py or other files

# Commit and push
git add .
git commit -m "Update: improved explanations"
git push origin main

# Streamlit Cloud auto-deploys in 2-3 minutes!
```

---

## 💰 Cost

**Streamlit Community Cloud**:
- ✅ FREE for public repos
- ✅ 1 app limit (free tier)
- ✅ Unlimited viewers

**Streamlit Cloud Paid** (if you need more):
- $20/month: 3 apps, 1GB RAM each
- Custom pricing for teams

---

## ✅ Deployment Checklist

Pre-deployment:
- [ ] Model uploaded to Hugging Face or Google Drive
- [ ] `requirements.txt` updated with `gdown` or `huggingface-hub`
- [ ] `packages.txt` created
- [ ] `.streamlit/config.toml` created
- [ ] Model download logic added to app
- [ ] Tested locally

Deployment:
- [ ] Streamlit Cloud account created
- [ ] New app created
- [ ] Secrets configured
- [ ] Deployed successfully
- [ ] Tested live app

Post-deployment:
- [ ] README badge added
- [ ] Live URL shared
- [ ] Custom domain setup (optional)
- [ ] Analytics enabled

---

## 🎓 Quick Summary

**Fastest path to deployment**:

1. **Upload model to Hugging Face**:
   ```bash
   huggingface-cli upload ad2546/lendsafe-granite models/granite-finetuned/
   ```

2. **Add model downloader** to `app.py`

3. **Push changes** to GitHub:
   ```bash
   git add .
   git commit -m "Add Streamlit Cloud support"
   git push origin main
   ```

4. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - New app → Select your repo
   - Add secret: `HF_MODEL_REPO = "ad2546/lendsafe-granite"`
   - Deploy!

5. **Share**: `https://lendsafe.streamlit.app`

---

## 📞 Support

**Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
**Community Forum**: https://discuss.streamlit.io
**GitHub Issues**: https://github.com/ad2546/LendSafe/issues

---

**Your app will be live at**: `https://YOUR-APP-NAME.streamlit.app`

Good luck! 🚀
