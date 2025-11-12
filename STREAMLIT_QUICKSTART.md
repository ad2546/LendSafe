# 🚀 Deploy to Streamlit Cloud - Quick Start

**5-minute guide to get your app live!**

---

## ⚡ Fastest Path to Deployment

Since your model files are too large for GitHub, we'll use **base model only** for the demo (no fine-tuning). This will work immediately on Streamlit Cloud.

### Option 1: Deploy with Base Model (Fastest - 5 minutes)

The app will work with the base IBM Granite 350M model (not fine-tuned). Explanations will be less tailored but still functional.

**Steps**:

1. **Go to Streamlit Cloud**: https://share.streamlit.io

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure**:
   - Repository: `ad2546/LendSafe`
   - Branch: `main`
   - Main file: `app.py`

5. **Click "Deploy!"**

6. **Wait 5-10 minutes** for deployment

7. **Your app is live!** at `https://lendsafe.streamlit.app` (or similar)

**That's it!** The app will:
- ✅ Load the base Granite model
- ✅ Work without fine-tuned weights
- ⚠️ Explanations won't be as good (but functional)

---

### Option 2: Deploy with Fine-Tuned Model (Best Quality - 30 minutes)

For the best explanations, upload your fine-tuned model first.

#### Step 1: Upload Model to Hugging Face (15 min)

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Create repo on Hugging Face website first:
# Go to https://huggingface.co/new
# Name: lendsafe-granite
# Type: Model
# License: MIT
# Click "Create"

# Upload your model
huggingface-cli upload ad2546/lendsafe-granite models/granite-finetuned/ --repo-type model
```

#### Step 2: Deploy to Streamlit Cloud (5 min)

1. **Go to**: https://share.streamlit.io

2. **New app** → Configure:
   - Repository: `ad2546/LendSafe`
   - Branch: `main`
   - Main file: `app.py`

3. **Advanced settings** → Secrets:
   ```toml
   HF_MODEL_REPO = "ad2546/lendsafe-granite"
   ```

4. **Deploy!**

5. **Wait 10-15 minutes** (first time downloads model)

6. **App is live!** with your fine-tuned model

---

## 🎯 What to Expect

### First Deployment
- Build time: 5-10 minutes
- Model download (if using fine-tuned): 5-10 minutes
- **Total**: 10-20 minutes

### Subsequent Visits
- Model is cached
- App loads in <30 seconds
- **Fast!**

---

## 📊 Your Deployed App Will Have

- ✅ Dark theme (matching local version)
- ✅ Loan application form
- ✅ AI explanation generation
- ✅ Example scenarios
- ✅ Professional UI

**URL Format**: `https://APP-NAME.streamlit.app`
- Example: `https://lendsafe.streamlit.app`
- Or: `https://ad2546-lendsafe.streamlit.app`

---

## 🐛 Troubleshooting

### "Out of memory" Error

**Solution**: App uses <2GB, should be fine. If issues:
1. Check logs in Streamlit Cloud
2. Try deploying with base model first

### Model Download Timeout

**Solution**:
1. Make sure HF_MODEL_REPO secret is set correctly
2. Verify model is public on Hugging Face
3. Check model size (<1GB recommended for Streamlit Cloud)

### App Won't Start

**Solution**:
1. Check deployment logs
2. Verify `requirements.txt` has all dependencies
3. Make sure `packages.txt` exists
4. Check for Python errors in logs

---

## ✅ Post-Deployment Checklist

- [ ] App deployed successfully
- [ ] Visited app URL and it loads
- [ ] Tested example scenarios
- [ ] Dark theme displays correctly
- [ ] Explanations generate without errors
- [ ] Added badge to GitHub README
- [ ] Shared on LinkedIn/Twitter

---

## 📝 Add Deployment Badge to README

Update your GitHub README.md:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)
```

Replace `YOUR-APP-URL` with your actual app URL.

---

## 🎬 Next Steps After Deployment

1. **Test the app** thoroughly
2. **Share the URL** on social media
3. **Update resume** with live demo link
4. **Monitor usage** in Streamlit Cloud dashboard
5. **Iterate** based on feedback

---

## 🔗 Resources

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Full Guide**: [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)
- **Troubleshooting**: https://docs.streamlit.io/knowledge-base
- **Community**: https://discuss.streamlit.io

---

## 🚀 Ready to Deploy?

**Fastest**: Go to https://share.streamlit.io → Sign in → New app → Deploy!

**Best Quality**: Upload model to HF first, then deploy with secrets

---

**Your app will be live in ~10 minutes!** 🎉

Visit: https://share.streamlit.io to get started!
