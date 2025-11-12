---
title: LendSafe
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app_gradio.py
pinned: false
license: mit
---

# 🏦 LendSafe - AI Loan Decision Explainer

AI-powered loan decision explanation system with FCRA-compliant explanations using fine-tuned IBM Granite 350M.

## Features

- ✅ FCRA/ECOA Compliant explanations
- 🔒 Fine-tuned IBM Granite 4.0 H 350M with LoRA
- ⚡ <10 second inference
- 🎨 Beautiful Gradio interface
- 🎯 Privacy-first architecture

## How to Use

1. Fill out the loan application form
2. Click "Analyze Application"
3. Get instant risk score and AI explanation

## Tech Stack

- **LLM**: IBM Granite 4.0 H 350M
- **Fine-tuning**: PEFT/LoRA (r=8, alpha=16)
- **Frontend**: Gradio
- **Deployment**: Hugging Face Spaces

## Model

The fine-tuned model is available at: [notatharva0699/lendsafe-granite](https://huggingface.co/notatharva0699/lendsafe-granite)

## Links

- **GitHub**: https://github.com/ad2546/LendSafe
- **Model**: https://huggingface.co/notatharva0699/lendsafe-granite

Built with IBM Granite AI 🤖
