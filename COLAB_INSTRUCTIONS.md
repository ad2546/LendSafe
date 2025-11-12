# 🚀 LendSafe Fine-tuning on Google Colab

## Why Colab?

Your M2 MacBook Air (8GB RAM) cannot handle the Granite 3.1 3B model fine-tuning due to:
- **MPS memory limit**: 9.07 GB maximum
- **Model memory**: 8.97 GB just to load the model
- **Training overhead**: Additional memory needed for gradients, optimizer states, etc.

**Google Colab provides**:
- Free Tesla T4 GPU (16GB VRAM)
- Training time: 15-30 minutes (vs 30-60 minutes on CPU)
- Zero cost for this project

---

## 📋 Step-by-Step Instructions

### 1. Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload: `LendSafe_Finetune_Colab.ipynb` (from this directory)

### 2. Configure Runtime

1. Click **Runtime → Change runtime type**
2. Select:
   - **Hardware accelerator**: GPU
   - **GPU type**: T4 (default, free tier)
3. Click **Save**

### 3. Run Fine-tuning

**Important**: Run cells in order. DO NOT skip cells.

#### Cell-by-Cell Workflow:

1. **Install Dependencies** (Cell 2)
   - Run time: ~30 seconds
   - Installs: torch, transformers, peft, datasets

2. **Upload Training Data** (Cell 4)
   - Click "Choose Files"
   - Upload: `data/synthetic/training_examples.jsonl` (677 KB)
   - Verify: "✅ Training data uploaded successfully!"

3. **Configure Model** (Cell 6)
   - Automatic - just run the cell
   - Sets up Granite 350M with optimized batch sizes for T4 GPU

4. **Load Model** (Cell 7)
   - Run time: ~2 minutes
   - Downloads IBM Granite 350M from Hugging Face
   - Verify: "✅ Model loaded: 3,298,793,472 parameters"

5. **Configure LoRA** (Cell 8)
   - Run time: ~10 seconds
   - Sets up parameter-efficient training
   - Verify: "Trainable params: 5,242,880 (0.16%)"

6. **Prepare Dataset** (Cell 10)
   - Run time: ~5 seconds
   - Tokenizes 1,500 examples
   - Verify: "Train: 1350, Val: 150"

7. **Train Model** (Cell 12-13)
   - **THIS IS THE MAIN TRAINING**
   - Run time: **15-30 minutes**
   - Expected steps: 255 total
   - Watch for decreasing loss values
   - Expected final loss: ~0.8-1.2

8. **Test Generation** (Cell 15)
   - Run time: ~10 seconds
   - Generates sample explanation
   - Verify output quality

9. **Save & Download** (Cell 17-18)
   - Creates zip file: `granite-finetuned-final.zip`
   - Download starts automatically
   - File size: ~7 GB (base model + LoRA weights)

---

## 📊 Expected Training Output

### Phase 1: Warmup (Steps 0-100)
```
{'loss': 2.5, 'learning_rate': 4e-05, 'epoch': 0.39}
{'loss': 2.2, 'learning_rate': 8e-05, 'epoch': 0.78}
{'loss': 2.0, 'learning_rate': 1.2e-04, 'epoch': 1.18}
```

### Phase 2: Main Training (Steps 100-200)
```
{'loss': 1.8, 'learning_rate': 2e-04, 'epoch': 1.57}
{'loss': 1.5, 'learning_rate': 2e-04, 'epoch': 1.96}
{'loss': 1.2, 'learning_rate': 1.8e-04, 'epoch': 2.35}
```

### Phase 3: Convergence (Steps 200-255)
```
{'loss': 1.0, 'learning_rate': 1.2e-04, 'epoch': 2.75}
{'loss': 0.9, 'learning_rate': 8e-05, 'epoch': 2.94}
{'loss': 0.8, 'learning_rate': 4e-05, 'epoch': 3.0}
```

---

## 🎯 Success Indicators

### ✅ Training is working if you see:
- Loss decreasing over time (2.5 → 1.0 → 0.8)
- Steps progressing (0/255 → 50/255 → 255/255)
- No error messages
- GPU utilization high (check Runtime → View resources)

### ⚠️ Warning signs:
- Loss not decreasing after 100 steps
- Loss > 2.0 after 200 steps
- "Out of memory" errors → Restart runtime, reduce batch size to 4
- Training stuck → Check if GPU is allocated

---

## 📥 After Training Completes

### 1. Download the Model

The download should start automatically. If not:
```python
from google.colab import files
files.download('granite-finetuned-final.zip')
```

### 2. Extract Locally

On your Mac:
```bash
cd ~/LendSafe
unzip ~/Downloads/granite-finetuned-final.zip
mv granite-finetuned-final models/granite-finetuned
```

### 3. Verify Model Files

```bash
ls -lh models/granite-finetuned/
```

Expected files:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights (~20 MB)
- `tokenizer_config.json` - Tokenizer settings
- `config.json` - Model configuration
- Plus other config files

### 4. Run Evaluation

```bash
source .venv/bin/activate
python scripts/evaluate_model.py
```

Expected output:
- ROUGE-L F1: ≥ 0.40 (Good) or ≥ 0.50 (Excellent)
- BERTScore F1: ≥ 0.80 (Good) or ≥ 0.85 (Excellent)
- Sample predictions displayed
- Results saved to `models/evaluation_results/`

---

## 🛠️ Troubleshooting

### Issue: "No GPU available"
**Solution**:
- Runtime → Change runtime type → GPU
- If still unavailable, Colab free tier may be at limit
- Try again in a few hours or upgrade to Colab Pro ($9.99/mo)

### Issue: "Out of memory" during training
**Solution**: Reduce batch size
- In Cell 6, change: `BATCH_SIZE = 8` to `BATCH_SIZE = 4`
- Rerun from Cell 6 onwards

### Issue: Training loss not decreasing
**Solution**:
- Check if you uploaded the correct training file
- Verify file has 1,500 examples
- Loss should start ~2.5 and decrease to ~0.8

### Issue: Download failed
**Solution**:
```python
# In Colab, create a new cell and run:
!cp granite-finetuned-final.zip /content/drive/MyDrive/
```
Then download from Google Drive

### Issue: Model too large for download
**Solution**: Use Google Drive
```python
# Mount Google Drive (run in new cell)
from google.colab import drive
drive.mount('/content/drive')

# Copy model
!cp -r granite-finetuned-final /content/drive/MyDrive/LendSafe/
```

---

## 💰 Cost Analysis

| Item | Cost |
|------|------|
| Google Colab Free T4 GPU | $0.00 |
| Training Time (30 min) | $0.00 |
| Storage | $0.00 |
| **Total** | **$0.00** |

**vs Local M2 Training**:
- Would require 30-60 min CPU-only (if it worked)
- Higher power consumption
- Risk of system crashes
- MPS compatibility issues

**vs Cloud GPU**:
- AWS p3.2xlarge (V100): $3.06/hour × 0.5 = **$1.53 saved**
- Lambda Labs (A100): $1.10/hour × 0.25 = **$0.28 saved**

---

## 📈 Timeline

| Step | Duration |
|------|----------|
| Upload notebook | 1 min |
| Configure runtime | 1 min |
| Upload training data | 1 min |
| Install dependencies | 30 sec |
| Load model | 2 min |
| Configure LoRA | 10 sec |
| Prepare dataset | 5 sec |
| **Training** | **15-30 min** |
| Test generation | 10 sec |
| Save & download | 3 min |
| **Total** | **~25-40 minutes** |

---

## 🎓 What Happens During Training

### LoRA (Low-Rank Adaptation)
- Only 0.16% of parameters trained (5.2M out of 3.3B)
- Adds small adapter layers to frozen base model
- 99.84% memory savings vs full fine-tuning
- Achieves 90%+ performance of full fine-tuning

### Training Process
1. **Warmup** (0-100 steps): Learning rate increases gradually
2. **Main Training** (100-200 steps): Full learning rate, rapid improvement
3. **Convergence** (200-255 steps): Learning rate decreases, model refinement

### Why It Works
- 1,500 high-quality examples
- FCRA-compliant templates
- Balanced approval/rejection ratio (61/39)
- Domain-specific vocabulary
- Structured prompt format

---

## 🚀 Next Steps After Fine-tuning

1. ✅ Download fine-tuned model from Colab
2. ✅ Extract to `models/granite-finetuned/`
3. ✅ Run evaluation: `python scripts/evaluate_model.py`
4. ✅ Review ROUGE & BERTScore metrics
5. ✅ Update WEEK2_PROGRESS.md with results
6. ⬜ Begin Week 3: RAG system with ChromaDB
7. ⬜ Build compliance document database
8. ⬜ Implement LangChain pipeline
9. ⬜ Generate PDF adverse action notices
10. ⬜ Create Streamlit dashboard (Week 4)

---

## 📝 Files Checklist

### Required for Colab:
- ✅ `LendSafe_Finetune_Colab.ipynb` - Jupyter notebook
- ✅ `data/synthetic/training_examples.jsonl` - Training data (677 KB)

### Generated by Colab:
- `granite-finetuned-final.zip` - Fine-tuned model (~7 GB)

### After extraction:
- `models/granite-finetuned/` - Production model directory

---

## 🎯 Success Metrics

### Training Success:
- ✅ Final training loss: < 1.0
- ✅ Final validation loss: < 1.2
- ✅ Loss steadily decreasing
- ✅ No error messages

### Evaluation Success:
- 🎉 **Excellent**: ROUGE-L ≥ 0.50 AND BERTScore ≥ 0.85
- 👍 **Good**: ROUGE-L ≥ 0.40 AND BERTScore ≥ 0.80
- ⚠️ **Fair**: ROUGE-L ≥ 0.30 OR BERTScore ≥ 0.75
- ❌ **Poor**: Below these thresholds

---

## 💡 Pro Tips

1. **Keep Colab active**: Click in notebook every 30 min to prevent disconnection
2. **Monitor GPU usage**: Runtime → View resources
3. **Save checkpoints**: Model saves every 100 steps automatically
4. **Check logs**: Scroll through training output to verify progress
5. **Test before downloading**: Run Cell 15 to verify quality

---

## ✅ Ready to Start?

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `LendSafe_Finetune_Colab.ipynb`
3. Change runtime to GPU (T4)
4. Run all cells in order
5. Wait 15-30 minutes
6. Download the fine-tuned model
7. Run evaluation locally

**Estimated total time: 25-40 minutes**

---

**Questions?** Check the Troubleshooting section above.

**Ready to proceed?** Open the notebook and let's train! 🚀
