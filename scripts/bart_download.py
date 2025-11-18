import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Absolute target path
target_dir = r"D:\Capstone_research\models\model weights\generalist model weights\BART-Large-MNLI"
os.makedirs(target_dir, exist_ok=True)

# Hard-lock all HF caches to D:
os.environ["HF_HOME"] = r"D:\Capstone_research\.huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Capstone_research\.huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\Capstone_research\.huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"   # allow download once

print(f"⏳ Downloading facebook/bart-large-mnli → {target_dir}")

# Download model + tokenizer directly into target_dir
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", cache_dir=target_dir)
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli", cache_dir=target_dir)

# Force load once to check everything
device = "cuda" if torch.cuda.is_available() else "cpu"
_ = model.to(device)
print("✅ Download complete and verified on", device)
print("📂 All files stored in:", target_dir)
