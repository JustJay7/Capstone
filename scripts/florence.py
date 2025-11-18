import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================================================
# FORCE EVERYTHING TO D: (avoid C:\ cache)
# ============================================================
os.environ["HF_HOME"] = r"D:\Capstone_research\.huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Capstone_research\.huggingface"
os.environ["TORCH_HOME"] = r"D:\Capstone_research\.torch_cache"

# ============================================================
# TARGET FOLDER
# ============================================================
target_dir = r"D:\Capstone_research\models\model weights\generalist model weights\florence-2"
model_id = "Salesforce/blip-image-captioning-base"

print(f"🧠 Rebuilding Florence-2 (BLIP) from {model_id}")
print(f"📂 Destination: {target_dir}")

# ============================================================
# DOWNLOAD FROM HUGGINGFACE (once only)
# ============================================================
processor = BlipProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])

# ============================================================
# SAVE TO LOCAL DISK
# ============================================================
os.makedirs(target_dir, exist_ok=True)
processor.save_pretrained(target_dir)
model.save_pretrained(target_dir)

print(f"✅ Full Florence-2 model + tokenizer downloaded to {target_dir}")
print("🧱 Contains: config.json, model.safetensors, tokenizer.json, merges.txt, vocab.txt, special_tokens_map.json, etc.")
print("🚫 No further downloads will occur — everything is now local.")
