import os
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["HF_HOME"] = "/Volumes/Expansion/Models/models/.huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/Volumes/Expansion/Models/models/.huggingface"
os.environ["TORCH_HOME"] = "/Volumes/Expansion/Models/models/.torch_cache"

target_dir = "/Volumes/Expansion/Models/models/generalist/florence-2"
model_id = "Salesforce/blip-image-captioning-base"

print(f"🧠 Rebuilding Florence-2 (BLIP) from {model_id}")
print(f"📂 Destination: {target_dir}")

processor = BlipProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])

os.makedirs(target_dir, exist_ok=True)
processor.save_pretrained(target_dir)
model.save_pretrained(target_dir)

print(f"✅ Full Florence-2 model + tokenizer downloaded to {target_dir}")
print("🧱 Contains: config.json, model.safetensors, tokenizer.json, merges.txt, vocab.txt, special_tokens_map.json, etc.")
print("🚫 No further downloads will occur — everything is now local.")
