from transformers import BlipProcessor, BlipForConditionalGeneration

# Force everything to D: drive
import os
os.environ["HF_HOME"] = "/Volumes/Expansion/Models/models/.huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/Volumes/Expansion/Models/models/.huggingface"

# Download official BLIP weights from Salesforce
model_id = "Salesforce/blip-image-captioning-base"
target_dir = "/Volumes/Expansion/Models/models/generalist/florence-2"

print("⏬ Downloading full Florence (BLIP) model to:", target_dir)
processor = BlipProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])

# Save everything locally to your Florence folder
processor.save_pretrained(target_dir)
model.save_pretrained(target_dir)
print("✅ Florence-2 rebuilt successfully at", target_dir)
