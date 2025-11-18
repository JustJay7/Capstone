from transformers import CLIPModel
from safetensors.torch import save_file
import torch
import os

# path to your local CLIP model
src = r"D:\Capstone_research\models\model weights\generalist model weights\clip"
dst = os.path.join(src, "model.safetensors")

print("⏳ Loading CLIP model weights from", src)
model = CLIPModel.from_pretrained(src, local_files_only=True)
state_dict = model.state_dict()
print("💾 Converting to .safetensors ...")
save_file(state_dict, dst)
print("✅ Saved:", dst)
