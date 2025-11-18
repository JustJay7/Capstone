import os

# Base directory
base_dir = "/Users/jay/Desktop/Projects/Capstone/models"

# Expected models and common weight files
expected_models = {
    "generalist/Florence-2": ["model.safetensors", "pytorch_model.bin"],
    "generalist/CLIP": ["openai/clip-vit-base-patch16", "pytorch_model.bin"],
    "object/yolov8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    "object/segment-anything/sam-vit-huge": ["model.safetensors", "pytorch_model.bin"],
    "object/segment-anything/sam-vit-large": ["model.safetensors", "pytorch_model.bin"],
    "object/segment-anything/sam-vit-base": ["model.safetensors", "pytorch_model.bin"],
    "object/detr": ["pytorch_model.bin"],
    "object/GroundingDINO": ["pytorch_model.bin"],
    "output_gate/blip-2": ["pytorch_model.bin", "model.safetensors"],
    "output_gate/llava": ["pytorch_model.bin", "model.safetensors"],
    "weather/satmae": ["model.safetensors", "pytorch_model.bin"],
    "weather/climax": ["model.safetensors", "pytorch_model.bin"],
    "medical/chexnet": ["chexnet_model.pt", "model.pth", "model.pt"],
    "medical/BiomedCLIP": ["pytorch_model.bin"],
    "medical/MONAI": ["sam_backbone.bin", "medsam_vit_b.pth"],
    "medical/nnUNet": ["checkpoint_final.pth"],
}

print("\n📋 Pretrained Weights Report")
print("=" * 40)

for model, candidates in expected_models.items():
    model_path = os.path.join(base_dir, model.replace("/", os.sep))
    if not os.path.exists(model_path):
        print(f"❌ {model}  [missing folder]")
        continue

    # Search for weight files
    found = []
    for root, _, files in os.walk(model_path):
        for f in files:
            if any(f.endswith(ext) or f == ext for ext in candidates):
                found.append(os.path.join(root, f))

    if found:
        print(f"✅ {model}  [found: {len(found)} file(s)]")
        for f in found:
            print(f"   └─ {f}")
    else:
        print(f"⚠️ {model}  [folder exists, but no weights found]")

print("=" * 40)
print("✔️ Report complete.\n")
