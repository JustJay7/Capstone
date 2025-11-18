import torch
from ultralytics import YOLO

# === Paths ===
weights_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\yolov8_crack_epoch_60.pt"
output_path  = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\rebuilt_yolov8s_crack.pt"

# === Step 1: Load base YOLOv8s model ===
base = YOLO("yolov8s.pt")

# === Step 2: Load raw state_dict ===
state_dict = torch.load(weights_path, map_location="cpu")
if "model" in state_dict:
    # in case it’s a full ckpt (unlikely)
    state_dict = state_dict["model"].state_dict() if hasattr(state_dict["model"], "state_dict") else state_dict["model"]

# === Step 3: Apply weights ===
missing, unexpected = base.model.load_state_dict(state_dict, strict=False)
print(f"⚙️ Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")

# === Step 4: Save rebuilt model ===
base.save(output_path)
print(f"\n✅ Rebuilt YOLOv8s checkpoint saved at:\n{output_path}")
