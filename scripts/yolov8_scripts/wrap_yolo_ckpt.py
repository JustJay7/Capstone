# D:\Capstone_research\scripts\wrap_yolo_ckpt.py
import torch

# Path to your fixed state dict
weights_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\fixed_yolov8s_crack.pt"
wrapped_path = weights_path.replace(".pt", "_wrapped.pt")

print("🔧 Wrapping raw state_dict into Ultralytics checkpoint format...")

# load state dict
state_dict = torch.load(weights_path, map_location="cpu")

# wrap in dict structure Ultralytics expects
ckpt = {"model": state_dict, "ema": None, "updates": 0, "optimizer": None, "best_fitness": None}
torch.save(ckpt, wrapped_path)

print(f"✅ Wrapped checkpoint saved to:\n{wrapped_path}")
