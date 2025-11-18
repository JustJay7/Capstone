from ultralytics import YOLO
import torch

rebuilt_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\rebuilt_yolov8s_crack.pt"
fixed_path = rebuilt_path.replace("rebuilt_yolov8s_crack", "final_yolov8s_crack.pt")

print("🔧 Loading model base architecture (YOLOv8s)...")
base = YOLO("yolov8s.pt")  # get correct architecture

print("🧠 Loading your fixed state_dict weights...")
state_dict = torch.load(r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\fixed_yolov8s_crack.pt", map_location="cpu")

# load into YOLO architecture
base.model.load_state_dict(state_dict, strict=False)

# update class metadata
base.model.model[-1].nc = 1
base.model.model[-1].names = {0: "crack"}

# save full checkpoint in proper format
ckpt = {"model": base.model.float(), "ema": None, "updates": 0, "optimizer": None, "best_fitness": None}
torch.save(ckpt, fixed_path)

print(f"✅ Full Ultralytics-compatible checkpoint saved to:\n{fixed_path}")
