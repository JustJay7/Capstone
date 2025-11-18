from ultralytics import YOLO
import torch

rebuilt_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\rebuilt_yolov8s_crack.pt"
output_path = rebuilt_path.replace("rebuilt_yolov8s_crack", "final_fixed_yolov8s_crack.pt")

print("🔧 Loading YOLOv8s base architecture...")
model = YOLO("yolov8s.pt")  # fresh YOLOv8s backbone

print("🧠 Loading your crack detection state dict...")
state_dict = torch.load(rebuilt_path, map_location="cpu")
model.model.load_state_dict(state_dict, strict=False)

print("🧩 Rebuilding detection head for 1 class (crack)...")
detect = model.model.model[-1]
detect.nc = 1
detect.names = {0: "crack"}
detect.no = detect.nc + 4  # bbox(4) + cls(1)

# rebuild conv layers to match correct output dimension
detect.cv2 = torch.nn.ModuleList([
    torch.nn.Sequential(
        torch.nn.Conv2d(c.in_channels, detect.no, kernel_size=1),
        torch.nn.Sigmoid()
    ) if isinstance(c, torch.nn.Conv2d) else c
    for c in detect.cv2
])

print("✅ Detection head rebuilt → 1 class only")
torch.save({"model": model.model.float()}, output_path)
print(f"✅ Saved final fixed checkpoint:\n{output_path}")
