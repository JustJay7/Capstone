from ultralytics import YOLO
import torch

base_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\rebuilt_yolov8s_crack.pt"

print("🔧 Loading YOLOv8 model...")
model = YOLO(base_path)
detect_layer = model.model.model[-1]

print(f"🧠 Detected Detect layer type: {type(detect_layer)}")

# Force 1-class configuration
detect_layer.nc = 1
detect_layer.names = {0: "crack"}
detect_layer.no = detect_layer.nc + 4

# If model has convolutional heads list (.cv2 or .cv3 or .m), rebuild them
if hasattr(detect_layer, "m") and isinstance(detect_layer.m, torch.nn.ModuleList):
    print("🔧 Found .m heads → rebuilding...")
    detect_layer.m = torch.nn.ModuleList([
        torch.nn.Conv2d(x.in_channels, detect_layer.no * len(detect_layer.anchors[0]), 1)
        for x in detect_layer.m
    ])
elif hasattr(detect_layer, "cv2"):
    print("🔧 Found .cv2 heads → rebuilding...")
    for branch in detect_layer.cv2:
        if isinstance(branch, torch.nn.ModuleList):
            for i, conv in enumerate(branch):
                if isinstance(conv, torch.nn.Conv2d):
                    branch[i] = torch.nn.Conv2d(conv.in_channels, detect_layer.no, 1)
        elif isinstance(branch, torch.nn.Conv2d):
            branch.out_channels = detect_layer.no

else:
    print("⚠️ Could not find .m or .cv2 — keeping detection heads as-is (should still work if shapes match).")

fixed_path = base_path.replace("rebuilt_yolov8s_crack", "fixed_yolov8s_crack")
torch.save(model.model.state_dict(), fixed_path)
print(f"✅ Fixed checkpoint saved to:\n{fixed_path}")
