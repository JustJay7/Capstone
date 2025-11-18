import torch

ckpt_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\yolov8_crack_epoch_60.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
print(ckpt.keys())
