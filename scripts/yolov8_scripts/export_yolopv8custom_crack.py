import sys, os
sys.path.append(r"D:\Capstone_research")  # add project root to import path



import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from models.customyolo.models.coord_attn import CoordAttention
from models.customyolo.models.gcc3 import GCC3
from models.customyolo.models.sgam import SGAM
from models.customyolo.models.deco_head import DecoupledHead

# ==============================
#  CONFIG (from your YAML files)
# ==============================
num_classes = 1
anchors = [
    [490, 192],
    [64, 50],
    [230, 63],
    [151, 196],
    [159, 499],
]
ckpt_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\yolov8_crack_epoch_60.pt"
onnx_out = ckpt_path.replace(".pt", "_custom.onnx")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
#  MODEL DEFINITION (exactly as trained)
# ==============================
class YOLOPv8Custom(nn.Module):
    def __init__(self, num_classes, anchors):
        super(YOLOPv8Custom, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        # Dummy attrs (Ultralytics compatibility)
        self.args = type('', (), {})()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yaml = {'anchors': self.anchors}

        # Load YOLOv8s backbone
        self.backbone = DetectionModel(cfg='yolov8s.yaml')
        checkpoint = torch.load('yolov8s.pt', map_location='cpu')
        self.backbone.load_state_dict(checkpoint['model'].float().state_dict(), strict=False)
        print("✅ Loaded YOLOv8s pretrained backbone weights.")

        # Freeze early backbone layers
        for param in self.backbone.model[0:6].parameters():
            param.requires_grad = False

        # Crack-optimized neck
        self.neck = nn.Sequential(
            GCC3(512),
            CoordAttention(512, 512),
            SGAM(512)
        )

        # Decoupled detection head
        self.head = DecoupledHead(
            in_channels=512,
            num_anchors=len(anchors),
            num_classes=num_classes
        )

        # Store for convenience
        self.model = nn.ModuleList([self.backbone.model, self.neck, self.head])

    def forward(self, x):
        # Extract features
        x = self.backbone.model[:10](x)
        # Pass through crack-specific neck
        x = self.neck(x)
        # Predict boxes + conf + cls
        x = self.head(x)
        return x


# ==============================
#  EXPORT SCRIPT
# ==============================
def main():
    print(f"🧠 Initializing YOLOPv8Custom on {device}...")
    model = YOLOPv8Custom(num_classes=num_classes, anchors=anchors).to(device)

    print(f"📦 Loading trained weights: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Sanity check
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out = model(dummy)
        print(f"✅ Sanity forward OK → output shape: {tuple(out.shape)}")

    print("💾 Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy,
        onnx_out,
        input_names=["images"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True
    )

    print(f"✅ Export complete: {onnx_out}")


if __name__ == "__main__":
    main()
