import os
import torch
from ultralytics import YOLO
from scripts.train_yolov8_crack_final import inject_custom_blocks  # must exist in PYTHONPATH

# ======= EDIT THIS TO YOUR CHECKPOINT =======
CRACK_WEIGHTS = "/Volumes/Expansion/Models/models/infrastructure_models/yolov8_crack/model.pt"
# ============================================

class CrackDetector:
    """
    Loads YOLOv8s backbone, injects your GCC3 / SGAM / CoordAttention blocks,
    then loads raw state_dict weights from your training checkpoint.
    Provides a simple .detect_crack(image_path) API for the router.
    """
    def __init__(self, device: str | None = None, imgsz: int = 640):
        self.imgsz = imgsz
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build model graph from yaml (NOT .pt) so we control the architecture
        self.model = YOLO("yolov8s.yaml")
        inject_custom_blocks(self.model.model)            # <- your custom modules
        self.model.model.to(self.device).eval()

        # Load your raw checkpoint (state_dict or dict-of-tensors)
        state = torch.load(CRACK_WEIGHTS, map_location=self.device)
        state_dict = state.get("state_dict", state)       # support plain dict
        # best effort load; custom heads may not map 1:1
        self.model.model.load_state_dict(state_dict, strict=False)
        print(f"✅ Crack model ready ({os.path.basename(CRACK_WEIGHTS)}) on {self.device}")

    def detect_crack(self, image_path: str, conf_thresh: float = 0.10) -> tuple[bool, float]:
        """
        Returns (present: bool, confidence: float[0..1]) based on mean box conf.
        """
        results = self.model.predict(
            source=image_path,
            imgsz=self.imgsz,
            conf=conf_thresh,
            device=self.device,
            verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return False, 0.0

        confs = results[0].boxes.conf.detach().cpu().tolist()
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return True, mean_conf
