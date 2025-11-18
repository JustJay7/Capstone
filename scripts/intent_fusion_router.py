# D:\Capstone_research\scripts\intent_fusion_router.py
# Updated version with user input for custom image + text prompt

import os
from pathlib import Path
import json
from typing import List, Tuple
import torch
from PIL import Image
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# =========================
# HARD PATHS
# =========================
ROOT = Path("/Volumes/Expansion/Models")

CLIP_CACHE = ROOT / ".clip_cache"
HF_CACHE   = ROOT / ".huggingface"
TORCH_HOME = ROOT / ".torch_cache"

TAXONOMY_JSON = ROOT / "configs" / "intent_taxonomy.json"
FLORENCE_DIR = ROOT / "models" / "generalist" / "florence-2"
CLIP_PT = ROOT / "models" / "generalist" / "clip" / "pytorch_model.bin"
YOLO_PT = ROOT / "models" / "object" / "yolov8" / "yolov8s.pt"

# Fusion weights
W_CLIP = 0.60
W_FLOR = 0.25
W_YOLO = 0.15

# =========================
# FORCE OFFLINE ENV
# =========================
for p in [CLIP_CACHE, HF_CACHE, TORCH_HOME]:
    p.mkdir(parents=True, exist_ok=True)

os.environ.update({
    "XDG_CACHE_HOME": str(CLIP_CACHE),
    "CLIP_CACHE_DIR": str(CLIP_CACHE),
    "HF_HOME": str(HF_CACHE),
    "TRANSFORMERS_CACHE": str(HF_CACHE),
    "TORCH_HOME": str(TORCH_HOME),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
})

# =========================
# VALIDATION
# =========================
def _ensure(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing {label}: {path}")

_ensure(TAXONOMY_JSON, "taxonomy JSON")
_ensure(FLORENCE_DIR,  "Florence directory")
_ensure(CLIP_PT,       "CLIP model file")
_ensure(YOLO_PT,       "YOLO weights")

# =========================
# LOAD MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Device: {device}")

print(f"✅ CLIP from {CLIP_PT}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root=str(CLIP_CACHE))

print("⏳ Loading Florence (BLIP) locally ...")
processor = BlipProcessor.from_pretrained(str(FLORENCE_DIR), local_files_only=True)
florence = BlipForConditionalGeneration.from_pretrained(str(FLORENCE_DIR), local_files_only=True).to(device)
florence.eval()
print("✅ Florence ready.")

print(f"⏳ Loading YOLO from {YOLO_PT}")
yolo_general = YOLO(str(YOLO_PT))
print("✅ YOLO ready.")

# =========================
# HELPERS
# =========================
def get_caption(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = florence.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.lower()

def get_clip_confidence(img_path: Path, prompts: List[str]) -> float:
    image = clip_preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        im_f = clip_model.encode_image(image)
        tx_f = clip_model.encode_text(text)
        im_f /= im_f.norm(dim=-1, keepdim=True)
        tx_f /= tx_f.norm(dim=-1, keepdim=True)
        sims = (100.0 * im_f @ tx_f.T).softmax(dim=-1)
    return float(sims.max().cpu().numpy())

def get_objects(img_path: Path, conf: float = 0.35) -> List[str]:
    res = yolo_general.predict(source=str(img_path), conf=conf, imgsz=640, verbose=False)
    names = res[0].names
    cls = res[0].boxes.cls.cpu().tolist() if res and res[0].boxes is not None else []
    return list({names[int(c)] for c in cls})

def fuse_score(clip_conf, pos_hits, neg_hits, obj_hits) -> float:
    return (W_CLIP * clip_conf) + (W_FLOR * (pos_hits - neg_hits)) + (W_YOLO * obj_hits)

def classify_intent(img_path: Path, taxonomy: dict, user_hint: str = ""):
    caption = get_caption(img_path)
    objs = get_objects(img_path)
    results = []

    for intent in taxonomy["intents"]:
        clip_conf = get_clip_confidence(img_path, intent.get("clip_text_prompts", []))
        pos_hits = sum(1 for k in intent.get("florence_positive_markers", []) if k in caption)
        neg_hits = sum(1 for k in intent.get("florence_negative_markers", []) if k in caption)
        obj_hits = sum(1 for k in intent.get("object_markers", []) if k in objs)
        score = fuse_score(clip_conf, pos_hits, neg_hits, obj_hits)

        for rule in intent.get("routing_policy", {}).get("boost_rules", []):
            if "if_hint_contains_any" in rule and any(k in user_hint.lower() for k in rule["if_hint_contains_any"]):
                score *= float(rule.get("boost_intent_by", 1.0))
            if "if_caption_contains_any" in rule and any(k in caption for k in rule["if_caption_contains_any"]):
                score *= float(rule.get("boost_intent_by", 1.0))
            if "if_objects_include_any" in rule and any(k in objs for k in rule["if_objects_include_any"]):
                score *= float(rule.get("boost_intent_by", 1.0))

        results.append((intent["intent_id"], round(score, 3), round(clip_conf, 3), pos_hits, neg_hits, obj_hits, objs))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[0], results[:5], caption

# =========================
# MAIN
# =========================
def main():
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    img_path = input("📁 Enter image path: ").strip().strip('"')
    hint = input("💬 Enter optional text prompt (press Enter to skip): ").strip()

    path = Path(img_path)
    if not path.exists():
        print(f"❌ Image not found: {path}")
        return

    print("\n🔍 Running multimodal analysis ...\n")
    best, top5, caption = classify_intent(path, taxonomy, user_hint=hint)

    print("=" * 100)
    print(f"🖼️  Image: {path.name}")
    print(f"🧾 Caption: {caption}")
    print("\n🏆 BEST MATCH")
    print(f"Intent: {best[0]} | Score={best[1]} | CLIP={best[2]} | +{best[3]}/-{best[4]} | obj_hits={best[5]}")
    print(f"Detected objects: {best[6]}\n")

    print("🔝 TOP 5 INTENTS:")
    for r in top5:
        print(f" - {r[0]} | Score={r[1]} | CLIP={r[2]} | +{r[3]}/-{r[4]} | obj_hits={r[5]}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
