# ============================================================
# 4-STREAM GENERALIST INTENT ROUTER (v3)
# Fixed: BART independent classification + domain YOLO usage
# ============================================================

import os, json
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
import clip
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================
ROOT = Path(r"D:\Capstone_research")

CLIP_CACHE = ROOT / ".clip_cache"
HF_CACHE = ROOT / ".huggingface"
TORCH_HOME = ROOT / ".torch_cache"

TAXONOMY_JSON = ROOT / "configs" / "intent_taxonomy.v2.json"
FLORENCE_DIR = ROOT / "models" / "model weights" / "generalist model weights" / "florence-2"
CLIP_PT = CLIP_CACHE / "ViT-B-32.pt"

# ---- YOLO MODELS ----
YOLO_GENERIC_PT = ROOT / "models" / "model weights" / "generalist model weights" / "yolov8" / "yolov8s.pt"
YOLO_CRACK_PT   = ROOT / "models" / "model weights" / "yolov8_crackdetection_selftrained" / "best.pt"

# ---- BART ----
BART_DIR = next((ROOT / "models" / "model weights" / "generalist model weights" / "BART-Large-MNLI").rglob("config.json")).parent

DEFAULT_FUSION = {"clip": 0.45, "florence": 0.25, "yolo": 0.15, "bart": 0.15}

# ============================================================
# ENVIRONMENT
# ============================================================
for p in [CLIP_CACHE, HF_CACHE, TORCH_HOME]:
    p.mkdir(parents=True, exist_ok=True)

os.environ.update({
    "HF_HOME": str(HF_CACHE),
    "TRANSFORMERS_CACHE": str(HF_CACHE),
    "HUGGINGFACE_HUB_CACHE": str(HF_CACHE),
    "TORCH_HOME": str(TORCH_HOME),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Device: {device}")

# ============================================================
# MODEL LOAD
# ============================================================
print(f"✅ Loading CLIP from {CLIP_PT}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root=str(CLIP_CACHE))

print("⏳ Loading Florence (BLIP) locally ...")
processor = BlipProcessor.from_pretrained(str(FLORENCE_DIR), local_files_only=True)
florence = BlipForConditionalGeneration.from_pretrained(str(FLORENCE_DIR), local_files_only=True).to(device).eval()
print("✅ Florence ready.")

print(f"⏳ Loading YOLO models ...")
yolo_generic = YOLO(str(YOLO_GENERIC_PT))
yolo_crack = YOLO(str(YOLO_CRACK_PT))
print("✅ YOLO (generic + crack) ready.")

print(f"⏳ Loading BART-MNLI from {BART_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(BART_DIR), local_files_only=True)
bart_model = AutoModelForSequenceClassification.from_pretrained(str(BART_DIR), local_files_only=True).to(device).eval()
print("✅ BART zero-shot ready.")

# ============================================================
# HELPERS
# ============================================================
def get_caption(image_path: Path, max_new_tokens=25) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = florence.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True).lower().strip()

def get_clip_confidence(image_path: Path, prompts: List[str]) -> Tuple[float, List[str]]:
    if not prompts:
        return 0.0, []
    img = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    txt = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        im_f = clip_model.encode_image(img)
        tx_f = clip_model.encode_text(txt)
        im_f /= im_f.norm(dim=-1, keepdim=True)
        tx_f /= tx_f.norm(dim=-1, keepdim=True)
        sims = (100.0 * im_f @ tx_f.T).softmax(dim=-1).squeeze(0)
    topk = torch.topk(sims, k=min(2, len(prompts)))
    top_prompts = [prompts[i] for i in topk.indices.cpu().tolist()]
    return float(sims.max().cpu().numpy()), top_prompts

def get_objects(image_path: Path, model: YOLO, conf=0.35):
    res = model.predict(source=str(image_path), conf=conf, imgsz=640, verbose=False)
    names = res[0].names
    classes = res[0].boxes.cls.detach().cpu().tolist() if res and res[0].boxes is not None else []
    return list({names[int(c)] for c in classes})

def bart_classify_intents(context_text: str, intents_json: list) -> dict:
    labels = [i.get("bart_label", i["intent_id"]) for i in intents_json]
    descriptions = [i.get("bart_description", i.get("bart_label", "")) for i in intents_json]
    candidate_texts = [f"This image represents {d}." for d in descriptions]
    with torch.no_grad():
        inputs = tokenizer(
            [context_text] * len(candidate_texts),
            candidate_texts,
            truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        logits = bart_model(**inputs).logits
        entail = logits[:, 2]
        probs = torch.softmax(entail, dim=0)
    return {lbl: float(p) for lbl, p in zip(labels, probs)}

def fuse_score(clip_s, pos, neg, yolo_hit, bart_s, weights):
    w = {**DEFAULT_FUSION, **(weights or {})}
    flor_term = max(pos - neg, 0)
    return (clip_s * w["clip"]) + (flor_term * w["florence"] * 0.05) + \
           (yolo_hit * w["yolo"] * 0.05) + (bart_s * w["bart"])

# ============================================================
# CLASSIFICATION PIPELINE
# ============================================================
def classify_intent(image_path: Path, taxonomy: dict, user_hint=""):
    caption = get_caption(image_path)
    clip_conf, top_prompts = get_clip_confidence(image_path, [p for i in taxonomy["intents"] for p in i.get("clip_text_prompts", [])])
    context_text = user_hint or " ".join(top_prompts) or caption
    bart_scores = bart_classify_intents(context_text, taxonomy["intents"])

    results = []
    for intent in taxonomy["intents"]:
        iid = intent["intent_id"]
        prompts = intent.get("clip_text_prompts", [])
        pos_m = intent.get("florence_positive_markers", [])
        neg_m = intent.get("florence_negative_markers", [])
        obj_m = intent.get("object_markers", [])
        weights = intent.get("routing_policy", {}).get("fusion_weights", {})

        yolo_model = yolo_crack if "crack" in iid else yolo_generic
        objs = get_objects(image_path, yolo_model)
        clip_conf_intent, _ = get_clip_confidence(image_path, prompts)
        pos_hits = sum(1 for k in pos_m if k in caption)
        neg_hits = sum(1 for k in neg_m if k in caption)
        obj_hits = sum(1 for k in obj_m if k in objs)
        bart_label = intent.get("bart_label", iid)
        bart_s = bart_scores.get(bart_label, 0.0)

        fused = fuse_score(clip_conf_intent, pos_hits, neg_hits, obj_hits, bart_s, weights)
        results.append((iid, round(fused, 3), round(clip_conf_intent, 3),
                        pos_hits, neg_hits, obj_hits, round(bart_s, 3), objs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0], results[:5], caption

# ============================================================
# MAIN
# ============================================================
def main():
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    print(f"📚 Loaded {len(taxonomy['intents'])} intents.\n")
    img_path = input("📁 Enter image path: ").strip().strip('"')
    user_hint = input("💬 Enter optional text prompt (press Enter to skip): ").strip()
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    print("\n🔍 Running multimodal analysis ...\n")
    best, top5, caption = classify_intent(p, taxonomy, user_hint)

    print("=" * 100)
    print(f"🖼️  Image: {p.name}")
    print(f"🧾 Caption: {caption}")
    print("\n🏆 BEST MATCH")
    print(f"Intent: {best[0]} | Score={best[1]} | CLIP={best[2]} | +{best[3]}/-{best[4]} | obj_hits={best[5]} | BART={best[6]}")
    print("\n🔝 TOP 5 INTENTS:")
    for r in top5:
        print(f" - {r[0]} | Score={r[1]} | CLIP={r[2]} | +{r[3]}/-{r[4]} | obj_hits={r[5]} | BART={r[6]}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
