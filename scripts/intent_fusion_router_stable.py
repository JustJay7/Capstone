# ============================================================
# 4-STREAM INTENT ROUTER (FINAL, DOMAIN-AWARE, HARD-GATES)
# YOLO(generic) + Florence(caption markers) + CLIP + BART
# - BART premise = user hint else Florence caption
# - Domain-adaptive fusion weights
# - Hard excludes per intent (from taxonomy or built-in heuristics)
# - Clear per-modality contributions in output
# ============================================================

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

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

# ----------------------------
# PATHS
# ----------------------------
ROOT = Path("/Volumes/Expansion/Models")
CLIP_CACHE = ROOT / ".clip_cache"
HF_CACHE = ROOT / ".huggingface"
TORCH_HOME = ROOT / ".torch_cache"

TAXONOMY_JSON = ROOT / "configs" / "gpt.json"
FLORENCE_DIR  = ROOT / "models" / "generalist" / "florence-2"
CLIP_PT       = ROOT / "models" / "generalist" / "clip" / "pytorch_model.bin"
YOLO_PT       = ROOT / "models" / "object" / "yolov8" / "yolov8s.pt"

BART_BASE = ROOT / "models" / "generalist" / "bart-large-mnli" / "models--facebook--bart-large-mnli" / "snapshots"

# ----------------------------
# ENV + VALIDATION
# ----------------------------
for p in [CLIP_CACHE, HF_CACHE, TORCH_HOME]:
    p.mkdir(parents=True, exist_ok=True)

os.environ.update({
    "HF_HOME": str(HF_CACHE),
    "TRANSFORMERS_CACHE": str(HF_CACHE),
    "HUGGINGFACE_HUB_CACHE": str(HF_CACHE),
    "TORCH_HOME": str(TORCH_HOME),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "XDG_CACHE_HOME": str(ROOT),
    "CLIP_CACHE_DIR": str(CLIP_CACHE),
})

def _must(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

_must(TAXONOMY_JSON, "taxonomy JSON")
_must(FLORENCE_DIR,  "Florence (BLIP) dir")
_must(CLIP_PT,       "CLIP .pt")
_must(YOLO_PT,       "YOLO weights")
_must(BART_BASE,     "BART snapshot root")

snapshots = [d for d in BART_BASE.iterdir() if d.is_dir()]
if not snapshots:
    raise FileNotFoundError(f"No BART snapshot folders found in {BART_BASE}")
BART_DIR = snapshots[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Device: {device}")

# ----------------------------
# LOAD MODELS
# ----------------------------
print(f"⏳ Loading CLIP from {CLIP_PT}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False,
                                        download_root=str(CLIP_CACHE))

print("⏳ Loading Florence (BLIP) locally ...")
processor = BlipProcessor.from_pretrained(str(FLORENCE_DIR), local_files_only=True)
florence  = BlipForConditionalGeneration.from_pretrained(str(FLORENCE_DIR),
                                                         local_files_only=True).to(device).eval()
print("✅ Florence ready.")

print(f"⏳ Loading YOLO (generic COCO) ...")
yolo_general = YOLO(str(YOLO_PT))
print("✅ YOLO (generic) ready.")

print(f"⏳ Loading BART-MNLI from {BART_DIR}")
tokenizer  = AutoTokenizer.from_pretrained(str(BART_DIR), local_files_only=True)
bart_model = AutoModelForSequenceClassification.from_pretrained(str(BART_DIR),
                                                                local_files_only=True).to(device).eval()
print("✅ BART ready.")

# ----------------------------
# FUSION PRESETS (adaptive)
# ----------------------------
GENERAL_FUSION   = {"clip": 0.35, "florence": 0.30, "yolo": 0.20, "bart": 0.15}
MEDICAL_FUSION   = {"clip": 0.25, "florence": 0.35, "yolo": 0.00, "bart": 0.40}
DISASTER_FUSION  = {"clip": 0.30, "florence": 0.35, "yolo": 0.25, "bart": 0.10}
NATURE_FUSION    = {"clip": 0.35, "florence": 0.25, "yolo": 0.25, "bart": 0.15}

def choose_fusion_weights(caption: str, objs: List[str]) -> Dict[str,float]:
    cap = caption.lower()
    oset = set(objs)
    if any(x in cap for x in ["x-ray", "radiograph", "thoracic", "chest", "lung", "dicom"]):
        return MEDICAL_FUSION
    if any(x in cap for x in ["collapsed", "rubble", "debris", "earthquake", "disaster", "destroyed"]) \
       or any(x in oset for x in ["helmet", "truck", "person"]) and "debris" in cap:
        return DISASTER_FUSION
    if "bird" in oset or any(x in cap for x in ["bird", "animal", "wildlife"]):
        return NATURE_FUSION
    return GENERAL_FUSION

# ----------------------------
# HELPERS
# ----------------------------
def get_caption(image_path: Path, max_new_tokens: int = 25) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = florence.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True).lower().strip()

def get_clip_conf(image_path: Path, prompts: List[str]) -> float:
    if not prompts:
        return 0.0
    img = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    txt = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        im = clip_model.encode_image(img)
        tx = clip_model.encode_text(txt)
        im = im / im.norm(dim=-1, keepdim=True)
        tx = tx / tx.norm(dim=-1, keepdim=True)
        sims = (100.0 * im @ tx.T).softmax(dim=-1)
    return float(sims.max().detach().cpu().numpy())

def get_objects(image_path: Path, conf: float = 0.35) -> List[str]:
    res = yolo_general.predict(source=str(image_path), conf=conf, imgsz=640, verbose=False)
    names = res[0].names
    classes = res[0].boxes.cls.detach().cpu().tolist() if res and res[0].boxes is not None else []
    return list({names[int(c)] for c in classes})

def bart_scores_independent(intents: list, premise_text: str) -> dict:
    labels = [i.get("bart_label", i["intent_id"]) for i in intents]
    descs  = [i.get("bart_description", i.get("description", i["intent_id"])) for i in intents]
    hypos  = [f"This image represents {d}." for d in descs]
    with torch.no_grad():
        inputs = tokenizer([premise_text] * len(hypos), hypos,
                           truncation=True, padding=True, return_tensors="pt").to(device)
        logits = bart_model(**inputs).logits
        entail = logits[:, 2]
        probs = torch.softmax(entail, dim=0)
    return {lbl: float(p) for lbl, p in zip(labels, probs)}

def apply_rule_boosts(score: float, caption: str, hint: str, intent: dict) -> float:
    rp = intent.get("routing_policy", {}) or {}
    boosts = rp.get("boost_rules", []) or []
    negs   = rp.get("negative_boost_rules", []) or []
    s = score
    cap = caption.lower()
    hint_l = (hint or "").lower()

    for rule in boosts:
        if "if_hint_contains_any" in rule and any(k.lower() in hint_l for k in rule["if_hint_contains_any"]):
            s *= float(rule.get("boost_intent_by", 1.0))
        if "if_caption_contains_any" in rule and any(k.lower() in cap for k in rule["if_caption_contains_any"]):
            s *= float(rule.get("boost_intent_by", 1.0))

    for rule in negs:
        if "if_caption_contains_any" in rule and any(k.lower() in cap for k in rule["if_caption_contains_any"]):
            s /= max(1e-6, float(rule.get("reduce_intent_by", 1.0)))

    # built-in medical bias if caption screams radiograph
    if intent.get("domain", "") == "medical" and any(x in cap for x in ["x-ray", "radiograph", "thoracic", "lung", "chest"]):
        s *= 1.5
    return s

def hard_excluded(intent: dict, caption: str, objs: List[str]) -> bool:
    cap = caption.lower()
    oset = set(objs)
    # taxonomy-provided hard_excludes (optional)
    texcl = (intent.get("hard_excludes") or {})
    cap_ex = [w.lower() for w in texcl.get("caption_contains_any", [])]
    obj_ex = [w.lower() for w in texcl.get("object_contains_any", [])]
    if cap_ex and any(w in cap for w in cap_ex):
        return True
    if obj_ex and any(w in oset for w in obj_ex):
        return True

    # built-in sanity: birds/animals should kill infrastructure-crack/post-disaster, unless the intent is nature-ecology
    if any(w in oset for w in ["bird", "cat", "dog", "animal"]):
        if intent.get("intent_id","").startswith(("infrastructure-", "post-disaster-", "industrial-")):
            return True
    return False

def fuse_modality(clip_conf: float, pos_hits: int, neg_hits: int, obj_hits: int, bart: float, weights: Dict[str,float]) -> Dict[str, float]:
    # Normalize florence hits contribution, don't nuke with negatives below zero.
    flor_term = max(pos_hits - neg_hits, 0)
    contrib = {
        "clip":     clip_conf * weights["clip"],
        "florence": flor_term * weights["florence"] * 0.05,
        "yolo":     obj_hits  * weights["yolo"]     * 0.05,
        "bart":     bart      * weights["bart"],
    }
    contrib["total"] = sum(contrib.values())
    return contrib

# ----------------------------
# CLASSIFICATION
# ----------------------------
def classify_intent(image_path: Path, taxonomy: dict, user_hint: str = "") -> Tuple[tuple, List[tuple], str, List[str]]:
    # 1) Caption + objects
    caption = get_caption(image_path)
    objs    = get_objects(image_path)

    # 2) BART premise = user hint else Florence caption else neutral
    premise = user_hint.strip() or caption or "This is an image."
    bart_all = bart_scores_independent(taxonomy["intents"], premise)

    # 3) Domain-adaptive fusion weights
    base_weights = choose_fusion_weights(caption, objs)

    # 4) Score each intent
    results = []
    for intent in taxonomy["intents"]:
        iid      = intent["intent_id"]
        prompts  = intent.get("clip_text_prompts", []) or []
        pos_m    = intent.get("florence_positive_markers", []) or []
        neg_m    = intent.get("florence_negative_markers", []) or []
        obj_m    = [o.lower() for o in intent.get("object_markers", []) or []]
        custom_w = (intent.get("routing_policy", {}) or {}).get("fusion_weights", {}) or {}

        # start with domain weights, then allow per-intent overrides
        weights = dict(base_weights)
        weights.update(custom_w)

        # hard excludes
        if hard_excluded(intent, caption, [o.lower() for o in objs]):
            contrib = {"clip":0.0,"florence":0.0,"yolo":0.0,"bart":0.0,"total":-1e9}
            results.append((iid, contrib, prompts, pos_m, neg_m, obj_m))
            continue

        clip_conf = get_clip_conf(image_path, prompts) if prompts else 0.0
        cap_low   = caption.lower()
        pos_hits  = sum(1 for k in pos_m if k.lower() in cap_low)
        neg_hits  = sum(1 for k in neg_m if k.lower() in cap_low)
        obj_hits  = sum(1 for k in obj_m if k in [o.lower() for o in objs])

        bart_lbl  = intent.get("bart_label", iid)
        bart_val  = bart_all.get(bart_lbl, 0.0)

        contrib = fuse_modality(clip_conf, pos_hits, neg_hits, obj_hits, bart_val, weights)
        # apply boosts/penalties
        boosted_total = apply_rule_boosts(contrib["total"], caption, user_hint, intent)

        # stash augmented total but keep modality breakdown
        contrib["total"] = boosted_total

        results.append((iid, contrib, prompts, pos_m, neg_m, obj_m))

    # 5) Rank
    results.sort(key=lambda r: r[1]["total"], reverse=True)

    # Pack top-5 summary tuples for printing
    top5_print = []
    for iid, c, prompts, pos_m, neg_m, obj_m in results[:5]:
        top5_print.append((
            iid, round(c["total"], 3),
            round(c["clip"], 3),
            int(sum(1 for k in pos_m if k.lower() in caption.lower())),
            int(sum(1 for k in neg_m if k.lower() in caption.lower())),
            int(sum(1 for k in obj_m if k in [o.lower() for o in objs])),
            round(c["bart"], 3)
        ))

    # best item in same format
    best_iid, best_c, _, pos_m0, neg_m0, obj_m0 = results[0]
    best_tuple = (
        best_iid, round(best_c["total"],3),
        round(best_c["clip"],3),
        int(sum(1 for k in pos_m0 if k.lower() in caption.lower())),
        int(sum(1 for k in neg_m0 if k.lower() in caption.lower())),
        int(sum(1 for k in obj_m0 if k in [o.lower() for o in objs])),
        round(best_c["bart"],3)
    )

    return best_tuple, top5_print, caption, objs

# ----------------------------
# MAIN
# ----------------------------
def main():
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    print(f"📚 Loaded {len(taxonomy.get('intents', []))} intents from taxonomy.\n")

    while True:
        img_path = input("📁 Enter image path (or 'exit' to quit): ").strip().strip('"')
        if not img_path or img_path.lower() in {"exit", "quit", "q"}:
            print("👋 Exiting inference loop.")
            break

        user_hint = input("💬 Enter optional text hint for BART (press Enter to skip): ").strip()
        p = Path(img_path)

        if not p.exists():
            print(f"❌ Image not found: {p}\n")
            continue

        print("\n🔍 Running multimodal analysis ...\n")
        best, top5, caption, objs = classify_intent(p, taxonomy, user_hint)

        print("=" * 110)
        print(f"🖼️  Image: {p.name}")
        print(f"🧾 Caption → {caption}")
        print(f"🧷 YOLO objects → {objs}")
        print("-" * 110)

        print("\n🏆 BEST MATCH")
        print(f"Intent: {best[0]} | Score={best[1]} | CLIP={best[2]} | +{best[3]}/-{best[4]} | obj_hits={best[5]} | BART={best[6]}")

        print("\n🔝 TOP 5 INTENTS (fused score | per-modality)")
        for r in top5:
            print(f" - {r[0]} | Score={r[1]} | CLIP={r[2]} | +{r[3]}/-{r[4]} | obj_hits={r[5]} | BART={r[6]}")

        print("\nℹ️ Notes:")
        print(" - CLIP = image-text similarity against each intent's prompts.")
        print(" - Florence = caption keyword matches (+pos/-neg).")
        print(" - YOLO = object markers matched (intent.object_markers vs detected objects).")
        print(" - BART = NLI entailment using (user hint else caption) vs each intent description.")
        print(" - Domain-adaptive weights applied automatically from caption/objects.")
        print("\n✅ Done.\n")
        print("=" * 110)
if __name__ == "__main__":
    main()
