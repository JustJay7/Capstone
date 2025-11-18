import os
from pathlib import Path
from PIL import Image
import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# ============================================================
# 1. FORCE ALL CACHES TO D:\
# ============================================================
ROOT_DIR = Path(r"D:\Capstone_research")

custom_clip_cache = ROOT_DIR / ".clip_cache"
custom_hf_cache = ROOT_DIR / ".huggingface"
custom_torch_cache = ROOT_DIR / ".torch_cache"
for p in [custom_clip_cache, custom_hf_cache, custom_torch_cache]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = str(custom_clip_cache)
os.environ["CLIP_CACHE_DIR"] = str(custom_clip_cache)
os.environ["HF_HOME"] = str(custom_hf_cache)
os.environ["TRANSFORMERS_CACHE"] = str(custom_hf_cache)
os.environ["TORCH_HOME"] = str(custom_torch_cache)

# ============================================================
# 2. PATHS
# ============================================================
TAXONOMY_PATH = ROOT_DIR / "configs" / "intent_taxonomy.json"
FLORENCE_MODEL_PATH = ROOT_DIR / "models" / "model weights" / "generalist model weights" / "florence-2"
SAMPLES_DIR = ROOT_DIR / "samples"
LOCAL_CLIP_PT = custom_clip_cache / "ViT-B-32.pt"

# ============================================================
# 3. LOAD TAXONOMY
# ============================================================
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

# ============================================================
# 4. LOAD MODELS
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")
print(f"✅ Forcing CLIP to load from {LOCAL_CLIP_PT}")

# Load CLIP manually from local cache only
model, preprocess = clip.load(
    "ViT-B/32",
    device=device,
    jit=False,
    download_root=str(custom_clip_cache)
)

# Double-check model file really came from D:
print(f"📦 CLIP model loaded from: {LOCAL_CLIP_PT if LOCAL_CLIP_PT.exists() else 'UNKNOWN'}")

# Load Florence (BLIP captioner)
print("⏳ Loading Florence model ...")
processor = BlipProcessor.from_pretrained(str(FLORENCE_MODEL_PATH), local_files_only=True)
florence = BlipForConditionalGeneration.from_pretrained(str(FLORENCE_MODEL_PATH), local_files_only=True).to(device)
print("✅ Florence model loaded.\n")

# ============================================================
# 5. FUNCTIONS
# ============================================================
def get_caption(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = florence.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.lower()

def get_clip_confidence(image_path, prompts):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return float(sims.max().cpu().numpy())

def classify_intent(image_path, user_hint=""):
    caption = get_caption(image_path)
    results = []

    for intent in taxonomy["intents"]:
        pol = intent["routing_policy"]
        clip_conf = get_clip_confidence(image_path, intent["clip_text_prompts"])
        pos_hits = sum(1 for kw in intent["florence_positive_markers"] if kw in caption)
        neg_hits = sum(1 for kw in intent["florence_negative_markers"] if kw in caption)
        score = clip_conf + 0.05 * pos_hits - 0.05 * neg_hits

        for rule in pol.get("boost_rules", []):
            if "if_hint_contains_any" in rule and any(k in user_hint.lower() for k in rule["if_hint_contains_any"]):
                score *= rule["boost_intent_by"]
            if "if_caption_contains_any" in rule and any(k in caption for k in rule["if_caption_contains_any"]):
                score *= rule["boost_intent_by"]

        results.append((intent["intent_id"], round(score, 3), round(clip_conf, 3), pos_hits, neg_hits))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[0], results[:5]

# ============================================================
# 6. MAIN TEST
# ============================================================
if __name__ == "__main__":
    TEST_IMAGES = [
        SAMPLES_DIR / "road_crack_test.jpg",
        SAMPLES_DIR / "Miami-Dade-Fire-Rescue.jpg",
        SAMPLES_DIR / "Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg"
    ]

    for img_path in TEST_IMAGES:
        print("=" * 80)
        print(f"🖼️  Testing image: {img_path.name}")
        best, top5 = classify_intent(img_path, user_hint="automated classification")

        print("\n🏆 BEST INTENT DETECTED:")
        print(f"Intent ID: {best[0]}\nScore: {best[1]}\nCLIP Conf: {best[2]}\nPos Hits: {best[3]} | Neg Hits: {best[4]}")

        print("\n🔝 TOP 5 INTENTS:")
        for r in top5:
            print(f" - {r[0]} | Score={r[1]} | CLIP={r[2]} | +{r[3]}/-{r[4]}")
        print("\n")

    print("=" * 80)
    print("✅ All tests completed.")
