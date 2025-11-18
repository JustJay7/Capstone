import os
from glob import glob

# ✅ Define base directory and image splits
base_dir = "data/raw/rdd2022/datasets"
splits = {
    "train": os.path.join(base_dir, "train", "images"),
    "val": os.path.join(base_dir, "valid", "images"),
    "test": os.path.join(base_dir, "test", "images"),
}

# ✅ Output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# ✅ Process each split
for split_name, img_dir in splits.items():
    # Accept both .jpg and .JPG extensions
    img_paths = sorted(
        glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.JPG"))
    )
    # Create clean, forward-slash relative paths
    rel_paths = [os.path.relpath(p, start=".").replace("\\", "/") for p in img_paths]

    # Save to .txt
    out_file = os.path.join(output_dir, f"rdd2022_{split_name}.txt")
    with open(out_file, 'w') as f:
        f.write("\n".join(rel_paths))

    print(f"✅ {split_name.title()} split saved to {out_file} ({len(rel_paths)} entries)")
