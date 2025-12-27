import os
import zipfile
import random
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import pydicom
from tqdm import tqdm

# ---------- USER CONFIG ----------
KAGGLE_DATASET = "zhangweiled/lidcidri"   # change if you prefer another Kaggle slug
DOWNLOAD_DIR = Path("raw_kaggle")
EXTRACT_DIR = Path("extracted")
OUTPUT_DIR = Path("output_dataset")
TARGET_COUNT = 3000   # total images desired
POS_LABEL_DIR_NAMES = ["nodule", "mask", "positive", "nodules"]  # heuristics
NEG_LABEL_DIR_NAMES = ["non_nodule", "negative", "background", "no_nodule"]
# ---------------------------------

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)

def download_kaggle_dataset(slug, dest):
    dest.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"])

def find_images_and_labels(base_dir):
    """
    Heuristic: search for image files (.dcm, .png, .jpg) and try to infer labels
    based on parent folder names or mask files. Returns list of (path, label) where label in {1,0}
    """
    images = []
    for p in base_dir.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".dcm", ".nii", ".nii.gz"]:
            parent = p.parent.name.lower()
            label = None
            for k in POS_LABEL_DIR_NAMES:
                if k in parent:
                    label = 1
                    break
            for k in NEG_LABEL_DIR_NAMES:
                if k in parent:
                    label = 0
                    break
            if label is None:
                label = -1
            images.append((p, label))
    return images

def convert_dcm_to_png(dcm_path, out_path):
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr - np.min(arr)
        if np.max(arr) != 0:
            arr = arr / np.max(arr)
        arr = (arr * 255.0).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(out_path, format="PNG")
        return True
    except Exception as e:
        print("DICOM convert error:", e, dcm_path)
        return False

def ensure_png(path_in, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path_in.suffix.lower()
    if suffix == ".dcm":
        return convert_dcm_to_png(path_in, dest_path)
    elif suffix in [".png", ".jpg", ".jpeg"]:
        shutil.copy2(path_in, dest_path)
        return True
    else:
        return False

def build_dataset(raw_dir, extracted_dir, output_dir, target_count):
    imgs = find_images_and_labels(raw_dir)
    print(f"Found {len(imgs)} candidate image files (label -1 = unknown).")

    pos = [p for p, l in imgs if l == 1]
    neg = [p for p, l in imgs if l == 0]
    unk = [p for p, l in imgs if l == -1]

    print(f"Positive candidates: {len(pos)}, Negative candidates: {len(neg)}, Unknown: {len(unk)}")

    needed_per_class = target_count // 2
    selected_pos = pos[:needed_per_class] if len(pos) >= needed_per_class else pos
    selected_neg = neg[:needed_per_class] if len(neg) >= needed_per_class else neg

    while len(selected_pos) < needed_per_class and unk:
        selected_pos.append(unk.pop())
    while len(selected_neg) < needed_per_class and unk:
        selected_neg.append(unk.pop())

    total_selected = len(selected_pos) + len(selected_neg)
    if total_selected < target_count:
        remaining = [p for p, l in imgs if p not in selected_pos and p not in selected_neg]
        to_add = min(len(remaining), target_count - total_selected)
        selected_neg.extend(random.sample(remaining, to_add))

    print(f"Selected Pos: {len(selected_pos)}, Neg: {len(selected_neg)}, Total: {len(selected_pos) + len(selected_neg)}")

    out_pos_dir = output_dir / "cancer"
    out_neg_dir = output_dir / "non_cancer"
    out_pos_dir.mkdir(parents=True, exist_ok=True)
    out_neg_dir.mkdir(parents=True, exist_ok=True)

    def write_list(file_list, out_dir, prefix):
        for idx, p in enumerate(tqdm(file_list, desc=f"Writing {prefix}")):
            out_path = out_dir / f"{prefix}_{idx:05d}.png"
            ok = ensure_png(p, out_path)
            if not ok:
                print("Skipped:", p)

    write_list(selected_pos, out_pos_dir, "cancer")
    write_list(selected_neg, out_neg_dir, "noncancer")

    print("Done. Output at:", output_dir.resolve())

if __name__ == "__main__":
    random.seed(42)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Downloading Kaggle dataset (may take a while)...")
    download_kaggle_dataset(KAGGLE_DATASET, DOWNLOAD_DIR)

    # Extract all zip files
    for z in DOWNLOAD_DIR.glob("*.zip"):
        print("Extracting:", z)
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(EXTRACT_DIR)

def safe_copytree(src, dest):
    """
    Windows-safe copy avoiding long path errors and skipping problematic files.
    """
    src_path = f"\\\\?\\{os.path.abspath(src)}"
    dest_path = f"\\\\?\\{os.path.abspath(dest)}"

    if os.path.exists(dest):
        shutil.rmtree(dest)

    def safe_copy(src_file, dest_file):
        try:
            shutil.copy2(src_file, dest_file)
        except Exception as e:
            print("⚠️ Skipped:", src_file, "Error:", e)

    # Walk manually instead of recursive copytree
    for root, dirs, files in os.walk(src_path):
        rel_path = os.path.relpath(root, src_path)
        dest_root = os.path.join(dest_path, rel_path)
        os.makedirs(dest_root, exist_ok=True)
        for f in files:
            safe_copy(os.path.join(root, f), os.path.join(dest_root, f))

for d in DOWNLOAD_DIR.iterdir():
    if d.is_dir():
        src = d
        dest = EXTRACT_DIR / d.name
        safe_copytree(src, dest)

    print("2) Building dataset...")
    build_dataset(EXTRACT_DIR, EXTRACT_DIR, OUTPUT_DIR, TARGET_COUNT)

    print("All done. You'll find roughly", TARGET_COUNT, "images (balanced) in", OUTPUT_DIR)
