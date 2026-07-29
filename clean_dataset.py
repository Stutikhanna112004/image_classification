"""
Scans dataset/cats and dataset/dogs, and for every image:
  - tries to open it with PIL
  - converts it to proper RGB (fixes grayscale, grayscale+alpha, palette, CMYK, etc.)
  - overwrites it as a clean RGB JPEG
  - deletes it if it's corrupt / unreadable / not actually an image

Run this once before training:
    python clean_dataset.py
"""

import os
from PIL import Image

DATASET_DIR = "dataset"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

removed = 0
fixed = 0
checked = 0

for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)

        if not fname.lower().endswith(VALID_EXTENSIONS):
            print(f"Removing non-image file: {fpath}")
            os.remove(fpath)
            removed += 1
            continue

        checked += 1
        try:
            with Image.open(fpath) as img:
                img.verify()  # checks the file isn't truncated/corrupt

            # re-open (verify() consumes the file handle) and force RGB
            with Image.open(fpath) as img:
                if img.mode != "RGB":
                    rgb_img = img.convert("RGB")
                else:
                    rgb_img = img.copy()

            # save as a clean .jpg, always with a consistent extension
            base, _ = os.path.splitext(fpath)
            new_path = base + ".jpg"
            rgb_img.save(new_path, "JPEG")

            # if the extension changed (e.g. .png -> .jpg), remove the old file
            if new_path != fpath:
                os.remove(fpath)

            fixed += 1

        except Exception as e:
            print(f"Removing corrupt/unreadable file: {fpath} ({e})")
            try:
                os.remove(fpath)
            except OSError:
                pass
            removed += 1

print("\nDone.")
print(f"Checked: {checked}")
print(f"Fixed/normalized to RGB jpg: {fixed}")
print(f"Removed (corrupt or invalid): {removed}")
