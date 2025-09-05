from medsegbench import Promise12MSBench
from PIL import Image
import numpy as np
import os

splits = ["train", "val", "test"]

for split in splits:
    dataset = Promise12MSBench(split=split, download=True)

    img_dir = f"data/Promise12MSBench/{split}/images"
    msk_dir = f"data/Promise12MSBench/{split}/masks"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for i in range(len(dataset)):
        img, msk = dataset[i]

        img = np.array(img)
        msk = np.array(msk)

        # Normalize to [0, 255] and convert to uint8
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        msk = ((msk - msk.min()) / (msk.max() - msk.min()) * 255).astype(np.uint8)

        Image.fromarray(img).save(f"{img_dir}/{split}_image_{i:04d}.png")
        Image.fromarray(msk).save(f"{msk_dir}/{split}_mask_{i:04d}.png")

    print(f"Saved {len(dataset)} image/mask pairs for split '{split}'")
