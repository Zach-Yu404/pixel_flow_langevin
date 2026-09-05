"""Demo images (ImageNet-val crops with class labels) -- identical preprocessing to the reference (center_crop_arr)."""
import json
import os
import numpy as np
import torch
from PIL import Image


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def load_demo_images(demo_dir, resolution=256):
    """[{short_name, class_idx, gt [3,H,W] in [-1,1], file}] from demo_dir/labels.json."""
    labels = json.load(open(os.path.join(demo_dir, "labels.json")))
    out = []
    for s in labels["samples"]:
        pil = Image.open(os.path.join(demo_dir, s["file"])).convert("RGB")
        arr = center_crop_arr(pil, resolution)
        t = torch.from_numpy(np.array(arr).copy()).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0
        short_name = s.get("short_name") or os.path.splitext(s["file"])[0]
        out.append(dict(short_name=short_name, class_idx=int(s["class_idx"]), gt=t, file=s["file"]))
    return out
