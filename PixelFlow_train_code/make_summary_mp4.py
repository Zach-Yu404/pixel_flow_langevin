"""Compose exp PNGs into a summary.mp4 with class + cfg titles."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


CLASS_NAMES = {
    0: "AXFLAIR",
    1: "AXT1POST",
    2: "AXT1PRE",
    3: "AXT1",
    4: "AXT2",
    5: "null/uncond",
}


def build_frame(imgs, title, cell=256, title_h=48):
    imgs = [im.resize((cell, cell), Image.BILINEAR).convert("L") for im in imgs]
    grid = Image.new("L", (cell * 2, cell * 2), 0)
    for i, im in enumerate(imgs):
        r, c = divmod(i, 2)
        grid.paste(im, (c * cell, r * cell))
    frame = Image.new("RGB", (cell * 2, cell * 2 + title_h), (0, 0, 0))
    frame.paste(grid.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((cell * 2 - tw) // 2, (title_h - th) // 2 - 2),
              title, fill=(255, 255, 255), font=font)
    return np.array(frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="exp0002", help="folder with PNGs")
    p.add_argument("--prefix", default="384", help="resolution prefix in filenames")
    p.add_argument("--out", default=None, help="output mp4 path")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--cell", type=int, default=256)
    p.add_argument("--num_examples", type=int, default=4)
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--cfgs", type=int, default=5)
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src / "summary.mp4"

    frames = []
    for cls in range(args.classes):
        for cfg in range(args.cfgs):
            imgs = []
            for idx in range(args.num_examples):
                f = src / f"{args.prefix}_{idx}_class{cls}_cfg{cfg}.png"
                imgs.append(Image.open(f))
            name = CLASS_NAMES.get(cls, f"class{cls}")
            title = f"Class {cls} ({name})  |  CFG = {cfg}"
            frames.append(build_frame(imgs, title, cell=args.cell))
            print(f"frame cls={cls} cfg={cfg}")

    with imageio.get_writer(str(out), fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
