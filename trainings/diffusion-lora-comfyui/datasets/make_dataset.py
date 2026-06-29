"""Generate a small, self-contained LoRA training set, so this folder has its OWN data (nothing
imported from the other training folders).

It draws a consistent synthetic "concept" - a distinctive emblem (fixed shape + palette, the thing
the LoRA should learn) placed on varied backgrounds/positions - and writes matching caption files.
A simple synthetic concept is enough to learn the trainer mechanics end to end (it will visibly be
learned or not), which is the point for milestones 1-4. For a real portfolio LoRA, drop 15-30 of
your own images of one subject into datasets/raw/ and caption them the same way.

Output (image/caption pairs, the convention AI-Toolkit and Musubi both read):
    datasets/concept/ok_001.png + ok_001.txt ...

Run (CPU, needs Pillow):
    ../../../.venv/bin/python make_dataset.py
"""

import math
import os
import random

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "concept")
RAW = os.path.join(HERE, "raw")

TRIGGER = "tk-emblem"          # the unique token the LoRA binds the concept to (rare, non-English-word)
N = 24                         # 15-30 is a typical small-concept LoRA set
SIZE = 512
SEED = 0

BACKDROPS = [
    ("a plain studio backdrop", (236, 236, 240)),
    ("a dark slate background", (32, 34, 40)),
    ("a warm beige background", (224, 206, 178)),
    ("a deep teal background", (18, 78, 86)),
    ("a soft lavender background", (210, 200, 232)),
    ("a muted olive background", (120, 124, 86)),
]
EMBLEM = {"ring": (242, 96, 64), "core": (64, 132, 242), "spokes": (250, 210, 90)}  # the fixed identity


def draw_emblem(d, cx, cy, r):
    # fixed identity: orange ring, blue core, yellow spokes — consistent across every image
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=EMBLEM["ring"], width=max(4, r // 8))
    cr = int(r * 0.42)
    d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=EMBLEM["core"])
    for k in range(6):
        a = math.pi / 3 * k
        x1, y1 = cx + math.cos(a) * cr, cy + math.sin(a) * cr
        x2, y2 = cx + math.cos(a) * r, cy + math.sin(a) * r
        d.line([x1, y1, x2, y2], fill=EMBLEM["spokes"], width=max(3, r // 12))


def main():
    random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    for i in range(1, N + 1):
        desc, bg = random.choice(BACKDROPS)
        img = Image.new("RGB", (SIZE, SIZE), bg)
        d = ImageDraw.Draw(img)
        r = random.randint(90, 150)
        cx = random.randint(r + 20, SIZE - r - 20)
        cy = random.randint(r + 20, SIZE - r - 20)
        draw_emblem(d, cx, cy, r)
        name = f"ok_{i:03d}"
        img.save(os.path.join(OUT, name + ".png"))
        caption = f"a photo of {TRIGGER}, a circular emblem, on {desc}"
        with open(os.path.join(OUT, name + ".txt"), "w") as f:
            f.write(caption + "\n")

    with open(os.path.join(RAW, "README.md"), "w") as f:
        f.write(
            "# raw/ — your own images go here\n\n"
            "For a real LoRA, put 15-30 images of ONE subject here and a matching `<name>.txt` caption\n"
            "per image (same convention as ../concept/). Tips that matter for quality:\n\n"
            "- One clear subject, varied pose/background/lighting; avoid duplicates and watermarks.\n"
            "- Caption with a rare trigger token + a short, true description; vary the wording.\n"
            "- Keep resolution >= the training resolution (e.g. 1024) and crop to the subject.\n"
            "- Point the trainer config at this folder instead of ../concept when you are ready.\n")

    print(f"[dataset] wrote {N} image/caption pairs to datasets/concept (trigger token: '{TRIGGER}')")
    print(f"[dataset] sample caption: a photo of {TRIGGER}, a circular emblem, on a plain studio backdrop")
    print("[dataset] for a real LoRA, fill datasets/raw/ with your own subject (see datasets/raw/README.md)")


if __name__ == "__main__":
    main()
