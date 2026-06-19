"""
Module 6: build a tiny synthetic image-QA dataset.

Each image is one colored shape on a white background. The question asks either the
color or the shape, and the answer is a single word. No downloads, fully reproducible,
and the answer depends only on what is in the image, so it is a real vision task.

Outputs:
  data/images/*.png
  data/train.jsonl, data/eval.jsonl   (one {image, question, answer} per line)

Run from this folder:
    aipy make_data.py
"""

import json
import os
import random

from PIL import Image, ImageDraw

COLORS = {
    "red":    (220, 40, 40),
    "green":  (40, 180, 70),
    "blue":   (50, 90, 220),
    "yellow": (240, 210, 40),
}
SHAPES = ["circle", "square", "triangle"]
SIZE = 224


def draw(path, shape, color_name, rng):
    """Draw one shape with a little position/size jitter so no two images are identical."""
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    c = COLORS[color_name]
    m = rng.randint(35, 60)              # margin
    ox, oy = rng.randint(-15, 15), rng.randint(-15, 15)
    box = [m + ox, m + oy, SIZE - m + ox, SIZE - m + oy]
    if shape == "circle":
        d.ellipse(box, fill=c)
    elif shape == "square":
        d.rectangle(box, fill=c)
    else:  # triangle
        d.polygon([((box[0] + box[2]) // 2, box[1]), (box[0], box[3]), (box[2], box[3])],
                  fill=c)
    img.save(path)


def make_split(name, copies, seed):
    rng = random.Random(seed)
    os.makedirs("data/images", exist_ok=True)
    combos = [(s, c) for s in SHAPES for c in COLORS]   # 3 x 4 = 12 combinations
    samples = combos * copies
    rng.shuffle(samples)

    rows = []
    for i, (shape, color) in enumerate(samples, 1):
        fname = f"images/{name}_{i:03d}.png"
        draw(f"data/{fname}", shape, color, rng)
        if i % 2 == 0:
            q, a = "What color is the shape?", color
        else:
            q, a = "What shape is in the image?", shape
        rows.append({"image": fname, "question": q, "answer": a})

    with open(f"data/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} images -> data/{name}.jsonl")


make_split("train", copies=3, seed=0)    # 12 combos x 3 = 36 images
make_split("eval",  copies=1, seed=99)   # 12 held-out images (different seed/jitter)
print("Done. Next: aipy train.py")
