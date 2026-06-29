"""Systematic comparison harness for LoRA/config sweeps - the "measure the improvement" core.

It does NOT need a GPU. After each sweep run renders its FIXED prompts at FIXED seeds, point this at
the run folders and it builds, per prompt, a side-by-side contact sheet (one column per run, same
prompt+seed across columns) plus a config-diff table. That side-by-side at a fixed seed is the only
honest way to say config B beat config A, and it is exactly the before/after evidence the job asks
for.

Expected layout (you control the sample filenames; keep them stable across runs):
    runs/
      base/    p01.png p02.png p03.png   meta.json   # meta.json optional: the config knobs for this run
      rank32/  p01.png p02.png p03.png   meta.json
      lr_high/ p01.png ...
Same basename (p01.png) across runs MUST be the same prompt + seed.

Run (CPU, needs Pillow):
    ../../../.venv/bin/python compare.py --runs runs --out report
Produces report/<prompt>.png contact sheets + report/index.html + report/configs.md
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def list_runs(root):
    runs = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(d)):
            runs.append(name)
    return runs


def prompt_keys(root, runs):
    keys = set()
    for r in runs:
        for f in os.listdir(os.path.join(root, r)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                keys.add(os.path.splitext(f)[0])
    return sorted(keys)


def _img(root, run, key):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(root, run, key + ext)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    return None


def contact_sheet(root, runs, key, cell=320, label_h=28):
    cols = len(runs)
    sheet = Image.new("RGB", (cols * cell, cell + label_h), (245, 245, 247))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, run in enumerate(runs):
        im = _img(root, run, key)
        x = i * cell
        if im is not None:
            im = im.resize((cell, cell))
            sheet.paste(im, (x, label_h))
        d.rectangle([x, 0, x + cell - 1, label_h - 1], fill=(28, 30, 36))
        d.text((x + 6, 7), run[:40], fill=(255, 255, 255), font=font)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="folder with one subfolder of samples per run")
    ap.add_argument("--out", default="report")
    ap.add_argument("--cell", type=int, default=320)
    args = ap.parse_args()

    if not os.path.isdir(args.runs):
        raise SystemExit(f"no runs folder at {args.runs}; render each sweep run's fixed prompts into "
                         f"{args.runs}/<run>/<prompt>.png first (see this file's header)")
    runs = list_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run subfolders with images under {args.runs}")
    keys = prompt_keys(args.runs, runs)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for key in keys:
        sheet = contact_sheet(args.runs, runs, key, cell=args.cell)
        sheet.save(os.path.join(args.out, f"{key}.png"))
        rows.append(f'<h3>{key}</h3><img src="{key}.png" style="max-width:100%">')
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("<html><body style='font-family:sans-serif'><h2>LoRA sweep comparison</h2>"
                f"<p>runs (columns): {', '.join(runs)}</p>" + "\n".join(rows) + "</body></html>")

    # config-diff table from each run's optional meta.json
    lines = ["# Run configs\n", "| run | knobs (from meta.json) |", "|-----|------------------------|"]
    for r in runs:
        mp = os.path.join(args.runs, r, "meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        knobs = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "(no meta.json)"
        lines.append(f"| {r} | {knobs} |")
    with open(os.path.join(args.out, "configs.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[compare] {len(runs)} runs x {len(keys)} prompts")
    print(f"[compare] wrote {args.out}/index.html (open it), {len(keys)} contact sheet(s), and configs.md")
    print("[compare] read the columns at a FIXED seed: that side-by-side is your before/after evidence")


if __name__ == "__main__":
    main()
