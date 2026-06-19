"""
Module 5, step 2: merge the fine-tuned model back with its base at several weights.

We use MergeKit's `linear` method, which is a weighted average of the two models'
weights. With normalize on (the default), giving the fine-tune weight `alpha` and the
base weight `1 - alpha` produces a clean interpolation:

    merged = alpha * fine_tuned + (1 - alpha) * base

alpha = 0 is the pure base, alpha = 1 is the pure fine-tune, and values in between are
blends. We build one merge per alpha so evaluate.py can show the whole curve.

Compatibility note: mergekit 0.1.4 leaves a few of its pydantic models with unresolved
forward references under transformers 5.x (it crashes with "ConfiguredModuleArchitecture
is not fully defined"). _patch_mergekit() rebuilds those models with torch in scope to
fix it, and we drive the merge through mergekit's Python API so the patch is in effect
(the mergekit-yaml CLI would spawn a fresh, unpatched process).

Run from this folder (after prepare_finetuned.py):
    aipy merge.py
"""

import os
import shutil

import torch
import transformers
import mergekit.architecture.base as _arch_base
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FT_DIR  = "./finetuned-full"
ALPHAS  = [0.25, 0.50, 0.75]      # weight on the fine-tune


def _patch_mergekit():
    """Resolve mergekit's deferred pydantic models (the torch forward ref)."""
    ns = {"torch": torch, "transformers": transformers,
          "PretrainedConfig": transformers.PretrainedConfig}
    for cls_name in ("WeightInfo", "ModuleDefinition", "ModelArchitecture",
                     "ConfiguredModuleArchitecture", "ConfiguredModelArchitecture"):
        cls = getattr(_arch_base, cls_name, None)
        if cls is not None:
            try:
                cls.model_rebuild(force=True, _types_namespace=ns)
            except Exception:
                pass


def merge(alpha, out_dir):
    cfg = MergeConfiguration.model_validate({
        "merge_method": "linear",
        "dtype": "float16",
        "models": [
            {"model": FT_DIR,  "parameters": {"weight": alpha}},
            {"model": BASE_ID, "parameters": {"weight": round(1 - alpha, 2)}},
        ],
    })
    run_merge(cfg, out_path=out_dir, options=MergeOptions())
    # MergeKit copies the tokenizer but not the chat template; bring it over so the
    # merged model can be prompted with apply_chat_template like base and finetuned.
    src = os.path.join(FT_DIR, "chat_template.jinja")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out_dir, "chat_template.jinja"))


_patch_mergekit()
for alpha in ALPHAS:
    tag = f"{int(alpha * 100):02d}"
    out_dir = f"./merged-{tag}"
    print(f"\n=== merging alpha={alpha} (fine-tune {tag}%) -> {out_dir} ===")
    merge(alpha, out_dir)

print("\nDone. Merged models in ./merged-25, ./merged-50, ./merged-75. "
      "Next: aipy evaluate.py")
