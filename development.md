# Development notes

How to run things in this repo on my machine, and the environment gotchas I keep
hitting so I do not rediscover them every time.

## Running scripts

Use the project venv at `.venv`. The catch is that `python` on this machine does not
reliably point at the venv (see pyenv note below), so call the venv python by path or
use the alias.

By path, from a module folder:

```bash
cd 02_dataset_from_scratch
../.venv/bin/python train.py
../.venv/bin/python test_base.py
../.venv/bin/python test_adapter.py
```

Or with the `aipy` alias (added to ~/.bashrc):

```bash
alias aipy="/home/ianrussel/projects/ait-lab/ai_training/.venv/bin/python"
```

```bash
source ~/.bashrc      # once, to load the alias
aipy train.py
```

Quick sanity check that it is the right interpreter and the GPU is visible:

```bash
aipy -c "import torch; print(torch.cuda.is_available())"   # want True
```

## Why `source .venv/bin/activate` does not stick

`~/.bashrc` runs pyenv hooks (`pyenv init -`, `pyenv init --path`,
`pyenv virtualenv-init -`). Their prompt hook re-prepends pyenv's shims to the front of
PATH on every prompt, so even after activating the venv, `python` resolves back to
`~/.pyenv/shims/python` (the pyenv global Python, which has none of this project's
packages). That is why a plain `python script.py` fails with ModuleNotFoundError.

Workaround: always call `.venv/bin/python` by path, or use `aipy`. The shims cannot
intercept an explicit path. The VS Code Run button is fine because the interpreter is
set to the absolute `.venv/bin/python` in `.vscode/settings.json`.

## `.venv/bin/pip: bad interpreter`

This venv was created when the folder was named `fine_tuning`, then the folder was
renamed to `ai_training`. Venv console scripts (`pip`, etc.) bake the absolute
interpreter path into their shebang line, so `.venv/bin/pip` still points at the old
`.../fine_tuning/.venv/bin/python` and fails with "bad interpreter". `.venv/bin/python`
itself is a symlink, so it still works.

Fixed: the shebangs in .venv/bin/* were rewritten from the old /fine_tuning/ path to
/ai_training/ in place, so `pip` and the other console scripts work again. The command,
in case it ever needs redoing after a move:

```bash
grep -rlI '/fine_tuning/.venv/bin/python' .venv/bin \
  | xargs sed -i 's#/fine_tuning/.venv/bin/python#/ai_training/.venv/bin/python#g'
```

If the venv is ever moved again, either rerun that with the new path, or just use the
move-proof form `.venv/bin/python -m pip install <package>` (the python symlink does not
care about the path).

## GPU / torch pinning

- GPU: RTX 3050 Ti Laptop, 4 GB VRAM.
- Driver: 535, supports CUDA 12.2 max.
- Working torch: 2.6.0+cu124 (cu124 runs fine on a 12.2 driver).

Do NOT run a plain `pip install torch`. It grabs the newest wheel (cu130), which is too
new for the 535 driver and makes `torch.cuda.is_available()` return False. requirements.txt
pins `torch==2.6.0+cu124` with the matching extra index so a reinstall stays correct.

If the GPU "disappears" after a reinstall, check the torch build:

```bash
aipy -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If it shows a cu13x build, reinstall the right one:

```bash
.venv/bin/pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
```

## Rebuilding the venv from scratch

Always pass the `.venv` target, or the env lands in the repo root and mixes with code.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Training on Kaggle

For real training I usually use a Kaggle GPU notebook (T4). Notes:

- Switch bf16 to fp16 on a T4 (it is bad at bf16). The module 2 train.py already picks
  fp16 on GPU.
- Install the stack and the companion packages together, then restart the kernel, or
  imports fail with version-mismatch errors (for example is_torch_neuron_available or an
  incompatible torchao). Restart is the step that is easy to forget.
