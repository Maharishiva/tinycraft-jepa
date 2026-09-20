"""Import a tinywmcraft training module from its own directory, ready to be called inside another jit.

Two details differ between generations of the repo and both break a naive import: every generation ships its own
kernels.py next to the training module (the cached one must go, or a checkpoint runs on another generation's
kernels), and the September module jits its encoder and decoder, which flax cannot nest inside the rollout's own
jit (it rebuilds the parameters one trace level down), so those wrappers are unwrapped."""
from __future__ import annotations

import importlib
import sys


def load(module_dir, module):
    sys.path.insert(0, str(module_dir))
    sys.modules.pop("kernels", None)
    m = importlib.import_module(module)
    for name in ("_encode", "_decode"):
        fn = vars(m).get(name)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(m, name, fn.__wrapped__)
    return m
