"""Package initializer for the eval package.

Keep this module minimal: expose submodule names but avoid importing
submodules at import time to prevent circular imports when running
scripts inside the package (for example `python eval/EvalCoco.py`).
"""

# Public submodules (no eager imports here)
__all__ = [
    'common', 'EvalCoco', 'EvalAuc', 'EvalAblations'
]

# Helpers to lazily import submodules when accessed
import importlib

def __getattr__(name):
    """Lazily import submodules like `eval.common` when attribute accessed.

    This avoids executing package-level imports in scripts that run
    from inside the `eval` directory.
    """
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

