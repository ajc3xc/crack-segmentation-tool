import importlib
import os

_SUBMODULES = [
    "plot_metrics",
    "save_load_files",
    "legacy",
    "metrics",  # keep this last
]

pkg_name = __name__

for name in _SUBMODULES:
    mod = importlib.import_module(f"{pkg_name}.{name}")
    for k, v in mod.__dict__.items():
        if not k.startswith("__"):
            globals()[k] = v
    print(f"[{pkg_name}] merged {name}")

print(f"[{pkg_name}] flatten complete")
