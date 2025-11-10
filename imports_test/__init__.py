import importlib, pkgutil, sys, os

pkg_name = __name__
pkg_dir = os.path.dirname(__file__)

# auto import every .py file under this package
for loader, mod_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
    full_name = f"{pkg_name}.{mod_name}"
    try:
        mod = importlib.import_module(full_name)
        for k, v in mod.__dict__.items():
            if not k.startswith("__"):
                globals()[k] = v
        print(f"[{pkg_name}] merged {mod_name}")
    except ModuleNotFoundError:
        print(f"[{pkg_name}] skipped missing {mod_name}")

print(f"[{pkg_name}] flatten complete")
