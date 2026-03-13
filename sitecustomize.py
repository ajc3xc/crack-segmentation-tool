"""Runtime compatibility fixes for the Pixi environment."""

import importlib.util
import os
import sys


def _bootstrap_cv2() -> None:
    """Load compiled cv2 bindings when cv2 imports as an empty namespace package."""
    try:
        import cv2  # type: ignore
    except Exception:
        return

    if hasattr(cv2, "imread"):
        return

    pkg_paths = list(getattr(cv2, "__path__", []))
    if not pkg_paths:
        return

    py_tag = f"python-{sys.version_info.major}.{sys.version_info.minor}"
    so_path = os.path.join(pkg_paths[0], py_tag, f"cv2.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so")
    if not os.path.isfile(so_path):
        return

    # Load extension module with the canonical name `cv2`.
    spec = importlib.util.spec_from_file_location("cv2", so_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["cv2"] = module
    spec.loader.exec_module(module)


_bootstrap_cv2()
