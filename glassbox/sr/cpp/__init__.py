"""Glassbox SR C++ package: native extension loader and shared Python helpers.

The extension is built as a bare ``_core`` shared library next to this package
(``setup.py`` / inplace). Import via :func:`load_cpp_core` rather than
mutating ``sys.path`` at call sites (P6-001 / P6-009 / P6-010).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Tuple

from glassbox.sr.cpp.graph_enums import *
from glassbox.sr.cpp.graph_enums import __all__ as _ENUM_ALL

__all__ = [
    "load_cpp_core",
    "get_cpp_core",
    "require_cpp_core",
    "call_with_optional_kwargs",
    "CPP_AVAILABLE",
    "CPP_UNAVAILABLE_REASON",
    "cpp_dir",
    *_ENUM_ALL,
]

_LOG = logging.getLogger(__name__)

_CPP_DIR = Path(__file__).resolve().parent
_core_module: ModuleType | None = None
_load_error: str | None = None
_loaded = False


def cpp_dir() -> Path:
    """Directory containing headers and the built ``_core`` extension."""
    return _CPP_DIR


def load_cpp_core() -> tuple[ModuleType | None, str | None]:
    """Load the native ``_core`` extension with ABI-aware diagnostics.

    Returns ``(module, None)`` on success, or ``(None, reason)`` on failure.
    Safe to call repeatedly; caches the first successful load.
    """
    global _core_module, _load_error, _loaded
    if _loaded:
        return _core_module, _load_error

    errors: list[str] = []

    # 1) Already on sys.modules (e.g. prior path insert).
    if "_core" in sys.modules:
        _core_module = sys.modules["_core"]
        _load_error = None
        _loaded = True
        return _core_module, None

    # 2) Package-relative: glassbox.sr.cpp._core (works when .so sits in package).
    try:
        mod = importlib.import_module("glassbox.sr.cpp._core")
        _core_module = mod
        sys.modules.setdefault("_core", mod)
        _load_error = None
        _loaded = True
        return _core_module, None
    except ImportError as exc:
        errors.append(f"glassbox.sr.cpp._core: {exc}")

    # 3) Load the shared library from this directory via importlib (no permanent
    #    sys.path mutation required for callers).
    candidates = sorted(_CPP_DIR.glob("_core.*"))
    # Prefer extension modules over unrelated files.
    ext_suffixes = (".so", ".pyd", ".dll")
    for path in candidates:
        if not path.name.startswith("_core"):
            continue
        if path.suffix not in ext_suffixes and not any(
            path.name.endswith(s) for s in (".so", ".pyd")
        ):
            # e.g. _core.cpython-314-x86_64-linux-gnu.so
            if ".so" not in path.name and ".pyd" not in path.name:
                continue
        try:
            spec = importlib.util.spec_from_file_location("_core", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            sys.modules["_core"] = mod
            sys.modules["glassbox.sr.cpp._core"] = mod
            _core_module = mod
            _load_error = None
            _loaded = True
            return _core_module, None
        except Exception as exc:  # pragma: no cover - ABI mismatch path
            errors.append(f"{path.name}: {exc}")

    # 4) Bare import after temporary path insert (legacy fallback).
    cpp_s = str(_CPP_DIR)
    inserted = False
    if cpp_s not in sys.path:
        sys.path.insert(0, cpp_s)
        inserted = True
    try:
        mod = importlib.import_module("_core")
        _core_module = mod
        sys.modules.setdefault("glassbox.sr.cpp._core", mod)
        _load_error = None
        _loaded = True
        return _core_module, None
    except ImportError as exc:
        errors.append(f"_core: {exc}")
    finally:
        if inserted:
            try:
                sys.path.remove(cpp_s)
            except ValueError:
                pass

    active_abi = getattr(sys.implementation, "cache_tag", "unknown ABI")
    built = sorted(p.name for p in _CPP_DIR.glob("_core.*") if p.is_file())
    if built:
        reason = (
            f"C++ _core extension unavailable for active ABI {active_abi}; "
            f"found {', '.join(built)}. Rebuild: "
            f"python glassbox/sr/cpp/setup.py build_ext --inplace --force"
        )
    else:
        reason = (
            f"C++ _core extension unavailable for active ABI {active_abi} "
            f"({'; '.join(errors) if errors else 'not built'}). "
            f"Build: python glassbox/sr/cpp/setup.py build_ext --inplace"
        )
    _core_module = None
    _load_error = reason
    _loaded = True
    return None, reason


def get_cpp_core() -> ModuleType | None:
    """Return the loaded extension module, or None if unavailable."""
    mod, _ = load_cpp_core()
    return mod


def require_cpp_core() -> ModuleType:
    """Return the extension or raise ImportError with an actionable message."""
    mod, reason = load_cpp_core()
    if mod is None:
        raise ImportError(reason or "C++ _core extension unavailable")
    return mod


def call_with_optional_kwargs(
    func: Callable[..., Any],
    *args: Any,
    optional_kwargs: dict | None = None,
    required_kwargs: dict | None = None,
    log: logging.Logger | None = None,
) -> Any:
    """Call ``func`` dropping kwargs not accepted by its signature (P6-003).

    On TypeError, inspects the callable signature and retries with only
    supported keyword names. Logs dropped keys at debug level.
    """
    log = log or _LOG
    required_kwargs = dict(required_kwargs or {})
    optional_kwargs = dict(optional_kwargs or {})
    kwargs = {**required_kwargs, **optional_kwargs}
    try:
        return func(*args, **kwargs)
    except TypeError:
        pass

    try:
        sig = inspect.signature(func)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kw:
            # Signature accepts **kwargs; TypeError is not about unknown names.
            raise
        allowed = {
            name
            for name, p in params.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = sorted(set(kwargs) - set(filtered))
        if dropped:
            log.debug(
                "call_with_optional_kwargs: dropped unsupported kwargs %s for %s",
                dropped,
                getattr(func, "__name__", repr(func)),
            )
        return func(*args, **filtered)
    except TypeError:
        # Last resort: required kwargs only (no optional extras).
        if optional_kwargs:
            log.debug(
                "call_with_optional_kwargs: retry with required kwargs only for %s",
                getattr(func, "__name__", repr(func)),
            )
            return func(*args, **required_kwargs)
        raise


# Eager availability flag for skipif / feature gates (P6-014: degrade with reason).
_core, CPP_UNAVAILABLE_REASON = load_cpp_core()
CPP_AVAILABLE = _core is not None

# Expose as attribute for ``from glassbox.sr.cpp import _core``.
if _core is not None:
    globals()["_core"] = _core
