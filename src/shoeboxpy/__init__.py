from __future__ import annotations

"""Top-level package for shoeboxpy.

The package version is derived from the installed distribution metadata via
importlib.metadata (set at build time by setuptools-scm). When the package
is imported from a source checkout without installation, a safe fallback
value is provided.
"""

from importlib import metadata as _metadata

try:  # Preferred path: distribution is installed (editable or regular)
	__version__ = _metadata.version("shoeboxpy")
except _metadata.PackageNotFoundError:  # Source tree without installed dist
	__version__ = "0.0.0"  # Fallback; not an authoritative release tag

__all__ = ["__version__"]