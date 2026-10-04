#!/usr/bin/env python3
"""
Copyright 2026 The WarpX Community

This file is part of WarpX.

Check that the pywarpx version resolves to the version in ``dependencies.json``
without importing a compiled module.

One installed pywarpx wheel contains every dimensionality side by side, and
importing any ``warpx_pybind_*`` module also imports the matching
``amrex.space*d`` module, pinning the process to that dimensionality. A version
query must therefore never trigger such an import.

Without arguments, ``Python/pywarpx/_version.py`` is loaded by path: this needs
neither a build of WarpX nor any third-party Python package. With
``--installed``, the installed ``pywarpx`` package is imported instead, which
additionally covers the packaging of ``dependencies.json`` into the wheel.

Authors: Axel Huebl
License: BSD-3-Clause-LBNL
"""

import importlib.util
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
installed = "--installed" in sys.argv[1:]

with open(repo_root / "dependencies.json", "r", encoding="utf-8") as f:
    expected = json.load(f)["version_warpx"]

if installed:
    import pywarpx

    version = pywarpx.__version__
    # no dimensionality has been selected yet, so there is no compiled module
    if pywarpx.__git_version__ is not None:
        sys.exit(
            "pywarpx.__git_version__ must be None before a dimensionality is "
            f"loaded, got '{pywarpx.__git_version__}'"
        )
else:
    spec = importlib.util.spec_from_file_location(
        "pywarpx_version_check", repo_root / "Python" / "pywarpx" / "_version.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = module.__version__

if version != expected:
    sys.exit(
        f"pywarpx version mismatch: resolved '{version}', "
        f"dependencies.json says '{expected}'"
    )

compiled = [
    name
    for name in sys.modules
    if "warpx_pybind" in name or name == "amrex" or name.startswith("amrex.")
]
if compiled:
    sys.exit(
        "resolving the pywarpx version must not import a dimensionality-specific "
        f"module, but these were imported: {compiled}"
    )

print(f"pywarpx version resolves to '{version}'")
