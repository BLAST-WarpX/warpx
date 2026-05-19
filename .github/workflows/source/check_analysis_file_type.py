#! /usr/bin/env python3
"""Check that every analysis_default_regression.py under Examples/Tests and
Examples/Physics_applications is a symlink pointing to the shared canonical
file at Examples/analysis_default_regression.py."""

import sys
from pathlib import Path

# Resolve paths from this script location so the check works from any cwd.
repo_root = Path(__file__).resolve().parents[3]
canonical = repo_root / "Examples" / "analysis_default_regression.py"
search_roots = [
    repo_root / "Examples" / "Tests",
    repo_root / "Examples" / "Physics_applications",
]


failed = False

# Only tests and physics applications are expected to link to the shared analysis file.
for search_root in search_roots:
    for path in search_root.rglob("analysis_default_regression.py"):
        path_relative = path.relative_to(repo_root)

        if not path.is_symlink():
            print(
                f"{path_relative} must be a symlink and point to Examples/analysis_default_regression.py"
            )
            failed = True
            continue

        # Keep broken or unexpected links reportable instead of raising early.
        target = path.resolve(strict=False)
        if target != canonical:
            print(f"{path_relative} links to {target}, expected {canonical}")
            failed = True

if failed:
    print("FAILED")
    sys.exit(1)
