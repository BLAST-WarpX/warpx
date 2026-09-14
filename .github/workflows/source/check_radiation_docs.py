#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Keep radiation qualification pages in their maintained Sphinx navigation."""

import re
from pathlib import Path


def missing_entries(documents, contents):
    blocks = re.findall(
        r"^```\{toctree\}\s*\n(.*?)^```\s*$", contents, re.MULTILINE | re.DOTALL
    )
    entries = set()
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith(":"):
                entries.add(Path(line.split("<")[-1].rstrip(">")).stem)
    return documents - {"radiation_qualification_wrapup"} - entries


if __name__ == "__main__":
    theory = Path(__file__).resolve().parents[3] / "Docs/source/theory"
    documents = {path.stem for path in theory.glob("radiation_*.md")}
    contents = (theory / "radiation_qualification_wrapup.md").read_text()
    missing = missing_entries(documents, contents)
    if missing:
        raise SystemExit(
            "Radiation pages missing from qualification toctree: "
            + ", ".join(sorted(missing))
        )
    print("Radiation qualification navigation passed")
