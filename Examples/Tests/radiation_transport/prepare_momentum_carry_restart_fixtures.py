#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Create isolated legacy and damaged-current momentum-carry checkpoints."""

import argparse
import shutil
from pathlib import Path


def fresh_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def carry_files(checkpoint: Path, field: str) -> list[Path]:
    return sorted(
        path
        for level in checkpoint.glob("Level_*")
        for path in level.iterdir()
        if path.name.startswith(f"{field}[level=")
    )


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("legacy", type=Path)
parser.add_argument("corrupt_current", type=Path)
args = parser.parse_args()

source_header = args.source / "WarpXHeader"
assert source_header.is_file()
source_lines = source_header.read_text().splitlines(keepends=True)
assert source_lines
assert (
    source_lines[0].strip()
    == "Checkpoint version: 2 radiation_momentum_carry_fields: 1"
)

streaming = "radiation_streaming_momentum_carry"
diffusion = "radiation_diffusion_momentum_carry"
for field in (streaming, diffusion):
    files = carry_files(args.source, field)
    assert any(path.name.endswith("_H") for path in files)
    assert any("_D_" in path.name for path in files)

fresh_copy(args.source, args.legacy)
legacy_header = args.legacy / "WarpXHeader"
legacy_lines = legacy_header.read_text().splitlines(keepends=True)
legacy_lines[0] = "Checkpoint version: 1\n"
legacy_header.write_text("".join(legacy_lines))
for field in (streaming, diffusion):
    files = carry_files(args.legacy, field)
    assert files
    for path in files:
        path.unlink()
    assert not carry_files(args.legacy, field)

fresh_copy(args.source, args.corrupt_current)
assert (args.corrupt_current / "WarpXHeader").read_text().splitlines()[
    0
] == source_lines[0].strip()
corrupt_headers = [
    path
    for path in carry_files(args.corrupt_current, diffusion)
    if path.name.endswith("_H")
]
assert corrupt_headers
for path in corrupt_headers:
    path.unlink()
assert not any(
    path.name.endswith("_H") for path in carry_files(args.corrupt_current, diffusion)
)
assert any("_D_" in path.name for path in carry_files(args.corrupt_current, diffusion))
# The source fixture is shared by the ordinary restart test and must remain intact.
assert source_header.read_text().splitlines()[0] == source_lines[0].strip()
for field in (streaming, diffusion):
    assert any(path.name.endswith("_H") for path in carry_files(args.source, field))

print(f"legacy carry fixture:         {args.legacy}")
print(f"damaged current carry fixture: {args.corrupt_current}")
