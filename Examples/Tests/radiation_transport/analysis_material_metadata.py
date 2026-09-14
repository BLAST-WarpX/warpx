#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL

"""Strict material identity checks; manufactured metadata, not opacity calibration."""

import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("executable", type=Path)
parser.add_argument("inputs", type=Path)
parser.add_argument("table", type=Path)
args = parser.parse_args()
template = args.inputs.read_text()
cases = [
    ("consistent", [], "Material metadata audit passed", True),
    (
        "wrong_mass",
        ["hybrid_pic_model.electron_composition_atomic_mass=12"],
        "Opacity/EOS/PIC atomic-mass metadata mismatch",
        False,
    ),
    (
        "wrong_nucleus",
        ["hybrid_pic_model.electron_composition_atomic_number=6"],
        "Opacity/EOS nuclear-charge metadata mismatch",
        False,
    ),
    (
        "excessive_charge",
        ["ions.charge=75*q_e"],
        "Fixed PIC ion charge exceeds declared nuclear charge",
        False,
    ),
    ("unknown_metadata", [], "explicit native hybrid composition", False),
    (
        "legacy_opt_out",
        [
            "hybrid_pic_model.electron_composition_atomic_number=6",
            "radiation_transport.require_material_metadata_consistency=0",
        ],
        "",
        True,
    ),
]
for name, overrides, expected, succeeds in cases:
    directory = Path(name)
    directory.mkdir(exist_ok=True)
    content = template
    if name == "unknown_metadata":
        content = (
            "\n".join(
                line
                for line in template.splitlines()
                if not line.startswith("hybrid_pic_model.electron_composition_")
            )
            + "\n"
        )
    (directory / "inputs").write_text(content)
    result = subprocess.run(
        [
            str(args.executable.resolve()),
            "inputs",
            f"materials.tungsten.opacity_table_file={args.table.resolve()}",
            "amrex.throw_exception=1",
            *overrides,
        ],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (directory / "run.log").write_text(result.stdout)
    assert (result.returncode == 0) == succeeds, (name, result.stdout)
    assert expected in result.stdout, (name, expected, result.stdout)
    print(f"Material metadata contract: {name} passed")
