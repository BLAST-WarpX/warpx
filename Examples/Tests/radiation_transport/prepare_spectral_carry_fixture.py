#!/usr/bin/env python3
"""Copy a generated checkpoint and remove only its spectral carry header."""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
args = parser.parse_args()
source = args.source.resolve()
target = source.with_name(source.name + "_missing_spectral_carry")
assert source.name == "chk000001"
assert source.parent.name == "diags"
assert source.parent.parent.name == "test_1d_radiation_transport_opposing_group_work"
assert (source / "WarpXHeader").read_text().splitlines()[0] == (
    "Checkpoint version: 3 radiation_momentum_carry_fields: 1 "
    "radiation_diffusion_momentum_groups: 2"
)
if target.exists():
    # The target is exclusively this test's disposable checkpoint copy.
    assert target.parent == source.parent and not target.is_symlink()
    shutil.rmtree(target)
shutil.copytree(source, target)
headers = list(target.glob("Level_*/radiation_diffusion_group_momentum_carry*_H"))
assert headers
for path in headers:
    path.unlink()
assert list(source.glob("Level_*/radiation_diffusion_group_momentum_carry*_H"))
