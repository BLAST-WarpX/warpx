#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Checks a preconditioner matrix written with <prefix>.print_matrix = 1:
# sizes and nonzero counts against the metadata, the DOF map, and the
# row statistics recomputed from the entries.
#
# Usage: analysis_pc_matrix.py <matrix file>

import os
import sys

import numpy as np
import scipy.io

sys.path.insert(
    1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../Tools/PostProcessing"
    ),
)
from read_pc_matrix import read_dofmap, read_header  # noqa: E402

filename = sys.argv[1]
info = read_header(filename)
A = scipy.io.mmread(filename).tocoo()

n = int(info["global_size"].split("x")[0])
print(f"matrix {info['name']}: {A.shape}, nnz = {A.nnz}")
assert A.shape == (n, n)
assert A.nnz == int(info["nnz"])
assert sum(int(x) for x in info["local_nnz"].split()) == A.nnz
row_offsets = [int(x) for x in info["row_offsets"].split()]
assert row_offsets[0] == 0 and row_offsets[-1] == n
assert len(row_offsets) == int(info["ranks"]) + 1

dofmap = read_dofmap(os.path.dirname(os.path.abspath(filename)))
assert dofmap["comp"].size == n

# Row statistics, recomputed from the entries
A = A.tocsr()
diag = A.diagonal()
offdiag = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(diag)
ratio = np.max(offdiag / np.abs(diag))
omega = 2.0 / (2.0 + ratio)
print(f"diagonal min = {diag.min()}, gershgorin omega = {omega}")
assert np.isclose(diag.min(), float(info["diagonal_min"]), rtol=1e-8)
assert np.isclose(ratio, float(info["max_offdiag_ratio"]), rtol=1e-8)
assert np.isclose(omega, float(info["gershgorin_omega"]), rtol=1e-8)

# The weighted Jacobi PC starts from the Gershgorin estimate
if "omega" in info:
    assert np.isclose(omega, float(info["omega"]), rtol=1e-5)
