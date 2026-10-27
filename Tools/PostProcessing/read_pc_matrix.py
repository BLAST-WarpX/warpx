#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Read a preconditioner matrix written by WarpX with <prefix>.print_matrix = 1.

The matrix file is in MatrixMarket coordinate format. Its "% key: value" header
lines hold the metadata; dofmap.txt in the same directory maps each row to its
AMR level, field component, and cell indices.

Usage as a script:
    python read_pc_matrix.py diags/pc_matrices/pc_jacobi_step00000_update0.mtx [--plot]

Usage as a module:
    from read_pc_matrix import read_pc_matrix, read_dofmap
    A, info = read_pc_matrix("diags/pc_matrices/pc_jacobi_step00000_update0.mtx")
    dofmap = read_dofmap("diags/pc_matrices")
"""

import argparse
import os

import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg


def read_header(filename):
    """Return the "% key: value" metadata lines of a matrix file as a dict of strings."""
    info = {}
    with open(filename) as f:
        for line in f:
            if line.startswith("%%"):
                continue
            if not line.startswith("%"):
                break
            key, sep, value = line[1:].strip().partition(":")
            if sep:
                info[key.strip()] = value.strip()
    return info


def read_pc_matrix(filename):
    """Return the matrix as a scipy CSR matrix, and its metadata as a dict."""
    A = scipy.sparse.csr_matrix(scipy.io.mmread(filename))
    return A, read_header(filename)


def read_dofmap(directory):
    """Return the DOF map as a dict of arrays indexed by the 0-based row.

    Keys: "lev", "comp" (0: x, 1: y, 2: z), "i", "j", "k" (cell indices).
    """
    data = np.loadtxt(
        os.path.join(directory, "dofmap.txt"), comments="#", dtype=np.int64
    )
    data = data[np.argsort(data[:, 0])]
    return {key: data[:, n + 1] for n, key in enumerate(("lev", "comp", "i", "j", "k"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("filename", help="matrix file (.mtx)")
    parser.add_argument("--plot", action="store_true", help="plot the sparsity pattern")
    args = parser.parse_args()

    A, info = read_pc_matrix(args.filename)
    for key, value in info.items():
        print(f"{key}: {value}")
    diag = A.diagonal()
    asym = scipy.sparse.linalg.norm(A - A.T) / scipy.sparse.linalg.norm(A)
    print(f"shape: {A.shape}, nnz: {A.nnz}")
    print(f"diagonal: min {diag.min():.6g}, max {diag.max():.6g}")
    print(f"relative asymmetry ||A - A^T||_F / ||A||_F: {asym:.3e}")

    dofmap = read_dofmap(os.path.dirname(os.path.abspath(args.filename)))
    for comp in range(3):
        rows = np.flatnonzero(dofmap["comp"] == comp)
        if rows.size > 0:
            print(f"rows of field component {comp}: {rows.size}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.spy(A, markersize=0.5)
        plt.title(info.get("name", os.path.basename(args.filename)))
        plt.show()


if __name__ == "__main__":
    main()
