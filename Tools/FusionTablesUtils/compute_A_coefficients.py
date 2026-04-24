#!/usr/bin/env python3
"""Convert L coefficients to A coefficients.

The algorithm is described in sections 2.4 and 2.5 of Higginson et al.,
J. Comput. Phys. 388 (2019), https://doi.org/10.1016/j.jcp.2019.03.020.
"""

import argparse
import math
from pathlib import Path

DEGREES = range(0, 17, 2)


def parse_args():
    # Require explicit file paths so the script is easy to reuse.
    parser = argparse.ArgumentParser(
        description="Convert even Legendre L coefficients to A coefficients."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="input table path",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output table path",
    )
    return parser.parse_args()


def binom(a, k):
    # Generalized binomial coefficient for non-integer upper arguments.
    value = 1.0
    for m in range(1, k + 1):
        value *= (a - m + 1) / m
    return value


def compute_a(n, l_coeffs):
    # Only even j contribute for this table; absent higher L_j values are zero.
    total = 0.0
    for j in range(n, 17, 2):
        total += (
            l_coeffs.get(j, 0.0) * 2**j * math.comb(j, n) * binom(0.5 * (j + n - 1), j)
        )
    return total


def format_table(rows):
    # Preserve a plain text table with aligned column starts.
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append(
            "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip()
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    lines = [
        line.split() for line in input_path.read_text().splitlines() if line.strip()
    ]
    header = lines[0]
    # Header labels are E L0 L2 ... L16.
    l_degrees = [int(label[1:]) for label in header[1:]]

    output_rows = [["E", *(f"A{n}" for n in DEGREES)]]
    for fields in lines[1:]:
        energy = fields[0]
        l_coeffs = {
            degree: float(value) for degree, value in zip(l_degrees, fields[1:])
        }
        output_rows.append(
            [
                energy,
                *(f"{compute_a(n, l_coeffs):.12g}" for n in DEGREES),
            ]
        )

    output_path.write_text(format_table(output_rows))


if __name__ == "__main__":
    main()
