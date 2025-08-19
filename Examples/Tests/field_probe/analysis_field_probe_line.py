#!/usr/bin/env python3

import argparse
import math

import pandas as pd


def checkEq(a, b):
    if a != b:
        raise ValueError(f"Expected {b} but got {a}")


def validate_fieldprobe_file(args):
    df = pd.read_csv(args.path, sep=" ")

    print(df.columns, len(df))
    checkEq(len(df.columns), 12)
    checkEq(len(df), 1 + (args.resolution * math.floor(args.max_step / args.intervals)))


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()
    # add arguments: output path
    parser.add_argument(
        "--path",
        help="path to field probe output file(s)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--intervals",
        help="intervals value from the input file",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--max_step",
        help="max_step value from the input file",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--resolution",
        help="resolution value from the input file",
        default=20,
        type=int,
    )
    # parse arguments
    args = parser.parse_args()

    # validate the field probe file
    validate_fieldprobe_file(args)
