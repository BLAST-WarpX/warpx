#!/usr/bin/env python3

import argparse
import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class NonZeroRange:
    start: float
    end: float


NON_ZERO_RANGES = {
    0: NonZeroRange(math.inf, -math.inf),
    10: NonZeroRange(4e-6, 3e-5),
    20: NonZeroRange(1e-6, 3e-5),
    30: NonZeroRange(-1e-6, 3e-5),
    40: NonZeroRange(-4e-6, 3e-5),
    50: NonZeroRange(-7e-6, 3e-5),
}

STEP_COLUMN = "[0]step()"
Z_COLUMN = "[4]part_z_lev0-(m)"
POYNTING_COLUMN = "[11]part_S_lev0-(W/m^2)"


def checkEq(a, b, msg=None):
    if a != b:
        if msg:
            raise ValueError(f"{msg}: Expected {b} but got {a}")
        else:
            raise ValueError(f"Expected {b} but got {a}")


def checkLt(a, b, msg=None):
    if a >= b:
        if msg:
            raise ValueError(f"{msg}: Expected {a} to be less than {b}")
        else:
            raise ValueError(f"Expected {a} to be less than {b}")


# Verify non-zero Poynting values in expected z ranges at each time step
def check_non_zero_ranges(df):
    for _, row in df.iterrows():
        step = row[STEP_COLUMN]
        z = row[Z_COLUMN]
        poynting = row[POYNTING_COLUMN]

        range = NON_ZERO_RANGES[step]
        is_zero = poynting == 0
        expected_zero = z < range.start or z > range.end
        checkEq(
            is_zero,
            expected_zero,
            f"Step {step} at z={z} has Poynting {poynting}, expected zero: {expected_zero}",
        )


def check_has_data_for_every_step(df, args):
    found_steps = sorted(df[STEP_COLUMN].unique())
    checkEq(found_steps, list(range(0, args.max_step + 1, args.intervals)))


def check_has_every_column(df):
    expected_columns = [
        STEP_COLUMN,
        "[1]time(s)",
        "[2]part_x_lev0-(m)",
        "[3]part_y_lev0-(m)",
        Z_COLUMN,
        "[5]part_Ex_lev0-(V/m)",
        "[6]part_Ey_lev0-(V/m)",
        "[7]part_Ez_lev0-(V/m)",
        "[8]part_Bx_lev0-(T)",
        "[9]part_By_lev0-(T)",
        "[10]part_Bz_lev0-(T)",
        POYNTING_COLUMN,
    ]
    checkEq(
        sorted(df.columns),
        sorted(expected_columns),
        "DataFrame columns do not match expected columns",
    )


def check_moving_windows(df, args):
    min_z_should_grow = args.moving_window or args.fp_moving_window
    max_z_should_grow = args.fp_moving_window

    sorted_steps = sorted(df[STEP_COLUMN].unique())
    initial_step_condition = df[STEP_COLUMN] == sorted_steps[0]
    prev_step_min = df[initial_step_condition][Z_COLUMN].min()
    prev_step_max = df[initial_step_condition][Z_COLUMN].max()

    for step in sorted_steps[1:]:
        is_current_step = df[STEP_COLUMN] == step
        current_step_min = df[is_current_step][Z_COLUMN].min()
        current_step_max = df[is_current_step][Z_COLUMN].max()

        if min_z_should_grow:
            checkLt(
                prev_step_min,
                current_step_min,
                f"Minimum z at step {step} should be greater than step {step - args.intervals}",
            )
        else:
            checkEq(
                prev_step_min,
                current_step_min,
                f"Minimum z at step {step} should be equal to step {step - args.intervals}",
            )

        if max_z_should_grow:
            checkLt(
                prev_step_max,
                current_step_max,
                f"Maximum z at step {step} should be greater than step {step - args.intervals}",
            )
        else:
            checkEq(
                prev_step_max,
                current_step_max,
                f"Maximum z at step {step} should be equal to step {step - args.intervals}",
            )

        prev_step_min = current_step_min
        prev_step_max = current_step_max


def validate_fieldprobe_file(args):
    df = pd.read_csv(args.path, sep=" ")

    check_has_every_column(df)
    check_has_data_for_every_step(df, args)
    check_non_zero_ranges(df)
    check_moving_windows(df, args)


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
    parser.add_argument(
        "--moving_window",
        help="moving window value from the input file",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fp_moving_window",
        help="field probe moving window value from the input file",
        default=False,
        action="store_true",
    )
    # parse arguments
    args = parser.parse_args()

    # validate the field probe file
    validate_fieldprobe_file(args)
