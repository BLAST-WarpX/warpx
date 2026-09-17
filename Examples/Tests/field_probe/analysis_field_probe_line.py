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
POSITION_COLUMNS = [
    "[2]part_x_lev0-(m)",
    "[3]part_y_lev0-(m)",
    "[4]part_z_lev0-(m)",
]
Z_COLUMN = "[4]part_z_lev0-(m)"
POYNTING_COLUMN = "[11]part_S_lev0-(W/m^2)"
POYNTING_COLUMN_INTEGRATE = "[11]part_S_lev0-(W*s/m^2)"


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


def checkLte(a, b, msg=None):
    if a > b:
        if msg:
            raise ValueError(f"{msg}: Expected {a} to be less than or equal to {b}")
        else:
            raise ValueError(f"Expected {a} to be less than or equal to {b}")


# Verify non-zero Poynting values in expected z ranges at each time step
def check_non_zero_ranges(df, args):
    S_column_name = POYNTING_COLUMN_INTEGRATE if args.integrate else POYNTING_COLUMN
    for _, row in df.iterrows():
        step = row[STEP_COLUMN]
        z = row[Z_COLUMN]
        poynting = row[S_column_name]

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


def check_expected_particle_count(df, args):
    if args.expected_count is None:
        return
    counts = df.groupby(STEP_COLUMN).size()
    for step, count in counts.items():
        checkEq(
            count,
            args.expected_count,
            f"Unexpected particle count at step {step}",
        )


def check_enters_domain(df, args):
    if df.empty:
        raise ValueError("The probe never entered the moving-window domain")
    found_steps = sorted(df[STEP_COLUMN].unique())
    checkLt(0, found_steps[0], "Probe should start outside the domain")
    checkEq(found_steps[-1], args.max_step, "Probe should remain visible at the end")


def check_window_bounds(df, args):
    z_by_step = df.groupby(STEP_COLUMN)[Z_COLUMN].first()
    before_start = z_by_step[z_by_step.index < args.window_start]
    after_end = z_by_step[z_by_step.index >= args.window_end]
    checkEq(
        before_start.nunique(),
        1,
        "Probe moved before the configured moving-window start",
    )
    checkEq(
        after_end.nunique(),
        1,
        "Probe moved after the configured moving-window end",
    )
    if z_by_step.nunique() <= 1:
        raise ValueError("Probe did not move while the moving window was active")


def check_restart_positions(df, args):
    reference_df = pd.read_csv(args.reference_path, sep=" ")
    positions = df[df[STEP_COLUMN] == args.max_step][POSITION_COLUMNS]
    reference_positions = reference_df[reference_df[STEP_COLUMN] == args.max_step][
        POSITION_COLUMNS
    ]

    if positions.empty:
        raise ValueError(f"Restart output has no probe data at step {args.max_step}")

    positions = (
        positions.drop_duplicates().sort_values(POSITION_COLUMNS).reset_index(drop=True)
    )
    reference_positions = (
        reference_positions.drop_duplicates()
        .sort_values(POSITION_COLUMNS)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        positions,
        reference_positions,
        check_exact=True,
    )


def check_has_every_column(df, args):
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
    if args.integrate:
        expected_columns = [
            STEP_COLUMN,
            "[1]time(s)",
            "[2]part_x_lev0-(m)",
            "[3]part_y_lev0-(m)",
            Z_COLUMN,
            "[5]part_Ex_lev0-(V*s/m)",
            "[6]part_Ey_lev0-(V*s/m)",
            "[7]part_Ez_lev0-(V*s/m)",
            "[8]part_Bx_lev0-(T*s)",
            "[9]part_By_lev0-(T*s)",
            "[10]part_Bz_lev0-(T*s)",
            POYNTING_COLUMN_INTEGRATE,
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
        elif args.moving_window:
            checkLte(
                prev_step_max,
                current_step_max,
                f"Maximum z at step {step} should not decrease from step {step - args.intervals}",
            )
        else:
            checkEq(
                prev_step_max,
                current_step_max,
                f"Maximum z at step {step} should be equal to step {step - args.intervals}",
            )

        prev_step_min = current_step_min
        prev_step_max = current_step_max


def check_integrate(df):
    sorted_steps = sorted(df[STEP_COLUMN].unique())
    prev_z_to_poynting = {}
    is_current_step = df[STEP_COLUMN] == sorted_steps[0]
    for _, row in df[is_current_step].iterrows():
        z = row[Z_COLUMN]
        poynting = row[POYNTING_COLUMN_INTEGRATE]
        prev_z_to_poynting[z] = poynting

    for step in sorted_steps[1:]:
        is_current_step = df[STEP_COLUMN] == step
        next_z_to_poynting = {}
        for _, row in df[is_current_step].iterrows():
            z = row[Z_COLUMN]
            poynting = row[POYNTING_COLUMN_INTEGRATE]
            if z in prev_z_to_poynting:
                checkLte(
                    prev_z_to_poynting[z],
                    poynting,
                    f"Poynting value at step {step} for position {z} should be greater than previous step's Poynting value",
                )
            next_z_to_poynting[z] = poynting
        prev_z_to_poynting = next_z_to_poynting


def validate_fieldprobe_file(args):
    df = pd.read_csv(args.path, sep=" ")

    check_has_every_column(df, args)
    if args.expect_empty:
        checkEq(len(df), 0, "Expected no in-domain probe rows")
        return
    if args.enters_domain:
        check_enters_domain(df, args)
        return
    if args.reference_path is not None:
        check_restart_positions(df, args)
        return

    check_has_data_for_every_step(df, args)
    check_expected_particle_count(df, args)
    if args.positions_only:
        return
    if args.window_start is not None:
        check_window_bounds(df, args)
        return

    check_non_zero_ranges(df, args)
    check_moving_windows(df, args)
    if args.integrate:
        check_integrate(df)


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
    parser.add_argument(
        "--integrate",
        help="integrate value from the input file",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--expected_count",
        help="expected number of in-domain probe particles per output step",
        type=int,
    )
    parser.add_argument(
        "--expect_empty",
        help="expect the output file to contain only its header",
        action="store_true",
    )
    parser.add_argument(
        "--enters_domain",
        help="expect an initially absent probe to enter the moving-window domain",
        action="store_true",
    )
    parser.add_argument(
        "--positions_only",
        help="only validate columns, steps, positions, and particle counts",
        action="store_true",
    )
    parser.add_argument(
        "--reference_path",
        help="uninterrupted FieldProbe output used for a restart comparison",
        type=str,
    )
    parser.add_argument(
        "--window_start",
        help="first step at which the moving window is active",
        type=int,
    )
    parser.add_argument(
        "--window_end",
        help="first step at which the moving window is inactive",
        type=int,
    )

    # parse arguments
    args = parser.parse_args()

    # validate the field probe file
    validate_fieldprobe_file(args)
