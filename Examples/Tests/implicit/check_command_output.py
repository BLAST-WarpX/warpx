#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Run a command and verify its exit status and output contract."""

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
outcome = parser.add_mutually_exclusive_group(required=True)
outcome.add_argument("--expect-success", action="store_true")
outcome.add_argument("--expect-failure", action="store_true")
parser.add_argument("--expect", action="append", default=[])
parser.add_argument("--forbid", action="append", default=[])
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()

command = args.command
if command and command[0] == "--":
    command = command[1:]
if not command:
    parser.error("a command is required after --")

result = subprocess.run(
    command,
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
print(result.stdout, end="")

errors = []
if args.expect_success and result.returncode != 0:
    errors.append(f"command unexpectedly failed with exit code {result.returncode}")
if args.expect_failure and result.returncode == 0:
    errors.append("command unexpectedly returned success")
for expected in args.expect:
    if expected not in result.stdout:
        errors.append(f"missing expected output: {expected!r}")
for forbidden in args.forbid:
    if forbidden in result.stdout:
        errors.append(f"found forbidden output: {forbidden!r}")

if errors:
    print("command-output contract was not satisfied:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    sys.exit(1)
