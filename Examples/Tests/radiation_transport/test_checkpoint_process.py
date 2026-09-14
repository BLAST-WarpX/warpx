#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Regression for expected-rejection process ownership and timeout evidence."""

import os
import sys
import tempfile
from pathlib import Path

from check_moving_moment_checkpoint import run_rejection


def check():
    with tempfile.TemporaryDirectory(
        prefix="checkpoint-process-", dir="."
    ) as directory:
        log = Path(directory) / "child.log"
        result = run_rejection(
            [
                sys.executable,
                "-c",
                "print('rejected', flush=True); raise SystemExit(7)",
            ],
            log,
        )
        assert result.returncode == 7 and "rejected" in result.stdout
        if os.name == "posix":
            # The direct child exits, while a descendant inherits its output
            # descriptor. Waiting for captured-pipe EOF would hang until timeout.
            result = run_rejection(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; subprocess.Popen([sys.executable,'-c',"
                    "'import time; time.sleep(60)']); print('rejected',flush=True); sys.exit(7)",
                ],
                log,
                timeout=5,
            )
            assert result.returncode == 7 and "rejected" in result.stdout
        try:
            run_rejection(
                [
                    sys.executable,
                    "-c",
                    "import time; print('partial',flush=True); time.sleep(60)",
                ],
                log,
                timeout=1,
            )
        except AssertionError as error:
            assert "Checkpoint rejection exceeded" in str(error)
            assert "partial" in log.read_text()
        else:
            raise AssertionError(
                "A timed-out command was incorrectly accepted as rejection"
            )
    print(
        "Checkpoint child exit, inherited output, cleanup and timeout evidence passed"
    )


if __name__ == "__main__":
    check()
