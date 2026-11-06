# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import threading
import time

import pytest
from helpers import make_sim

import pywarpx
from pywarpx import callbacks

pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("1d", "2d", "3d"),
    reason="make_sim builds Cartesian geometries only",
)


def test_step_releases_gil():
    """Other Python threads keep running while ``sim.step`` advances.

    WarpX releases the GIL while it advances the simulation. If it kept the
    GIL, then the ticker thread below could only run where the main thread
    executes Python code: right before and after the call into C++, i.e., a few
    ticks at most. Callbacks re-acquire the GIL, also while the ticker wants it.
    """
    sim = make_sim()
    sim.initialize_inputs()
    sim.initialize_warpx()

    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    calls = []

    def count_calls():
        calls.append(1)

    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        # double the number of steps until a single call takes long enough to
        # count ticks in it, also on a fast machine
        nsteps = 1
        while True:
            start = time.perf_counter()
            sim.step(nsteps)
            end = time.perf_counter()
            if end - start > 0.3:
                break
            nsteps *= 2

        callbacks.installcallback("afterstep", count_calls)
        sim.step(3)
        callbacks.uninstallcallback("afterstep", count_calls)
    finally:
        stop.set()
        thread.join()

    ticks_in_step = sum(start < t < end for t in ticks)
    assert ticks_in_step >= 10, (
        f"another Python thread ran only {ticks_in_step} times during "
        f"sim.step({nsteps}), which took {end - start:.3f} s"
    )
    assert len(calls) == 3
