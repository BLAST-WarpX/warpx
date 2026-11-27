#!/usr/bin/env python3
"""Normal WarpX Evolve/checkpoint/restart of the isolated native pressure owner."""

import os
import pathlib
import shutil
import subprocess
import sys

binary, inputs = map(pathlib.Path, sys.argv[1:3])
grid = sys.argv[3]
mpi = sys.argv[4:]  # executable and its process-count flag, if MPI is enabled


def run(name, *arguments, ranks=1, fail=None):
    folder = pathlib.Path(name).absolute()
    folder.mkdir(exist_ok=True)
    command = [str(binary), str(inputs), f"warpx.grid_type={grid}", *arguments]
    if ranks > 1:
        command = [*mpi, str(ranks), *command]
    result = subprocess.run(
        command,
        cwd=folder,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    if fail:
        assert result.returncode != 0, result.stdout
        assert fail in result.stdout, result.stdout
    else:
        assert result.returncode == 0, result.stdout
        assert "ordinary native pressure_only energy audit passed" in result.stdout
    return folder


def state(folder, filename="state.txt"):
    result = {}
    for line in (folder / filename).read_text().splitlines():
        name, *values = line.split()
        if name in {"step", "substeps", "energy", "kinetic", "wall_work"}:
            result[name] = float(values[0])
        else:
            size, *numbers = values
            assert int(size) == len(numbers)
            result[name] = tuple(map(float, numbers))
    return result


def same(a, b):
    for name in (
        "step",
        "substeps",
        "B0",
        "B1",
        "B2",
        "E0",
        "E1",
        "E2",
        "U",
        "T",
        "rho",
        "V",
    ):
        assert a[name] == b[name], f"Restart changed {name}"
    assert a["kinetic"] == b["kinetic"]
    assert a["wall_work"] == b["wall_work"] == 0


seed = run("seed", "max_step=0", "test.initial_checkpoint=1")
initial_checkpoint = seed / "diags" / "chk000000"
assert initial_checkpoint.is_dir()
baseline = run("baseline", f"amr.restart={initial_checkpoint}")
partial = run("partial", f"amr.restart={initial_checkpoint}", "max_step=2")
same(state(seed), state(baseline, "initial_state.txt"))
same(state(seed), state(partial, "initial_state.txt"))
checkpoint = partial / "diags" / "chk000002"
assert checkpoint.is_dir()
manifest = checkpoint / "HybridNativePressure.txt"
assert manifest.is_file()
restored = run("restart", f"amr.restart={checkpoint}")
same(state(partial), state(restored, "initial_state.txt"))
same(state(baseline), state(restored))
assert state(partial)["substeps"] > 2
assert state(baseline)["B1"] != state(baseline, "initial_state.txt")["B1"]
assert state(baseline)["U"] != state(baseline, "initial_state.txt")["U"]
assert state(baseline)["kinetic"] == state(baseline, "initial_state.txt")["kinetic"]
assert min(state(baseline)["rho"]) > 0
run("rollback", "test.rollback=1")
if mpi:
    changed_rank = run("restart_changed_rank", f"amr.restart={checkpoint}", ranks=2)
    same(state(partial), state(changed_rank, "initial_state.txt"))
    same(state(baseline), state(changed_rank))
    run("rollback_mpi", "test.rollback=1", ranks=2)

run(
    "changed_channel",
    f"amr.restart={checkpoint}",
    "hybrid_pic_model.field_evolution_channel=full",
    "hybrid_pic_model.solve_electron_energy_equation=1",
    fail="preserve the native pressure_only field channel",
)
run(
    "changed_boundary",
    f"amr.restart={checkpoint}",
    "boundary.field_hi=pmc periodic",
    fail="native pressure_only stencil, EOS, boundary",
)
run(
    "changed_stencil",
    f"amr.restart={checkpoint}",
    f"warpx.grid_type={'staggered' if grid == 'collocated' else 'collocated'}",
    fail="native pressure_only stencil, EOS, boundary",
)
run(
    "changed_controller",
    f"amr.restart={checkpoint}",
    "hybrid_pic_model.substep_rtol=1.e-8",
    fail="native pressure_only stencil, EOS, boundary",
)
missing_owner = pathlib.Path("missing_owner_checkpoint").absolute()
shutil.copytree(checkpoint, missing_owner, dirs_exist_ok=True)
owner = missing_owner / "Level_0" / "hybrid_native_pressure_energy_fp[level=0]_H"
owner.rename(owner.with_name(owner.name + ".withheld"))
run(
    "missing_owner",
    f"amr.restart={missing_owner}",
    fail="Missing required native pressure_only checkpoint owner",
)
run("moving_ions", "ions.do_not_push=0", fail="requires every species to be frozen")
run(
    "resistivity",
    "hybrid_pic_model.plasma_resistivity(rho,J,t)=1",
    fail="pressure_only requires double-precision ideal electrons",
)
print(
    f"Native {grid} ordinary evolution, exact normal restart, changed-rank and rejection gates passed"
)
