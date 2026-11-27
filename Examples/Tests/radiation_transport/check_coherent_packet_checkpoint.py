#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Exercise packet counter and vector-model restart guards on disposable copies.

The supplied input/overrides must reproduce the producer's coherent mode.
No producer checkpoint is edited. A timeout is a failure, not a rejection.
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from check_moving_moment_checkpoint import run_rejection

MARKER = "chronological_coherent_packets_v1"
MANIFEST = "RadiationCoherentPackets_data.txt"
ACCOUNTING = "RadiationTransport_data.txt"
UINT64_MAX = 2**64 - 1
CASES = (
    "mode_disabled",
    "missing_manifest",
    "wrong_marker",
    "truncated_counter",
    "negative_counter",
    "plus_counter",
    "overflow_counter",
    "trailing_counter",
    "missing_accounting",
    "missing_residual_seed",
    "missing_seed",
    "missing_carry_owner",
    "counter_exhausted",
)


def diagnostic_text(text):
    # WarpX wraps assertion messages with '#            ' continuation lines.
    return " ".join(re.sub(r"\n\s*#\s*", " ", text).split())


def check_moving_diffusion(
    executable, checkpoint, inputs, next_step, parameters, timeout
):
    name = "RadiationMovingDiffusion_data.txt"
    original = (checkpoint / name).read_bytes()
    assert original.decode().split() == ["moving_asymptotic_v1"]
    if next_step is None:
        match = re.search(r"(\d+)$", checkpoint.name)
        assert match, "A checkpoint without a step suffix needs --next-step"
        next_step = int(match.group(1))
    for case in ("missing", "wrong_model", "trailing", "switch_to_explicit"):
        with tempfile.TemporaryDirectory(
            prefix="moving-diffusion-model-", dir="."
        ) as directory:
            candidate = Path(directory).resolve() / "checkpoint"
            shutil.copytree(checkpoint, candidate)
            manifest = candidate / name
            overrides = []
            if case == "missing":
                manifest.unlink()
            elif case == "wrong_model":
                manifest.write_text("unrecognized_moving_diffusion_v999\n")
            elif case == "trailing":
                manifest.write_text("moving_asymptotic_v1 unexpected\n")
            else:
                overrides.append("radiation_transport.diffusion_solver=explicit")
            expected = (
                "Restart must preserve the moving diffusion model and its manifest."
                if case in ("missing", "switch_to_explicit")
                else "Invalid moving diffusion checkpoint model metadata."
            )
            result = run_rejection(
                [
                    str(executable),
                    str(inputs),
                    *parameters,
                    *overrides,
                    f"amr.restart={candidate}",
                    f"max_step={next_step}",
                    "amrex.throw_exception=0",
                    "amrex.call_addr2line=0",
                    "amrex.the_arena_init_size=0",
                ],
                Path(f"moving_diffusion_checkpoint_{case}.log"),
                timeout=timeout,
            )
            assert result.returncode != 0, result.stdout
            assert expected in diagnostic_text(result.stdout), result.stdout
        assert (checkpoint / name).read_bytes() == original, (
            "Producer model metadata changed"
        )
        print(f"moving_diffusion_{case}: rejected during restart model validation")


def check_m1_switch(executable, checkpoint, inputs, next_step, parameters, timeout):
    manifest = checkpoint / "RadiationFluxRelaxed_data.txt"
    original = manifest.read_bytes()
    assert original.decode().startswith("limited_m1_packet_state_v1 ")
    if next_step is None:
        match = re.search(r"(\d+)$", checkpoint.name)
        assert match, "A checkpoint without a step suffix needs --next-step"
        next_step = int(match.group(1))  # Restart validation, no requested advance.
    with tempfile.TemporaryDirectory(
        prefix="m1-model-checkpoint-", dir="."
    ) as directory:
        candidate = Path(directory).resolve() / "checkpoint"
        shutil.copytree(checkpoint, candidate)
        result = run_rejection(
            [
                str(executable),
                str(inputs),
                *parameters,
                "radiation_transport.flux_relaxed_transport=implicit_p1",
                f"amr.restart={candidate}",
                f"max_step={next_step}",
                "amrex.throw_exception=0",
                "amrex.call_addr2line=0",
                "amrex.the_arena_init_size=0",
                "warpx.verbose=1",
            ],
            Path("checkpoint_limited_m1_to_p1.log"),
            timeout=timeout,
        )
        assert result.returncode != 0, result.stdout
        assert (
            "Invalid or mismatched flux-relaxed checkpoint metadata."
            in diagnostic_text(result.stdout)
        ), result.stdout
    assert manifest.read_bytes() == original, "Producer model metadata changed"
    print("limited_m1_to_p1: rejected during restart model validation")


def corrupt(candidate, case, counter):
    manifest = candidate / MANIFEST
    accounting = candidate / ACCOUNTING
    if case == "mode_disabled":
        return (
            "Restart must preserve the coherent-packet mode and event-counter manifest."
        )
    if case == "missing_manifest":
        manifest.unlink()
        return (
            "Restart must preserve the coherent-packet mode and event-counter manifest."
        )
    if case in ("missing_accounting", "missing_residual_seed", "missing_seed"):
        if case == "missing_accounting":
            accounting.unlink()
        else:
            lines = accounting.read_text().splitlines()
            assert len(lines) >= 3, "Producer must contain complete seed accounting"
            count = 1 if case == "missing_residual_seed" else 2
            accounting.write_text("\n".join(lines[:count]) + "\n")
        return "Coherent-packet restart requires complete resolved-seed checkpoint accounting."
    if case == "counter_exhausted":
        manifest.write_text(f"{MARKER} {UINT64_MAX}\n")
        return "Coherent packet accepted-segment counter overflow."
    if case == "missing_carry_owner":
        (candidate / "RadiationParticleCarry_data.txt").unlink()
        return "Restart must preserve radiation momentum_carry ownership"
    if case == "wrong_marker":
        text = f"unrecognized_coherent_packets_v999 {counter}\n"
    elif case == "truncated_counter":
        text = f"{MARKER}\n"
    elif case == "negative_counter":
        text = f"{MARKER} -1\n"
    elif case == "plus_counter":
        text = f"{MARKER} +{counter}\n"
    elif case == "overflow_counter":
        text = f"{MARKER} {UINT64_MAX + 1}\n"
    elif case == "trailing_counter":
        text = f"{MARKER} {counter} unexpected\n"
    else:
        raise ValueError(case)
    manifest.write_text(text)
    return (
        "Invalid coherent-packet event counter."
        if case == "overflow_counter"
        else "Invalid coherent-packet checkpoint manifest."
    )


def check(executable, checkpoint, inputs, cases, next_step, parameters, timeout):
    executable = executable.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    inputs = inputs.resolve(strict=True)
    if cases == ("limited_m1_to_p1",):
        check_m1_switch(executable, checkpoint, inputs, next_step, parameters, timeout)
        return
    if cases == ("moving_diffusion_models",):
        check_moving_diffusion(
            executable, checkpoint, inputs, next_step, parameters, timeout
        )
        return
    originals = {
        name: (checkpoint / name).read_bytes() for name in (MANIFEST, ACCOUNTING)
    }
    tokens = originals[MANIFEST].decode().split()
    assert len(tokens) == 2 and tokens[0] == MARKER
    assert tokens[1].isascii() and tokens[1].isdigit()
    counter = int(tokens[1])
    assert 0 <= counter <= UINT64_MAX
    if next_step is None:
        match = re.search(r"(\d+)$", checkpoint.name)
        assert match, "A checkpoint without a step suffix needs --next-step"
        next_step = int(match.group(1)) + 1
    assert next_step > 0
    for parameter in parameters:
        assert not parameter.startswith(("amr.restart=", "max_step=")), parameter
    for case in cases:
        with tempfile.TemporaryDirectory(
            prefix="coherent-checkpoint-", dir="."
        ) as directory:
            candidate = Path(directory).resolve() / "checkpoint"
            shutil.copytree(checkpoint, candidate)
            expected = corrupt(candidate, case, counter)
            log = Path(f"coherent_checkpoint_{case}.log")
            result = run_rejection(
                [
                    str(executable),
                    str(inputs),
                    *parameters,
                    *(
                        ["radiation_transport.coherent_packet_transport=0"]
                        if case == "mode_disabled"
                        else []
                    ),
                    f"amr.restart={candidate}",
                    f"max_step={next_step}",
                    "radiation_transport.coherent_verbosity=1",
                    "amrex.throw_exception=0",
                    "amrex.call_addr2line=0",
                    "amrex.the_arena_init_size=0",
                    "warpx.verbose=1",
                ],
                log,
                timeout=timeout,
            )
            assert result.returncode != 0, result.stdout
            assert expected in diagnostic_text(result.stdout), result.stdout
            assert "Coherent packets accepted_substeps=" not in result.stdout, (
                result.stdout
            )
            assert f"STEP {next_step} ends" not in result.stdout, result.stdout
        for name, original in originals.items():
            assert (checkpoint / name).read_bytes() == original, (
                "Producer metadata changed"
            )
        print(f"{case}: rejected before any accepted coherent interval; log={log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument(
        "case", choices=(*CASES, "all", "limited_m1_to_p1", "moving_diffusion_models")
    )
    parser.add_argument("--next-step", type=int)
    parser.add_argument(
        "--param", action="append", default=[], help="Producer-compatible key=value"
    )
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    assert args.timeout > 0
    check(
        args.executable,
        args.checkpoint,
        args.inputs,
        CASES if args.case == "all" else (args.case,),
        args.next_step,
        args.param,
        args.timeout,
    )
