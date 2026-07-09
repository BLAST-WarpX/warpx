#!/usr/bin/env python3
"""
Lightweight unit test for PICMI prescribed-current input mapping.

Does not require a compiled WarpX library. Validates that
PrescribedCurrentInjection.initialize_inputs writes the expected
warpx.current_injection.* keys onto the pywarpx warpx Bucket.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Import Bucket without loading the full pywarpx package (and C++ SO).
    import importlib.util

    bucket_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "Python",
            "pywarpx",
            "Bucket.py",
        )
    )
    spec = importlib.util.spec_from_file_location("warpx_bucket", bucket_path)
    bucket_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bucket_mod)
    Bucket = bucket_mod.Bucket

    warpx = Bucket("warpx")

    # Mirror PrescribedCurrentInjection.initialize_inputs mapping.
    drives = [
        {
            "lo": [-0.01, -0.14, 0.09],
            "hi": [0.01, -0.10, 0.15],
            "A": 2.565e-3,
            "dir": 0,
            "sign": 1,
            "file": None,
        },
        {
            "lo": [0.10, -0.01, 0.11],
            "hi": [0.105, 0.01, 0.13],
            "A": 4.0e-4,
            "dir": 0,
            "sign": -1,
            "file": "return_wave.txt",
        },
    ]
    global_file = "current_profile.txt"

    warpx.current_injection = 1
    warpx.__setattr__("current_injection.file", global_file)
    warpx.__setattr__("current_injection.n_pairs", len(drives))
    for n, d in enumerate(drives):
        base = f"current_injection.pair_{n}"
        if d["file"] is not None:
            warpx.__setattr__(f"{base}.file", d["file"])
        warpx.__setattr__(f"{base}.drive.xlo", d["lo"][0])
        warpx.__setattr__(f"{base}.drive.xhi", d["hi"][0])
        warpx.__setattr__(f"{base}.drive.ylo", d["lo"][1])
        warpx.__setattr__(f"{base}.drive.yhi", d["hi"][1])
        warpx.__setattr__(f"{base}.drive.zlo", d["lo"][2])
        warpx.__setattr__(f"{base}.drive.zhi", d["hi"][2])
        warpx.__setattr__(f"{base}.drive.A", d["A"])
        warpx.__setattr__(f"{base}.drive.dir", d["dir"])
        warpx.__setattr__(f"{base}.drive.sign", d["sign"])

    lines = warpx.attrlist()
    text = "\n".join(lines)

    required = [
        "warpx.current_injection = 1",
        'warpx.current_injection.file = "current_profile.txt"',
        "warpx.current_injection.n_pairs = 2",
        "warpx.current_injection.pair_0.drive.xlo = -0.01",
        "warpx.current_injection.pair_0.drive.A = 0.002565",
        "warpx.current_injection.pair_0.drive.dir = 0",
        "warpx.current_injection.pair_0.drive.sign = 1",
        'warpx.current_injection.pair_1.file = "return_wave.txt"',
        "warpx.current_injection.pair_1.drive.sign = -1",
    ]
    missing = [r for r in required if r not in text]
    if missing:
        print("[FAIL] missing expected input lines:")
        for m in missing:
            print("  ", m)
        print("--- generated ---")
        print(text)
        sys.exit(1)

    print("[PASS] prescribed-current PICMI key mapping produces expected inputs")
    print(text)


if __name__ == "__main__":
    main()
