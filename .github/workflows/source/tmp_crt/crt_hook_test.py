# TEMPORARY: test AMReX's MSVC debug CRT report hooks (amrex.handle_crt_reports)
import ctypes
import sys

import amrex.space3d as amr

case = sys.argv[1]
ucrtbased = ctypes.CDLL("ucrtbased.dll")
_CRT_ASSERT = 2

amr.initialize([])
print(f"case {case}: AMReX initialized", flush=True)
if case == "finalized":
    amr.finalize()
    print("AMReX finalized", flush=True)

if case == "abort":
    ucrtbased.abort()
else:
    r = ucrtbased._CrtDbgReport(_CRT_ASSERT, None, 0, None, b"tmp test assertion\n")
    print("_CrtDbgReport(_CRT_ASSERT) returned", r, flush=True)
print(f"case {case}: done", flush=True)
