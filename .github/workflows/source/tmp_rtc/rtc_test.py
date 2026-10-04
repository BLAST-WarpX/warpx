# TEMPORARY: which debug-CRT failures open blocking dialogs on the runner?
import ctypes
import os
import sys

case = sys.argv[1]
ucrtbased = ctypes.CDLL("ucrtbased.dll")
print(f"case {case}: start", flush=True)
if case == "rtc":
    dll = ctypes.CDLL(os.path.abspath("rtc_test.dll"))
    print("rtc_uninit(0) returned", dll.rtc_uninit(0), flush=True)
elif case == "assert":
    r = ucrtbased._CrtDbgReport(2, None, 0, None, b"test: CRT assert\n")
    print("_CrtDbgReport(_CRT_ASSERT) returned", r, flush=True)
elif case == "abort":
    ucrtbased.abort()
print(f"case {case}: done", flush=True)
