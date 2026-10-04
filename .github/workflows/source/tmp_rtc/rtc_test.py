# TEMPORARY: which debug-CRT failures open blocking dialogs on the runner?
import ctypes
import os
import sys

case = sys.argv[1]
print(f"case {case}: start", flush=True)
dll = ctypes.CDLL(os.path.abspath("rtc_test.dll"))
if case == "rtc":
    print("rtc_uninit(0) returned", dll.rtc_uninit(0), flush=True)
elif case == "segfault":
    dll.segfault(None)
print(f"case {case}: done", flush=True)
