# TEMPORARY: self-test for run_without_crt_dialogs.py
import ctypes
import os
import sys

ucrtbased = ctypes.CDLL("ucrtbased.dll")
r = ucrtbased._CrtDbgReport(2, None, 0, None, b"self-test: CRT assert report\n")
print("_CrtDbgReport(_CRT_ASSERT) returned", r, flush=True)
r = ucrtbased._CrtDbgReport(1, None, 0, None, b"self-test: CRT error report\n")
print("_CrtDbgReport(_CRT_ERROR) returned", r, flush=True)

dll = ctypes.CDLL(os.path.abspath("rtc_test.dll"))
print("loaded rtc_test.dll", flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "init":
    print("rtc_uninit(1) returned", dll.rtc_uninit(1), flush=True)
print("calling rtc_uninit(0) ...", flush=True)
print("rtc_uninit(0) returned", dll.rtc_uninit(0), flush=True)
print("self-test passed: no dialog blocked", flush=True)
