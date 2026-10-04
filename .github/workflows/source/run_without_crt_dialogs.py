#! /usr/bin/env python3
#
# Run a Python script on Windows CI without blocking error dialogs.
#
# Debug builds with MSVC (/MDd, /RTC1) report failed CRT/STL assertions,
# run-time check failures and abort() in a modal dialog box. On a headless CI
# runner, nobody can click it away and the job hangs until it is killed,
# without ever printing the error. This sends these reports to stderr instead,
# similar to what CPython's test suite does in test.support.SuppressCrashReport.
#
# Usage: python3 run_without_crt_dialogs.py <script.py> [script args ...]

import ctypes
import os
import runpy
import sys

if sys.platform != "win32":
    raise RuntimeError("This helper is only meant for Windows.")

if len(sys.argv) < 2:
    raise RuntimeError(f"Usage: {sys.argv[0]} <script.py> [script args ...]")

# Windows error mode: no dialogs for crashes (WER) and critical errors
SEM_FAILCRITICALERRORS = 0x0001
SEM_NOGPFAULTERRORBOX = 0x0002
SEM_NOOPENFILEERRORBOX = 0x8000
ctypes.windll.kernel32.SetErrorMode(
    SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
)

# The debug CRT (ucrtbased.dll) is shared by all /MDd modules in the process,
# e.g., the pyAMReX and pyWarpX extension modules that are imported later.
ucrtbased = ctypes.CDLL("ucrtbased.dll")

# crtdbg.h
_CRT_WARN = 0
_CRT_ERROR = 1
_CRT_ASSERT = 2
_CRTDBG_MODE_FILE = 0x1
_CRTDBG_FILE_STDERR = ctypes.c_void_p(-5)
ucrtbased._CrtSetReportFile.argtypes = [ctypes.c_int, ctypes.c_void_p]
ucrtbased._CrtSetReportFile.restype = ctypes.c_void_p
for report_type in [_CRT_WARN, _CRT_ERROR, _CRT_ASSERT]:
    ucrtbased._CrtSetReportMode(report_type, _CRTDBG_MODE_FILE)
    ucrtbased._CrtSetReportFile(report_type, _CRTDBG_FILE_STDERR)

# stdlib.h: write runtime errors to stderr and do not open the
# "abort() has been called" dialog
_OUT_TO_STDERR = 1
_WRITE_ABORT_MSG = 0x1
_CALL_REPORTFAULT = 0x2
ucrtbased._set_error_mode(_OUT_TO_STDERR)
ucrtbased._set_abort_behavior(_WRITE_ABORT_MSG, _WRITE_ABORT_MSG | _CALL_REPORTFAULT)

# run the actual script as if it was called directly
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")
