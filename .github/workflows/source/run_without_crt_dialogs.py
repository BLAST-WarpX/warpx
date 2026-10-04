#! /usr/bin/env python3
#
# Run a Python script on Windows CI without blocking error dialogs.
#
# Debug builds with MSVC (/MDd, /RTC1) report failed CRT/STL assertions,
# run-time check failures (e.g., the use of an uninitialized variable) and
# abort() in a modal dialog box. On a CI runner, nobody can click it away and
# the job hangs until it is killed, without ever printing the error. There is
# no environment variable or registry setting to change this, only the
# in-process CRT API.
#
# This script installs a CRT report hook that prints such reports to stderr
# and then breaks into the debugger. Without a debugger attached, this
# terminates the process with exit code 0x80000003 (STATUS_BREAKPOINT), so the
# CI job fails at the first error. If Windows Error Reporting is configured to
# write local crash dumps, the C++ call stack can then be printed from the
# dump, e.g., with `cdb -z <file.dmp> -c ".ecxr; kpn; q"`.
#
# Known limitation: the hook is a Python function, so it needs the GIL. A
# report from a non-Python thread (e.g., OpenMP) while the GIL is held
# elsewhere would block. This is fine for builds without OpenMP.
#
# Usage: python3 -X faulthandler run_without_crt_dialogs.py <script.py> [args ...]

import ctypes
import os
import runpy
import sys

if sys.platform != "win32":
    raise RuntimeError("This helper is only meant for Windows.")

if len(sys.argv) < 2:
    raise RuntimeError(f"Usage: {sys.argv[0]} <script.py> [script args ...]")

# Windows error mode: no dialogs for critical errors. We do not set
# SEM_NOGPFAULTERRORBOX, because it disables Windows Error Reporting (WER)
# completely, including local crash dumps. Instead, disable the WER UI with
# the registry value `DontShowUI` (see .github/workflows/windows.yml).
SEM_FAILCRITICALERRORS = 0x0001
SEM_NOOPENFILEERRORBOX = 0x8000
ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX)

# The debug CRT (ucrtbased.dll) is shared by all /MDd modules in the process,
# e.g., the pyAMReX and pyWarpX extension modules that are imported later.
ucrtbased = ctypes.CDLL("ucrtbased.dll")

# crtdbg.h
_CRT_WARN = 0
_CRT_ERROR = 1
_CRT_ASSERT = 2
_CRT_REPORT_TYPES = {_CRT_WARN: "warning", _CRT_ERROR: "error", _CRT_ASSERT: "assert"}
_CRTDBG_MODE_FILE = 0x1
_CRTDBG_FILE_STDERR = ctypes.c_void_p(-5)
_CRT_RPTHOOK_INSTALL = 0

# Fallback, in case a report does not reach the hooks below: print to stderr
# instead of opening a dialog box
ucrtbased._CrtSetReportFile.argtypes = [ctypes.c_int, ctypes.c_void_p]
ucrtbased._CrtSetReportFile.restype = ctypes.c_void_p
for report_type in _CRT_REPORT_TYPES:
    ucrtbased._CrtSetReportMode(report_type, _CRTDBG_MODE_FILE)
    ucrtbased._CrtSetReportFile(report_type, _CRTDBG_FILE_STDERR)


def _report(report_type, message, return_value):
    """Print a CRT report; break (and thus fail) on errors and assertions"""
    sys.stderr.write(
        f"\nMSVC debug CRT {_CRT_REPORT_TYPES.get(report_type, report_type)}: "
        f"{message.rstrip()}\n"
    )
    sys.stderr.flush()
    # 1: the reporting function calls _CrtDbgBreak()/__debugbreak()
    return_value[0] = 0 if report_type == _CRT_WARN else 1
    # TRUE: the report is handled, no further output or dialog box
    return 1


# keep references to the callbacks for the lifetime of the process
_report_hook = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
)(lambda t, m, r: _report(t, m.decode(errors="replace"), r))
_report_hook_w = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)
)(_report)
for set_hook, hook in [
    (ucrtbased._CrtSetReportHook2, _report_hook),
    (ucrtbased._CrtSetReportHookW2, _report_hook_w),
]:
    if set_hook(_CRT_RPTHOOK_INSTALL, hook) == -1:
        raise RuntimeError(f"{set_hook.__name__} failed")

# stdlib.h: write runtime errors to stderr; abort() reports via the hook
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
