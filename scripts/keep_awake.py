"""Keep Windows awake (no idle sleep / Modern Standby) while a process is running.

Uses SetThreadExecutionState, the same per-application request video players use.
It changes no power settings and is released automatically when this script exits.
Closing the lid or choosing Sleep still sleeps the machine.

  python scripts/keep_awake.py --pid 1234     # until process 1234 exits
"""
import argparse
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002  # needed on Modern Standby laptops: screen-off -> standby


def hold():
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


def release():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def pid_alive(pid: int) -> bool:
    SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x102
    h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not h:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    args = ap.parse_args()
    hold()
    try:
        while pid_alive(args.pid):
            time.sleep(30)
    finally:
        release()
