"""Run the remaining pipeline unattended, one stage after another.

  python scripts/overnight.py --wait-pid <PID of the running eval sweep>

Stages (each logged to outputs/overnight/<stage>.log; a failed stage does not stop
the next one). The machine is kept awake for the whole run.
"""
import argparse
import ctypes
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG_DIR = ROOT / "outputs" / "overnight"

STAGES = [
    ("0_finish_sweep", [PY, "-W", "ignore", "scripts/run_eval_inference.py",
                        "--settings", "full", "512", "256", "128"]),
    ("1_score_sweep", [PY, "-W", "ignore", "scripts/evaluate.py",
                       "--settings", "full", "512", "256", "128"]),
    ("2_profile", [PY, "-W", "ignore", "scripts/profile_decode.py", "--tokens", "7000",
                   "--settings", "full", "512", "ring512", "ring128", "--repeats", "3",
                   "--out", "outputs/profile_final"]),
    ("3_multipage_run", [PY, "-W", "ignore", "scripts/multipage_eval.py", "run"]),
    ("4_multipage_score", [PY, "-W", "ignore", "scripts/multipage_eval.py", "score"]),
]


def log(msg):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "overnight.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid):
    h = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
    if not h:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0)
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from keep_awake import hold
    hold()

    if args.wait_pid:
        log(f"WAIT for sweep PID {args.wait_pid}")
        while pid_alive(args.wait_pid):
            time.sleep(30)
    env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8",
           "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    for name, cmd in STAGES:
        log(f"STAGE {name} start")
        t = time.time()
        with open(LOG_DIR / f"{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        log(f"STAGE {name} {'OK' if rc == 0 else f'FAILED rc={rc}'} in {(time.time() - t) / 60:.1f} min")
    log("ALL DONE")


if __name__ == "__main__":
    main()
