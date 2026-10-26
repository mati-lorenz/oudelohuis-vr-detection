# -*- coding: utf-8 -*-
"""
run_3x.py -- run the pair-analysis scripts 3a ... 3g one after the other, unattended.

    python run_3x.py                              all of 3a-3g, default settings
    python run_3x.py --n-jobs 4                   sessions in parallel per script (RAM: one session per worker)
    python run_3x.py --aggregate mouse            extra arguments are passed to every script
    python run_3x.py --animals DNL01 DNL02        (restricted run -> out/selection/<tag>/)
    python run_3x.py --only 3a 3g                 only some scripts
    python run_3x.py --skip 3e 3f                 all except these
    python run_3x.py --dry-run                    print the commands without running

Order and dependencies: 3a first (writes the pair list that 3b-3f read, so if 3a fails 3b-3f are skipped; 3g needs only
2a/2b and still runs). Scripts run strictly one at a time (never two sessions' worth of RAM x two scripts). A script that
fails does not stop the others. Everything is cached per session in each script's store/, so after an interruption (or a
crash, or a closed laptop) just run the same command again: finished sessions are loaded, only the rest is computed.
A log per script is written to logs/3x_<timestamp>/<script>.log and a summary to logs/3x_<timestamp>/summary.txt.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "3a": "3a_pair_activity_statistics.py",
    "3b": "3b_pair_information_breakdown.py",
    "3c": "3c_pair_information_pid.py",
    "3d": "3d_pair_activity_statistics_temporal.py",
    "3e": "3e_pair_information_breakdown_temporal.py",
    "3f": "3f_pair_information_pid_temporal.py",
    "3g": "3g_pair_hit_miss_correlations.py",
}
NEEDS_3A = {"3b", "3c", "3d", "3e", "3f"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--only", nargs="+", choices=sorted(SCRIPTS))
ap.add_argument("--skip", nargs="+", choices=sorted(SCRIPTS), default=[])
ap.add_argument("--dry-run", action="store_true")
args, extra = ap.parse_known_args()      # everything else goes to the scripts
todo = [k for k in SCRIPTS if (not args.only or k in args.only) and k not in args.skip]

env = dict(os.environ, PYTHONPATH=str(HERE / "utils") + os.pathsep + os.environ.get("PYTHONPATH", ""),
           PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
logdir = HERE / "logs" / f"3x_{datetime.now():%Y%m%d_%H%M%S}"
if not args.dry_run:
    logdir.mkdir(parents=True, exist_ok=True)
print(f"Scripts: {' '.join(todo)}   extra arguments: {' '.join(extra) or '(none)'}")
if not args.dry_run:
    print(f"Logs: {logdir}")

results, failed = {}, set()
for key in todo:
    cmd = [sys.executable, str(HERE / SCRIPTS[key])] + extra
    if key in NEEDS_3A and "3a" in failed:
        results[key] = ("SKIPPED (3a failed)", 0.0)
        print(f"[{key}] skipped: 3a failed")
        continue
    print(f"[{key}] {datetime.now():%H:%M:%S}  {' '.join(cmd[1:])}", flush=True)
    if args.dry_run:
        continue
    t0 = time.time()
    with open(logdir / f"{SCRIPTS[key]}.log", "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    results[key] = ("OK" if code == 0 else f"FAILED (exit {code}, see log)", dt)
    if code != 0:
        failed.add(key)
    print(f"[{key}] {results[key][0]} in {dt / 60:.1f} min", flush=True)

if not args.dry_run:
    lines = [f"{k}  {SCRIPTS[k]:<45} {v[0]:<28} {v[1] / 60:6.1f} min" for k, v in results.items()]
    (logdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    sys.exit(1 if failed else 0)
