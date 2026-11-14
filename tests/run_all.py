#!/usr/bin/env python3
"""Run every test in this directory and report a one-line summary each."""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    tests = sorted(f for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py"))
    if not tests:
        print("No tests found.")
        return 1

    failures = []
    for name in tests:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-I", os.path.join(HERE, name)],
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        ok = proc.returncode == 0
        print(f"{name:22s} {'PASS' if ok else 'FAIL'}  {dt:5.1f}s")
        if not ok:
            failures.append((name, proc.stdout, proc.stderr))

    if failures:
        for name, out, err in failures:
            print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
            if out.strip():
                print(out.rstrip())
            if err.strip():
                print(err.rstrip())
        print(f"\n{len(failures)} of {len(tests)} failed.")
        return 1

    print(f"\nAll {len(tests)} passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
