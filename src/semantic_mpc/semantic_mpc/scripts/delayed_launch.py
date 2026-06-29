#!/usr/bin/env python
import subprocess
import sys
import time


def main():
    if len(sys.argv) < 3:
        print("usage: delayed_launch.py <delay_seconds> <command> [args...]", file=sys.stderr)
        return 2

    try:
        delay_seconds = float(sys.argv[1])
    except ValueError:
        print("invalid delay_seconds: {}".format(sys.argv[1]), file=sys.stderr)
        return 2

    time.sleep(delay_seconds)
    return subprocess.call(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
