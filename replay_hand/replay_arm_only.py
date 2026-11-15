#!/usr/bin/env python3
"""Only Tron2 arms; never initializes or controls Wuji hands."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT/'replay'), str(ROOT/'tools')]
from isolated_cli import main
from arm_only import run

if __name__ == '__main__':
    raise SystemExit(main('arms', run))
