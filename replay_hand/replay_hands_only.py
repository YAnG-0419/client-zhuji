#!/usr/bin/env python3
"""Only Wuji hands; never connects to Tron2 or loads an arm model."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT/'tools')]
from isolated_cli import main
from hand_only import run

if __name__ == '__main__':
    raise SystemExit(main('hands', run))
