"""Operator-selected replay thresholds, updated 2026-11-11.

Checked collision pairs require 5 mm, except same-wrist yaw/roll at 1 mm.
Pose errors apply to SDK EEF targets. Units are meters and radians.
"""
import math

IK_POSITION_TOL = .020
IK_ROTATION_TOL = math.radians(2.)
CHORD_ROTATION_TOL = math.radians(3.)
MODEL_CLEARANCE_M = .005
INTERNAL_WRIST_CLEARANCE_M = .001
LIVE_CLEARANCE_M = MODEL_CLEARANCE_M
