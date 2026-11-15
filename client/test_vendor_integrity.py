import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np


CLIENT = Path(__file__).resolve().parent
VENDOR = CLIENT / "vendor/keypoints21_to_wujihand2"
EXPECTED_TREE_SHA256 = "f3fb5e0ec0b088ddce5669b45b47a8ae24f3b62b2492864b016232cc34366485"
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def vendored_files():
    files = [VENDOR / name for name in (
        "retarget.py", "convert.py", "README.md",
        "WUJI_MODEL_LICENSE", "WUJI_VENDORED_LICENSE")]
    files.extend(sorted((VENDOR / "assets/wuji").rglob("*")))
    return sorted((path for path in files if path.is_file()),
                  key=lambda path: str(path.relative_to(VENDOR)))


def tree_sha256():
    digest = hashlib.sha256()
    for path in vendored_files():
        relative = str(path.relative_to(VENDOR))
        digest.update(relative.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def vendor_module():
    os.environ.setdefault("MUJOCO_GL", "disable")
    spec = importlib.util.spec_from_file_location("xflow_vendor_retarget", VENDOR / "retarget.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def model_names(side):
    result = []
    for finger in FINGERS:
        suffixes = (("cmc_flex", "cmc_abd", "mcp", "ip") if finger == "thumb"
                    else ("mcp_flex", "mcp_abd", "pip", "dip"))
        result.extend("%s_%s_%s" % (side[0], finger, suffix) for suffix in suffixes)
    return result


class VendorIntegrityTests(unittest.TestCase):
    def test_complete_vendor_tree_is_byte_identical(self):
        self.assertEqual(len(vendored_files()), 65)
        self.assertEqual(tree_sha256(), EXPECTED_TREE_SHA256)

    def test_joint_order_and_limits_match_installed_wuji_control(self):
        module = vendor_module()
        limits_path = CLIENT.parent / "install/wuji_control/share/wuji_control/config/joint_limits.json"
        limits = json.loads(limits_path.read_text())
        for side in ("left", "right"):
            hand = module.Hand("wuji", side)
            self.assertEqual(hand.names, model_names(side))
            firmware_names = ["%s_finger%d_joint%d" % (side, finger, joint)
                              for finger in range(1, 6) for joint in range(1, 5)]
            self.assertEqual(len(firmware_names), len(hand.names))
            self.assertEqual(len(set(firmware_names)), 20)
            np.testing.assert_allclose(np.c_[hand.lo, hand.hi], limits[side], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
