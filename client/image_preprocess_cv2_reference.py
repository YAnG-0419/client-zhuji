#!/usr/bin/env python3
"""System-Python/OpenCV cross-check; read-only, no ROS graph mutation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from image_preprocess import ImagePreprocessor, camera_info_dict, config_section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("deployment_config.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = config_section(args.config)
    value = config["zhuji_live_reference"]
    camera = camera_info_dict(value["width"], value["height"],
                              value["distortion_model"], value["K"], value["D"],
                              value["frame_id"])
    processor = ImagePreprocessor(camera, config)
    virtual_K = np.asarray(config["cam0"]["virtual_K_320"], float).reshape(3, 3)
    ref_x, ref_y = cv2.initUndistortRectifyMap(
        camera["K"], camera["D"], None, virtual_K, (320, 320), cv2.CV_32FC1)
    report = {
        "opencv_version": cv2.__version__,
        "max_abs_map_x_px": float(np.max(np.abs(ref_x - processor.virtual_map[0]))),
        "max_abs_map_y_px": float(np.max(np.abs(ref_y - processor.virtual_map[1]))),
        "threshold_px": 0.05,
        "passed": bool(np.max(np.abs(ref_x - processor.virtual_map[0])) <= 0.05
                       and np.max(np.abs(ref_y - processor.virtual_map[1])) <= 0.05),
        "hardware_commands_sent": 0,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

