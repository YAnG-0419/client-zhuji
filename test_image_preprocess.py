import json
from pathlib import Path
import unittest

import numpy as np

from image_preprocess import (ImagePreprocessor, camera_info_dict,
                              config_section, resize_area,
                              validate_camera_info)


HERE = Path(__file__).resolve().parent


def live_info(config):
    value = config["zhuji_live_reference"]
    return camera_info_dict(value["width"], value["height"],
                            value["distortion_model"], value["K"], value["D"],
                            value["frame_id"])


class ImagePreprocessTests(unittest.TestCase):
    def setUp(self):
        self.config = config_section(HERE / "deployment_config.json")
        self.camera = live_info(self.config)

    def test_virtual_map_matches_opencv_reference_statistics(self):
        processor = ImagePreprocessor(self.camera, self.config, "virtual_cam0")
        map_x, map_y = processor.virtual_map
        self.assertAlmostEqual(float(map_x.min()), -77.1340, places=3)
        self.assertAlmostEqual(float(map_x.max()), 728.0729, places=3)
        self.assertAlmostEqual(float(map_y.min()), -13.4122, places=3)
        self.assertAlmostEqual(float(map_y.max()), 489.4876, places=3)
        self.assertAlmostEqual(processor.black_border_ratio, 25227 / 102400, places=8)

    def test_known_virtual_pixel_projects_through_zhuji_distortion(self):
        processor = ImagePreprocessor(self.camera, self.config)
        K = np.asarray(self.config["cam0"]["virtual_K_320"], dtype=float).reshape(3, 3)
        u, v = 140, 120
        ray = np.linalg.inv(K) @ [u, v, 1.0]
        x, y = ray[0] / ray[2], ray[1] / ray[2]
        D = self.camera["D"]; r2 = x*x + y*y
        radial = 1 + D[0]*r2 + D[1]*r2*r2 + D[4]*r2*r2*r2
        xd = x*radial + 2*D[2]*x*y + D[3]*(r2 + 2*x*x)
        yd = y*radial + D[2]*(r2 + 2*y*y) + 2*D[3]*x*y
        expected = [self.camera["K"][0,0]*xd + self.camera["K"][0,2],
                    self.camera["K"][1,1]*yd + self.camera["K"][1,2]]
        np.testing.assert_allclose([processor.virtual_map[0][v, u],
                                    processor.virtual_map[1][v, u]], expected,
                                   atol=5e-5, rtol=0)

    def test_modes_return_rgb320_and_area_resize_preserves_constant(self):
        image = np.empty((480, 640, 3), np.uint8)
        image[:] = [17, 91, 203]
        for mode in ("virtual_cam0", "stretch_undistort"):
            output = ImagePreprocessor(self.camera, self.config, mode).process(image)
            self.assertEqual(output.shape, (320, 320, 3))
            self.assertEqual(output.dtype, np.uint8)
        resized = resize_area(image, 320, 320)
        np.testing.assert_array_equal(resized, np.broadcast_to([17,91,203], resized.shape))

    def test_live_intrinsic_archive_gate(self):
        archived = dict(self.camera)
        archived["K"] = self.camera["K"].copy()
        archived["K"][0, 0] = 387.36358642578125
        archived["K"][1, 1] = 386.4456787109375
        report = validate_camera_info(self.camera, archived)
        self.assertLess(report["max_relative_intrinsic"], 0.001)
        bad = dict(archived); bad["K"] = archived["K"].copy(); bad["K"][0,0] *= 1.01
        with self.assertRaisesRegex(ValueError, "drift"):
            validate_camera_info(self.camera, bad)


if __name__ == "__main__":
    unittest.main()

