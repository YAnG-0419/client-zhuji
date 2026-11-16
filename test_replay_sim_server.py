import os
import socket
import threading
import unittest
from pathlib import Path

import numpy as np

from replay_sim_server import Handler, ReplayServer, ReplayService
from xflow_rpc import XFlowClient, XFlowRemoteError


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PENGYANG_ROOT", HERE.parent)).resolve()
EPISODE = ROOT / "replay_robot/episode2_wuji2_a5da567e_20261116/replay_input_50hz.npz"


class ReplaySimServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ReplayService(ROOT, ROOT / "client/deployment_config.json", EPISODE)
        cls.server = ReplayServer(("127.0.0.1", 0), Handler)
        cls.server.service = cls.service
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def client(self):
        return XFlowClient("127.0.0.1", self.port, model=None, timeout_s=5)

    def infer(self, client, sample_ns, keypoints=None):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        return client.infer(
            task="tidying", head_rgb=image, history_rgb=image,
            keypoints=(self.service.keypoints[0].copy() if keypoints is None else keypoints),
            sample_ns=sample_ns,
            image_space="rgb320", speed_ratio=2.0)

    def test_protocol_and_cursor_then_final_hold(self):
        with self.client() as client:
            info = client.info()
            self.assertEqual(info["default_model"], "replay_episode2")
            client.reset()
            first = self.infer(client, 1)
            self.assertEqual(first["keypoints"].shape, (32, 2, 21, 3))
            self.assertAlmostEqual(float(first["keypoint_time_s"][-1]), 32 / 60, places=6)
            self.assertGreater(first["replay_cursor_end_s"], first["replay_cursor_start_s"])
            second = self.infer(client, 2)
            self.assertGreater(second["replay_cursor_start_s"], first["replay_cursor_start_s"])
            self.assertFalse(np.array_equal(first["keypoints"], second["keypoints"]))
            last = second
            for sample_ns in range(3, 24):
                last = self.infer(client, sample_ns)
            self.assertTrue(last["replay_complete"])
            self.assertTrue(np.array_equal(last["keypoints"],
                                           np.broadcast_to(last["keypoints"][-1], last["keypoints"].shape)))

    def test_each_chunk_is_rooted_at_request_proprio(self):
        shifted = self.service.keypoints[0].copy()
        shifted[0] += np.asarray([0.12, -0.03, 0.04], np.float32)
        shifted[1] += np.asarray([-0.09, 0.02, -0.01], np.float32)
        with self.client() as client:
            client.info(); client.reset()
            response = self.infer(client, 20, shifted)
        first_step = response["keypoints"][0]
        root_step = np.linalg.norm(first_step[:, 0] - shifted[:, 0], axis=-1)
        self.assertLess(float(root_step.max()), 0.02)
        self.assertEqual(len(response["replay_request_anchor"]), 2)

    def test_reset_restarts_episode_and_unknown_task_is_rejected(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        with self.client() as client:
            client.info(); client.reset()
            original = self.infer(client, 10)["keypoints"]
            client.reset()
            restarted = self.infer(client, 11)["keypoints"]
            self.assertTrue(np.array_equal(original, restarted))
            with self.assertRaises((ValueError, XFlowRemoteError)):
                client.infer(task="not-a-task", head_rgb=image, history_rgb=image,
                             keypoints=self.service.keypoints[0].copy(), sample_ns=12,
                             image_space="rgb320", speed_ratio=2.0)


if __name__ == "__main__":
    unittest.main()
