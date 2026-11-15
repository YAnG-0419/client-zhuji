import socket
import threading
import unittest

import numpy as np

from xflow_rpc import (PROTOCOL, XFlowClient, XFlowProtocolError,
                       XFlowRemoteError, recv_message, send_message)


TIMES = np.arange(1, 33, dtype=np.float32) / 120.0


def info_response(deployment="deploy-test"):
    return {
        "protocol": PROTOCOL, "protocol_version": 4, "wire": "wire_v4",
        "default_model": "hh_a12", "horizon": 32,
        "models": {"hh_a12": {"deployment_id": deployment,
                                 "contract": "humanhand",
                                 "history_offset_frames": 4,
                                 "keypoint_time_s": TIMES.tolist()}},
        "tasks": {"flower": {"instruction": "test"}},
    }


def infer_response(request, deployment="deploy-test"):
    points = request["observation"]["keypoints"].copy()
    valid = request["observation"].get("keypoint_valid")
    if valid is not None:
        points[valid == 0] = 0
    return {
        "protocol": PROTOCOL, "protocol_version": 4,
        "request_id": request["request_id"], "session_id": request["session_id"],
        "sample_ns": request["sample_ns"], "model": "hh_a12",
        "deployment_id": deployment, "contract": "humanhand",
        "task": request["task"], "horizon": 32,
        "hand_order": ["left", "right"], "frame": "cam0_color_optical_frame",
        "units": "metres", "keypoints": np.zeros((32, 2, 21, 3), np.float32),
        "keypoint_time_s": TIMES.copy(), "proprio_keypoints": points,
        "timings_ms": {"total": 1.0}, "inference_seconds": 0.001,
    }


class ScriptedServer:
    def __init__(self, steps):
        self.client, self.server = socket.socketpair()
        self.steps = steps
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            for expected, response in self.steps:
                request = recv_message(self.server)
                if request["cmd"] != expected:
                    raise AssertionError((request["cmd"], expected))
                answer = response(request) if callable(response) else response
                send_message(self.server, answer)
        except BaseException as error:
            self.error = error
        finally:
            self.server.close()

    def finish(self):
        self.thread.join(2)
        if self.thread.is_alive():
            raise AssertionError("fake server did not finish")
        if self.error:
            raise self.error


def sample(sample_ns=100):
    return dict(task="flower", head_rgb=np.zeros((320, 320, 3), np.uint8),
                history_rgb=np.zeros((320, 320, 3), np.uint8),
                keypoints=np.zeros((2, 21, 3), np.float32), sample_ns=sample_ns,
                image_space="rgb320")


class XFlowRpcTests(unittest.TestCase):
    def test_strict_info_reset_and_rgb320_infer(self):
        server = ScriptedServer([
            ("info", info_response()),
            ("reset", lambda request: {"request_id": request["request_id"],
                                         "session_id": "session-1"}),
            ("infer", infer_response),
        ])
        client = XFlowClient(sock=server.client, deployment_id="deploy-test")
        client.info(); client.reset()
        answer = client.infer(**sample())
        self.assertEqual(answer["keypoints"].shape, (32, 2, 21, 3))
        self.assertEqual(client.infer_count, 1)
        client.close(); server.finish()

    def test_shadow_request_may_omit_history_image(self):
        def response(request):
            self.assertNotIn("history_rgb", request["observation"])
            return infer_response(request)
        server = ScriptedServer([
            ("info", info_response()),
            ("reset", lambda request: {"request_id": request["request_id"],
                                         "session_id": "session-shadow"}),
            ("infer", response),
        ])
        client = XFlowClient(sock=server.client)
        client.info(); client.reset()
        values = sample(); values["history_rgb"] = None
        client.infer(**values)
        client.close(); server.finish()

    def test_deployment_pin_rejects_different_model(self):
        server = ScriptedServer([("info", info_response("changed"))])
        client = XFlowClient(sock=server.client, deployment_id="expected")
        with self.assertRaisesRegex(XFlowProtocolError, "deployment_id"):
            client.info()
        client.close(); server.finish()

    def test_stale_session_is_not_reused_with_old_sample(self):
        def stale(request):
            return {"error": {"code": "STALE_SESSION", "message": "restart"},
                    "request_id": request["request_id"]}
        server = ScriptedServer([
            ("info", info_response()),
            ("reset", lambda request: {"request_id": request["request_id"],
                                         "session_id": "old"}),
            ("infer", stale),
        ])
        client = XFlowClient(sock=server.client)
        client.info(); client.reset()
        with self.assertRaises(XFlowRemoteError):
            client.infer(**sample())
        self.assertIsNone(client.session_id)
        self.assertEqual(client.last_attempted_sample_ns, 100)
        client.close(); server.finish()

    def test_inference_budget_and_shapes_fail_before_send(self):
        server = ScriptedServer([("info", info_response())])
        client = XFlowClient(sock=server.client, infer_budget=0)
        client.info(); client.session_id = "test"
        with self.assertRaisesRegex(Exception, "budget"):
            client.infer(**sample())
        broken = sample(); broken["head_rgb"] = np.zeros((400, 640, 3), np.uint8)
        client.infer_budget = 1
        with self.assertRaisesRegex(ValueError, "head_rgb"):
            client.infer(**broken)
        client.close(); server.finish()


if __name__ == "__main__":
    unittest.main()
