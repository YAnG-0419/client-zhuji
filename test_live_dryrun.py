from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from live_dryrun import collect_snapshot, ros_environment, ros_snapshot_command


class LiveDryrunTests(unittest.TestCase):
    def test_ros_helper_is_subscription_snapshot_only(self):
        command = ros_snapshot_command(Path("/client"), Path("/tmp/sample.npz"))
        joined = " ".join(command).lower()
        for forbidden in ("execute", "servop", "servoj", "movej", "moveh",
                          "emgy", "publisher"):
            self.assertNotIn(forbidden, joined)
        self.assertEqual(command[0], "/usr/bin/python3")
        source = (Path(__file__).with_name("ros_readonly_observer.py").read_text())
        self.assertNotIn("create_publisher", source)
        self.assertNotIn("request_emgy_stop", source)

    def test_missing_topics_fail_before_inference(self):
        calls = []
        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=2,
                                   stdout='{"missing_topics":["/wuji/left/joint_states"]}',
                                   stderr="")
        with self.assertRaisesRegex(RuntimeError, "wuji"):
            collect_snapshot(Path("/client"), Path("/tmp/not-created.npz"), runner)
        self.assertEqual(len(calls), 1)

    def test_dds_environment_is_explicit(self):
        environment = ros_environment({})
        self.assertEqual(environment["ROS_DOMAIN_ID"], "0")
        self.assertEqual(environment["ROS_LOCALHOST_ONLY"], "0")
        self.assertIn("udp_only", environment["FASTRTPS_DEFAULT_PROFILES_FILE"])


if __name__ == "__main__":
    unittest.main()
