import unittest

from relay_mac import relay_commands


class RelayCommandTests(unittest.TestCase):
    def test_single_route_forwards_both_directions(self):
        local, reverse = relay_commands([("10.0.0.5", 18765, 28765)])
        self.assertEqual(local[-1], "b300")
        self.assertEqual(reverse[-1], "zhuji")
        self.assertIn("127.0.0.1:28765:10.0.0.5:18765", local)
        self.assertIn("127.0.0.1:28765:127.0.0.1:28765", reverse)
        self.assertIn("ExitOnForwardFailure=yes", local)

    def test_four_checkpoints_share_one_ssh_pair(self):
        routes = [("10.0.0.5", 18765 + i, 28765 + i) for i in range(4)]
        local, reverse = relay_commands(routes)
        self.assertEqual(local.count("-L"), 4)
        self.assertEqual(reverse.count("-R"), 4)
        self.assertEqual(local.count("ssh"), 1)
        for i in range(4):
            self.assertIn("127.0.0.1:%d:10.0.0.5:%d" % (28765 + i, 18765 + i), local)

    def test_rejects_duplicate_or_invalid_ports(self):
        with self.assertRaises(ValueError):
            relay_commands([("h", 18765, 28765), ("h", 18766, 28765)])
        with self.assertRaises(ValueError):
            relay_commands([("h", 0, 28765)])
        with self.assertRaises(ValueError):
            relay_commands([])


if __name__ == "__main__":
    unittest.main()
