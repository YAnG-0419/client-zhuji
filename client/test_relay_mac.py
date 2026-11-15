import unittest

from relay_mac import relay_commands


class RelayCommandTests(unittest.TestCase):
    def test_both_forwards_bind_loopback_and_have_no_gateway_flag(self):
        local, reverse = relay_commands("10.0.0.142")
        self.assertIn("127.0.0.1:28765:10.0.0.142:18765", local)
        self.assertIn("127.0.0.1:28765:127.0.0.1:28765", reverse)
        self.assertEqual(local[-1], "b300")
        self.assertEqual(reverse[-1], "zhuji")
        self.assertNotIn("-g", local + reverse)


if __name__ == "__main__":
    unittest.main()

