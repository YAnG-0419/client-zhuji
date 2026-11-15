import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from wire_trace import WireTrace, attach_trace, save_trace


class TraceTests(unittest.TestCase):
    def test_exact_payload_and_return_are_preserved(self):
        sent = []
        socket = SimpleNamespace(send=lambda x: sent.append(x) or 7,
                                 recv=lambda: '{"q":[1],"dq":[9.41]}', value=123)
        trace = WireTrace(socket)
        self.assertEqual(trace.send('{"q": [0.1], "filter_ratio": 1.0}'), 7)
        self.assertEqual(trace.recv(), '{"q":[1],"dq":[9.41]}')
        self.assertEqual(trace.rows[0]['payload'], sent[0])
        self.assertEqual(trace.value, 123)

    def test_send_failure_preserves_exception_and_raw_request(self):
        failure = RuntimeError('connection lost')
        def fail(x):
            raise failure
        trace = WireTrace(SimpleNamespace(send=fail))
        with self.assertRaises(RuntimeError) as raised:
            trace.send('packet')
        self.assertIs(raised.exception, failure)
        self.assertEqual(trace.rows[-1]['payload'], 'packet')
        self.assertIn('connection lost', trace.rows[-1]['error'])

    def test_tail_is_bounded_and_saved_afterward(self):
        trace = WireTrace(SimpleNamespace(send=lambda x: None), capacity=2)
        for i in range(3):
            trace.send(str(i))
        with TemporaryDirectory() as folder:
            r = trace.save(Path(folder)/'wire.jsonl')
            rows = [json.loads(s) for s in (Path(folder)/'wire.jsonl').read_text().splitlines()]
        self.assertEqual(r['dropped'], 1)
        self.assertEqual([r['payload'] for r in rows], ['1', '2'])

    def test_attach_is_passive(self):
        calls = []
        r = SimpleNamespace(ws=SimpleNamespace(send=lambda x: calls.append(x)))
        attach_trace(r)
        self.assertFalse(calls)
        with TemporaryDirectory() as folder:
            result = {}
            save_trace(r, Path(folder), result)
        self.assertEqual(result['wire_trace']['events'], 0)


if __name__ == '__main__':
    unittest.main()
