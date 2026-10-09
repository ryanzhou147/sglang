"""Unit tests for timestamped JSONL workloads (trace replay with real prompts)."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.benchmark import serving
from sglang.benchmark.datasets.timed_jsonl import TimedJsonlDataset
from sglang.benchmark.serving import make_request_generator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _write_jsonl(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _load(rows, num_requests=100):
    path = _write_jsonl(rows)
    try:
        return TimedJsonlDataset(dataset_path=path, num_requests=num_requests).load(
            tokenizer=_WordTokenizer()
        )
    finally:
        os.remove(path)


class TestTimedJsonlDataset(CustomTestCase):
    def test_row_fields_reach_the_request(self):
        rows = _load(
            [
                {
                    "prompt": "one two three",
                    "output_len": 7,
                    "timestamp": 250.0,
                    "rid": "math-0001",
                    "routing_key": "short",
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.prompt, "one two three")
        self.assertEqual(row.output_len, 7)
        self.assertEqual(row.timestamp, 250.0)
        self.assertEqual(row.routing_key, "short")
        # The rid tags each server-side record, so client and server results join on it.
        self.assertEqual(row.extra_request_body["rid"], "math-0001")
        # prompt_len is computed when the file does not carry it.
        self.assertEqual(row.prompt_len, 3)

    def test_given_prompt_len_is_kept(self):
        rows = _load(
            [{"prompt": "a b", "prompt_len": 40, "output_len": 1, "timestamp": 0}]
        )
        self.assertEqual(rows[0].prompt_len, 40)

    def test_truncation_keeps_the_earliest_arrivals(self):
        rows = _load(
            [
                {"prompt": "c", "output_len": 1, "timestamp": 300},
                {"prompt": "a", "output_len": 1, "timestamp": 100},
                {"prompt": "b", "output_len": 1, "timestamp": 200},
            ],
            num_requests=2,
        )
        self.assertEqual([r.prompt for r in rows], ["a", "b"])

    def test_missing_required_field_names_the_line(self):
        for missing in ("prompt", "output_len", "timestamp"):
            with self.subTest(missing=missing):
                row = {"prompt": "x", "output_len": 1, "timestamp": 0}
                del row[missing]
                with self.assertRaisesRegex(ValueError, f"line 2.*{missing}"):
                    _load([{"prompt": "ok", "output_len": 1, "timestamp": 0}, row])


class TestTraceReplay(CustomTestCase):
    def test_trace_timestamps_set_the_send_times(self):
        """With --use-trace-timestamps the client must send at the trace times,
        not all at once or on a Poisson clock."""
        rows = _load(
            [
                {"prompt": "a", "output_len": 1, "timestamp": 0},
                {"prompt": "b", "output_len": 1, "timestamp": 300},
                {"prompt": "c", "output_len": 1, "timestamp": 600},
            ]
        )

        clock = [0.0]
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            clock[0] += seconds
            await real_sleep(0)

        async def send_offsets():
            generator = make_request_generator(
                input_requests=rows,
                request_rate=float("inf"),
                use_trace_timestamps=True,
                burstiness=1.0,
            )
            return [clock[0] async for _ in generator]

        # A fake clock keeps the check exact on a loaded CI machine.
        with (
            mock.patch.object(
                serving.time, "perf_counter", side_effect=lambda: clock[0]
            ),
            mock.patch.object(serving.asyncio, "sleep", fake_sleep),
        ):
            offsets = asyncio.run(send_offsets())
        self.assertEqual(offsets, [0.0, 0.3, 0.6])


if __name__ == "__main__":
    unittest.main()
