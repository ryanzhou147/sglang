"""Unit tests for Gamma-distributed (bursty) request arrivals in bench_serving."""

import asyncio
import time
import unittest

import numpy as np

from sglang.benchmark.serving import get_request, sample_arrival_interval
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_SAMPLES = 200_000


def _intervals(request_rate: float, burstiness: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.array(
        [
            sample_arrival_interval(request_rate, burstiness, rng)
            for _ in range(NUM_SAMPLES)
        ]
    )


class TestSampleArrivalInterval(CustomTestCase):
    def test_burstiness_one_is_poisson(self):
        intervals = _intervals(request_rate=4.0, burstiness=1.0)
        self.assertAlmostEqual(intervals.mean(), 0.25, delta=0.005)
        # Exponential inter-arrivals have coefficient of variation 1.
        self.assertAlmostEqual(intervals.std() / intervals.mean(), 1.0, delta=0.02)

    def test_mean_rate_is_preserved_for_any_burstiness(self):
        for burstiness in (0.0625, 0.25, 4.0):
            with self.subTest(burstiness=burstiness):
                intervals = _intervals(request_rate=2.0, burstiness=burstiness)
                self.assertAlmostEqual(intervals.mean(), 0.5, delta=0.02)

    def test_coefficient_of_variation_is_inverse_sqrt_burstiness(self):
        # CV 4 (FastServe's bursty setting) corresponds to burstiness 1/16.
        for burstiness, expected_cv in ((0.0625, 4.0), (0.25, 2.0), (4.0, 0.5)):
            with self.subTest(burstiness=burstiness):
                intervals = _intervals(request_rate=1.0, burstiness=burstiness)
                cv = intervals.std() / intervals.mean()
                self.assertAlmostEqual(cv, expected_cv, delta=expected_cv * 0.05)

    def test_rejects_non_positive_burstiness(self):
        rng = np.random.default_rng(0)
        for burstiness in (0.0, -1.0):
            with self.subTest(burstiness=burstiness):
                with self.assertRaises(ValueError):
                    sample_arrival_interval(1.0, burstiness, rng)


class TestGetRequestBurstiness(CustomTestCase):
    def _drain(self, **kwargs):
        async def collect():
            return [r async for r in get_request(**kwargs)]

        return asyncio.run(collect())

    def test_infinite_rate_yields_everything_without_waiting(self):
        start = time.perf_counter()
        out = self._drain(
            input_requests=list(range(50)), request_rate=float("inf"), burstiness=0.1
        )
        self.assertEqual(out, list(range(50)))
        self.assertLess(time.perf_counter() - start, 0.5)


if __name__ == "__main__":
    unittest.main()
