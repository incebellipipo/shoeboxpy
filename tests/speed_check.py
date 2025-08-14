"""Performance benchmark tests for shoeboxpy models (unittest version).

Runs 10,000 integration steps with dt = 0.001 (total simulated time 10 s)
for both the 3DOF and 6DOF shoebox models and reports wall-clock timing.

This is structured as a unittest.TestCase so you can run with:

	python -m unittest tests.speed_check -v

or via pytest (pytest will auto-discover unittest tests):

	python -m pytest tests/speed_check.py -s

By default these benchmarks only print results and always pass. To enforce a
maximum allowed wall time, set an environment variable:

	SHOEBOXPY_BENCH_ASSERT_SEC=0.5  (applies to each benchmark separately)

If you need per-model thresholds, you can additionally set:
	SHOEBOXPY_BENCH_ASSERT_3DOF_SEC
	SHOEBOXPY_BENCH_ASSERT_6DOF_SEC
which override the generic one for the corresponding test.
"""

from __future__ import annotations

import os
import time
import unittest
import numpy as np

from shoeboxpy.model3dof import Shoebox as Shoebox3DOF
from shoeboxpy.model6dof import Shoebox as Shoebox6DOF


def _get_threshold(env_key: str) -> float | None:
	"""Return float threshold from environment variable if set and valid."""
	val = os.getenv(env_key)
	if not val:
		return None
	try:
		return float(val)
	except ValueError:
		return None


def _resolve_threshold(label: str) -> float | None:
	specific_key = f"SHOEBOXPY_BENCH_ASSERT_{label.upper()}_SEC"
	specific = _get_threshold(specific_key)
	if specific is not None:
		return specific
	return _get_threshold("SHOEBOXPY_BENCH_ASSERT_SEC")


def _print_result(label: str, steps: int, dt: float, wall: float):
	steps_per_sec = steps / wall if wall > 0 else float("inf")
	ns_per_step = wall / steps * 1e9
	sim_time = steps * dt
	print(
		f"[shoeboxpy benchmark] {label}: steps={steps:,} dt={dt} sim_time={sim_time:.3f}s -> "
		f"wall={wall:.4f}s, steps/s={steps_per_sec:,.0f}, ns/step={ns_per_step:,.0f}"
	)


class BenchmarkShoebox(unittest.TestCase):
	STEPS = 10_000
	DT = 0.001

	def benchmark_core(self, label: str, boat, tau: np.ndarray):
		start = time.perf_counter()
		for _ in range(self.STEPS):
			boat.step(tau=tau, dt=self.DT)
		wall = time.perf_counter() - start
		_print_result(label, self.STEPS, self.DT, wall)
		threshold = _resolve_threshold(label)
		if threshold is not None:
			self.assertLessEqual(
				wall,
				threshold,
				msg=f"{label} benchmark {wall:.4f}s exceeded threshold {threshold:.4f}s",
			)

	def test_benchmark_3dof(self):
		boat = Shoebox3DOF(L=10.0, B=5.0, T=3.0)
		tau = np.array([1.0, 0.2, 0.05])
		self.benchmark_core("3DOF", boat, tau)

	def test_benchmark_6dof(self):
		boat = Shoebox6DOF(L=2.0, B=1.0, T=0.5)
		tau = np.array([1.0, 0.2, 0.0, 0.01, 0.0, 0.02])
		self.benchmark_core("6DOF", boat, tau)


if __name__ == "__main__":  # manual run convenience
	unittest.main(verbosity=2)
