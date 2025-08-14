"""Wrapper to expose speed benchmarks to unittest discovery.

Unittest discover only loads files matching the pattern test*.py and classes
starting with Test. We keep the original benchmark implementation in
speed_check.BenchmarkShoebox and subclass it here so it is auto-discovered.
"""
from tests.speed_check import BenchmarkShoebox as _BenchmarkBase

class TestBenchmarkShoebox(_BenchmarkBase):
    """Auto-discovered subclass running the benchmark tests."""
    pass
