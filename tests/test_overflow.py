import unittest
import warnings
import numpy as np

from shoeboxpy.model3dof import Shoebox as Shoebox3DOF
from shoeboxpy.model6dof import Shoebox as Shoebox6DOF


class TestOverflow3DOF(unittest.TestCase):
    def setUp(self):
        self.model = Shoebox3DOF(L=10.0, B=5.0, T=3.0)

    def test_overflow_trigger(self):
        """Applying forces near float max should produce non-finite state (overflow)."""
        big = np.finfo(float).max / 10.0  # still enormous
        tau = np.array([big, big, big])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(100):
                self.model.step(tau=tau, dt=0.01)
        eta, nu = self.model.get_states()
        # Either we saw an overflow warning or states turned non-finite

        # report velocity after run
        print(f"Final velocity: {nu} "
              f"Final position: {eta}")

        saw_overflow_warning = any("overflow" in str(
            item.message).lower() for item in w)
        self.assertTrue(
            saw_overflow_warning or (not np.isfinite(
                eta).all()) or (not np.isfinite(nu).all()),
            "Expected overflow warning or non-finite states with huge input forces.",
        )

    def test_no_overflow_with_large_but_safe_force(self):
        """Moderately large forces should remain finite for a short integration."""
        tau = np.array([1e6, -1e6, 5e5])
        for _ in range(50):
            self.model.step(tau=tau, dt=1e-4)
        eta, nu = self.model.get_states()

        print(f"Final velocity: {nu} "
              f"Final position: {eta}")

        self.assertTrue(np.isfinite(eta).all())
        self.assertTrue(np.isfinite(nu).all())


class TestOverflow6DOF(unittest.TestCase):
    def setUp(self):
        self.model = Shoebox6DOF(L=2.0, B=1.0, T=0.5)

    def test_overflow_trigger(self):
        big = np.finfo(float).max / 10.0
        tau = np.array([big, big, 0.0, 0.0, 0.0, big])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(100):
                self.model.step(tau=tau, dt=0.01)

        eta, nu = self.model.get_states()
        saw_overflow_warning = any("overflow" in str(
            item.message).lower() for item in w)
        self.assertTrue(
            saw_overflow_warning or (not np.isfinite(
                eta).all()) or (not np.isfinite(nu).all()),
            "Expected overflow warning or non-finite states with huge input forces (6DOF).",
        )

    def test_no_overflow_with_large_but_safe_force(self):
        tau = np.array([1e5, -2e5, 1e5, 5e4, -5e4, 2e4])
        for _ in range(50):
            self.model.step(tau=tau, dt=1e-4)
        eta, nu = self.model.get_states()
        self.assertTrue(np.isfinite(eta).all())
        self.assertTrue(np.isfinite(nu).all())


if __name__ == "__main__":  # manual run
    unittest.main(verbosity=2)
