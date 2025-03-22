import unittest
import numpy as np
from shoeboxpy.model3dof import Shoebox

class TestShoebox(unittest.TestCase):
    def setUp(self):
        # Define ship dimensions and create an instance of Shoebox.
        self.L = 10.0
        self.B = 5.0
        self.T = 3.0
        self.shoebox = Shoebox(self.L, self.B, self.T)

    def test_initial_states(self):
        # Ensure the initial states are correctly set.
        eta, nu = self.shoebox.get_states()
        self.assertEqual(eta.shape, (3,))
        self.assertEqual(nu.shape, (3,))
        np.testing.assert_array_almost_equal(eta, np.zeros(3))
        np.testing.assert_array_almost_equal(nu, np.zeros(3))

    def test_dynamics_zero_forces(self):
        # With zero forces and zero initial velocities, the state should remain unchanged.
        eta_initial, nu_initial = self.shoebox.get_states()
        dt = 0.01
        self.shoebox.step(tau=np.zeros(3), tau_ext=np.zeros(3), dt=dt)
        eta_new, nu_new = self.shoebox.get_states()
        np.testing.assert_array_almost_equal(eta_initial, eta_new, decimal=6)
        np.testing.assert_array_almost_equal(nu_initial, nu_new, decimal=6)

    def test_step_nonzero_tau(self):
        # Applying a non-zero surge force should change the states after several steps.
        tau = np.array([1.0, 0.0, 0.0])
        eta_initial, nu_initial = self.shoebox.get_states()

        dt = 0.01
        steps = 10
        for _ in range(steps):
            self.shoebox.step(tau=tau, dt=dt)
        eta_new, nu_new = self.shoebox.get_states()

        self.assertFalse(np.allclose(eta_initial, eta_new),
                         "State eta should change when forces are applied.")
        self.assertFalse(np.allclose(nu_initial, nu_new),
                         "State nu should change when forces are applied.")

if __name__ == '__main__':
    unittest.main()
