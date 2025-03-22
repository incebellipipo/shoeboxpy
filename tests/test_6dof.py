import unittest
import numpy as np
from shoeboxpy.model6dof import Shoebox


class TestShoeboxModel6DOF(unittest.TestCase):
    def setUp(self):
        self.L = 2.0
        self.B = 1.0
        self.T = 1.0
        self.rho = 1000.0

    def test_initial_states(self):
        # Verify that initial eta and nu are zeros.
        shoebox = Shoebox(L=self.L, B=self.B, T=self.T)
        eta, nu = shoebox.get_states()
        np.testing.assert_array_almost_equal(eta, np.zeros(6))
        np.testing.assert_array_almost_equal(nu, np.zeros(6))

    def test_mass_and_inertia(self):
        # Verify that the computed mass and inertia match expected values.
        shoebox = Shoebox(L=self.L, B=self.B, T=self.T, rho=self.rho)
        expected_mass = self.rho * self.L * self.B * self.T
        self.assertTrue(np.isclose(shoebox.m, expected_mass))

        # Uniform rectangular box moments of inertia.
        Ix = (1 / 12) * expected_mass * (self.B**2 + self.T**2)
        Iy = (1 / 12) * expected_mass * (self.L**2 + self.T**2)
        Iz = (1 / 12) * expected_mass * (self.L**2 + self.B**2)
        np.testing.assert_allclose(shoebox.MRB[3, 3], Ix)
        np.testing.assert_allclose(shoebox.MRB[4, 4], Iy)
        np.testing.assert_allclose(shoebox.MRB[5, 5], Iz)

    def test_jacobian_transformation(self):
        # For zero Euler angles, the transformation matrix J should return identity blocks.
        shoebox = Shoebox(L=self.L, B=self.B, T=self.T)
        eta = np.zeros(6)  # [x, y, z, phi, theta, psi] with zero angles.
        J = shoebox.J(eta)
        np.testing.assert_array_almost_equal(J[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(J[3:, 3:], np.eye(3))

    def test_dynamics_no_forces(self):
        # With zero control and external forces, the state should remain unchanged.
        shoebox = Shoebox(L=self.L, B=self.B, T=self.T)
        eta0, nu0 = shoebox.get_states()
        shoebox.step(tau=np.zeros(6), tau_ext=np.zeros(6), dt=0.01)
        eta1, nu1 = shoebox.get_states()
        np.testing.assert_array_almost_equal(eta0, eta1)
        np.testing.assert_array_almost_equal(nu0, nu1)

    def test_step_function_with_restoring(self):
        # Testing that nonzero restoring parameters change the boat's angular state.
        GM_phi = 0.1
        GM_theta = 0.1
        shoebox = Shoebox(
            L=self.L, B=self.B, T=self.T, GM_phi=GM_phi, GM_theta=GM_theta
        )
        # Set a small roll and pitch disturbance
        shoebox.eta[3] = 0.1  # Roll
        shoebox.eta[4] = -0.05  # Pitch
        initial_eta, initial_nu = shoebox.get_states()
        shoebox.step(dt=0.01)
        new_eta, new_nu = shoebox.get_states()

        # Check that roll and pitch angles have updated and the velocity state has changed.
        self.assertFalse(np.isclose(new_eta[3], initial_eta[3]))
        self.assertFalse(np.isclose(new_eta[4], initial_eta[4]))
        self.assertFalse(np.allclose(new_nu, initial_nu))


if __name__ == "__main__":
    unittest.main()
