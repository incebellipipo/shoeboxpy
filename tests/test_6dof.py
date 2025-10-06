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

    def test_to3dof_state_projection_and_setters(self):
        """Verify that to3dof.get_states() matches get_states(dof3=True)
        and that setting eta/nu through the adapter updates the parent."""
        shoebox = Shoebox(L=self.L, B=self.B, T=self.T)
        eta3_a, nu3_a = shoebox.get_states(dof3=True)
        eta3_b, nu3_b = shoebox.to3dof.get_states()
        np.testing.assert_allclose(eta3_a, eta3_b)
        np.testing.assert_allclose(nu3_a, nu3_b)

        # Set via adapter and verify parent updated
        shoebox.to3dof.eta = np.array([1.0, 2.0, 0.5])
        shoebox.to3dof.nu = np.array([0.1, -0.2, 0.05])
        eta_parent, nu_parent = shoebox.get_states()
        np.testing.assert_allclose(eta_parent[[0, 1, 5]], np.array([1.0, 2.0, 0.5]))
        np.testing.assert_allclose(nu_parent[[0, 1, 5]], np.array([0.1, -0.2, 0.05]))

    def test_to3dof_matrix_subblocks(self):
        """Verify the 3x3 matrices from the adapter equal the corresponding
        subblocks from the 6x6 matrices."""
        shoebox = Shoebox(L=self.L, B=self.B, T=self.T)
        idx = [0, 1, 5]
        np.testing.assert_allclose(shoebox.to3dof.MRB, shoebox.MRB[np.ix_(idx, idx)])
        np.testing.assert_allclose(shoebox.to3dof.MA, shoebox.MA[np.ix_(idx, idx)])
        np.testing.assert_allclose(shoebox.to3dof.M_eff, shoebox.M_eff[np.ix_(idx, idx)])
        np.testing.assert_allclose(shoebox.to3dof.D, shoebox.D[np.ix_(idx, idx)])

    def test_to3dof_step_equivalence(self):
        """Check that stepping the parent with a 3-element tau produces the
        same 3-DOF states as stepping via the adapter."""
        shoebox_a = Shoebox(L=self.L, B=self.B, T=self.T)
        shoebox_b = Shoebox(L=self.L, B=self.B, T=self.T)

        tau3 = np.array([1.0, 0.5, 0.2])
        dt = 0.01

        # Step parent with 3-element tau (parent handles expansion)
        shoebox_a.step(tau=tau3, dt=dt)

        # Step via adapter
        shoebox_b.to3dof.step(tau=tau3, dt=dt)

        eta_a3, nu_a3 = shoebox_a.get_states(dof3=True)
        eta_b3, nu_b3 = shoebox_b.get_states(dof3=True)
        np.testing.assert_allclose(eta_a3, eta_b3)
        np.testing.assert_allclose(nu_a3, nu_b3)


if __name__ == "__main__":
    unittest.main()
