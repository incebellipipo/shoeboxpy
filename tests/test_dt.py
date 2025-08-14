import unittest
import numpy as np

from shoeboxpy.model3dof import Shoebox as Shoebox3DOF
from shoeboxpy.model6dof import Shoebox as Shoebox6DOF


def run_sim(model_cls, dt: float, T: float, tau, tau_ext=None, model_kwargs=None):
	"""Run a simulation for total time T with step dt and constant inputs.

	Returns final (eta, nu).
	"""
	if model_kwargs is None:
		model_kwargs = {}
	model = model_cls(**model_kwargs)
	steps = int(round(T / dt))
	for _ in range(steps):
		model.step(tau=tau, tau_ext=tau_ext, dt=dt)
	return model.get_states()


def l2_state_error(ref_eta, ref_nu, eta, nu):
	return np.linalg.norm(ref_eta - eta) + np.linalg.norm(ref_nu - nu)


class TestTimeStepConvergence(unittest.TestCase):
	"""Convergence tests: smaller dt should reduce integration error (RK4 ~ O(dt^4))."""

	def _check_convergence(self, model_cls, tau, size, dof_label):
		# Simulation parameters (keep fast):
		T_total = 0.5  # seconds total
		# dt values halving: expect ~16x error reduction per halving (allow wide bounds)
		dts = [0.02, 0.01, 0.005, 0.0025]

		# Reference = smallest dt
		ref_eta, ref_nu = run_sim(model_cls, dts[-1], T_total, tau, model_kwargs=size)

		errors = []
		for dt in dts[:-1]:  # skip smallest (reference) in error list
			eta, nu = run_sim(model_cls, dt, T_total, tau, model_kwargs=size)
			err = l2_state_error(ref_eta, ref_nu, eta, nu)
			errors.append(err)

		# Basic monotonic decrease check (allow tiny numerical fluctuations)
		for earlier, later in zip(errors, errors[1:]):
			self.assertLess(
				later, earlier * 0.9 + 1e-12,
				msg=f"Error did not decrease sufficiently for {dof_label}: {errors}",
			)

		# Estimate observed order p using last two error levels: e ~ C dt^p
		# p = log(e_i/e_{i+1}) / log(2)
		observed_orders = []
		for e_i, e_ip1 in zip(errors, errors[1:]):
			if e_ip1 > 0 and e_i > e_ip1:
				observed_orders.append(np.log(e_i / e_ip1) / np.log(2))
		if observed_orders:
			# RK4 ideal ~4; accept > 2.5 (loose because dynamics nonlinear & short horizon)
			self.assertGreater(
				np.median(observed_orders),
				2.5,
				msg=f"Observed order too low for {dof_label}: {observed_orders}",
			)

	def test_convergence_3dof(self):
		tau = np.array([1.0, 0.2, 0.05])
		size = dict(L=10.0, B=5.0, T=3.0)
		self._check_convergence(Shoebox3DOF, tau, size, "3DOF")

	def test_convergence_6dof(self):
		tau = np.array([1.0, 0.2, 0.0, 0.05, 0.0, 0.02])
		size = dict(L=2.0, B=1.0, T=0.5)
		self._check_convergence(Shoebox6DOF, tau, size, "6DOF")


if __name__ == "__main__":  # manual run
	unittest.main(verbosity=2)
