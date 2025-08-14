import numpy as np
import numpy.typing as npt
import typing as tp


class Shoebox:
    r"""
    3-DOF planar ship model (surge, sway, yaw).

    States:

    .. math::
        \eta &= [x, y, \psi] \\
        \nu  &= [u, v, r]

    Dynamics:

    .. math::
        & \dot{\eta} = J(\psi)\nu \\
        & (M_{RB}+M_A)\dot{\nu} + \Big(C_{RB}(\nu)+C_A(\nu)\Big)\nu + D\nu = \tau

    where :math:`\tau` is the control/input force/moment vector in surge, sway, and yaw.

    :param L: Length of the vessel.
    :param B: Breadth of the vessel.
    :param T: Draft of the vessel.
    :param rho: Density of water (default is 1000.0).
    :param alpha_u: Added mass coefficient in surge (default is 0.1).
    :param alpha_v: Added mass coefficient in sway (default is 0.2).
    :param alpha_r: Added mass coefficient in yaw (default is 0.1).
    :param beta_u: Damping factor in surge (default is 0.05).
    :param beta_v: Damping factor in sway (default is 0.05).
    :param beta_r: Damping factor in yaw (default is 0.05).
    :param eta0: Initial state vector for position and orientation, i.e. [x, y, \psi]. Default is a zero array of length 3.
    :param nu0: Initial state vector for velocities, i.e. [u, v, r]. Default is a zero array of length 3.
    """

    def __init__(
        self,
        L: float,
        B: float,
        T: float,
        rho: float = 1000.0,
        # Added mass coefficients for 3DOF
        alpha_u: float = 0.1,
        alpha_v: float = 0.2,
        alpha_r: float = 0.1,
        # Damping factors for 3DOF
        beta_u: float = 0.05,
        beta_v: float = 0.05,
        beta_r: float = 0.05,
        # Initial states: [x, y, psi] and [u, v, r]
        eta0: npt.NDArray[np.float64] = np.zeros(3),
        nu0: npt.NDArray[np.float64] = np.zeros(3),
    ):
        # Rigid-body mass from volume
        self.m = rho * L * B * T

        # Yaw moment of inertia for a rectangular (shoebox) vessel
        Izz = (1.0 / 12.0) * self.m * (L**2 + B**2)

        # Rigid-body mass matrix (3x3)
        self.MRB = np.diag([self.m, self.m, Izz])

        # Added mass (diagonal)
        self.MA = np.diag(
            [alpha_u * self.m, alpha_v * self.m, alpha_r * Izz]  # surge  # sway  # yaw
        )

        self.M_eff = self.MRB + self.MA

        # Damping matrix (diagonal)
        self.D = np.diag([beta_u * self.m, beta_v * self.m, beta_r * Izz])

        self.invM_eff = np.linalg.inv(self.M_eff)

        # Store states
        self.eta = eta0.astype(float)  # [x, y, psi]
        self.nu = nu0.astype(float)  # [u, v, r]
        # Initialize reusable buffers
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Allocate reusable numpy arrays to reduce per-step allocations."""
        self._k_eta = np.zeros((4, 3), dtype=float)
        self._k_nu = np.zeros((4, 3), dtype=float)
        self._eta_tmp = np.zeros(3, dtype=float)
        self._nu_tmp = np.zeros(3, dtype=float)
        self._eta_dot_buf = np.zeros(3, dtype=float)
        self._nu_dot_buf = np.zeros(3, dtype=float)
        self._zeros3 = np.zeros(3, dtype=float)

    def J(self, eta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute the 3x3 transformation matrix from body velocities to inertial frame.
        """
        psi = eta[2]
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def C_RB(self, nu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Rigid-body Coriolis/centripetal matrix for 3DOF.
        """
        u, v, r = nu
        m = self.m
        Izz = self.MRB[2, 2]
        # Standard form for surge-sway-yaw
        return np.array([[0, 0, -m * v], [0, 0, m * u], [m * v, -m * u, 0]])

    def C_A(self, nu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Added mass Coriolis/centripetal matrix for 3DOF.
        """
        u, v, r = nu
        Xu_dot = self.MA[0, 0]
        Yv_dot = self.MA[1, 1]
        Nr_dot = self.MA[2, 2]
        return np.array(
            [[0, 0, -Yv_dot * v], [0, 0, Xu_dot * u], [Yv_dot * v, -Xu_dot * u, 0]]
        )

    def dynamics(
        self,
        eta: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        tau_ext: npt.NDArray[np.float64] = None,
    ) -> tp.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Returns the time derivatives :math:`(\dot{\eta}, \dot{\nu})` for the 3DOF model:

        .. math:
            \dot{eta} = J(\eta) \nu \\
            (M_{RB} + M_A) \dot{\nu} + (C_{RB}(\nu) + C_A(\nu)) \nu + D \nu = \tau + \tau_{ext}.

        External forces :math:`\tau_{ext}` can be provided (default is zero).
        """
        # Public method keeps original semantics (returns new arrays) while
        # delegating core math to in-place implementation to reduce internal allocations.
        eta_dot = np.empty(3, dtype=float)
        nu_dot = np.empty(3, dtype=float)
        self._dynamics_into(eta, nu, tau, tau_ext, eta_dot, nu_dot)
        return eta_dot, nu_dot

    # ------------------------------------------------------------------
    # Internal optimized dynamics: writes results into provided buffers.
    # Avoids constructing intermediate small arrays (J, C matrices, rhs).
    # ------------------------------------------------------------------
    def _dynamics_into(
        self,
        eta: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        tau_ext: npt.NDArray[np.float64],
        eta_dot_out: npt.NDArray[np.float64],
        nu_dot_out: npt.NDArray[np.float64],
    ) -> None:
        if tau_ext is None:
            tau_ext = self._zeros3

        # Unpack states
        psi = eta[2]
        u, v, r = nu

        c = np.cos(psi)
        s = np.sin(psi)

        # eta_dot = J(psi) * nu (manual expand to avoid matrix mult allocation)
        eta_dot_out[0] = c * u - s * v
        eta_dot_out[1] = s * u + c * v
        eta_dot_out[2] = r

        # Coriolis * nu contributions (pre- multiplied form) for rigid-body & added mass
        m = self.m
        Xu_dot = self.MA[0, 0]
        Yv_dot = self.MA[1, 1]
        # C_total @ nu only has first two components non-zero
        cnu0 = -(m + Yv_dot) * v * r
        cnu1 = (m + Xu_dot) * u * r

        # Damping (diagonal) * nu
        Dnu0 = self.D[0, 0] * u
        Dnu1 = self.D[1, 1] * v
        Dnu2 = self.D[2, 2] * r

        # rhs = tau + tau_ext - Dnu - Cnu  (no restoring forces in this planar model)
        rhs0 = tau[0] + tau_ext[0] - Dnu0 - cnu0
        rhs1 = tau[1] + tau_ext[1] - Dnu1 - cnu1
        rhs2 = tau[2] + tau_ext[2] - Dnu2  # cnu2 == 0

        # nu_dot = invM_eff @ rhs (manual 3x3 matvec multiply)
        M = self.invM_eff
        nu_dot_out[0] = M[0, 0] * rhs0 + M[0, 1] * rhs1 + M[0, 2] * rhs2
        nu_dot_out[1] = M[1, 0] * rhs0 + M[1, 1] * rhs1 + M[1, 2] * rhs2
        nu_dot_out[2] = M[2, 0] * rhs0 + M[2, 1] * rhs1 + M[2, 2] * rhs2

    def step(
        self,
        tau: npt.NDArray[np.float64] = None,
        tau_ext: npt.NDArray[np.float64] = None,
        dt: float = 0.01,
    ):
        r"""
        Advance the state :math:`(\eta, \nu)` one time step dt using 4th-order Runge-Kutta.
        """
        if tau is None:
            tau = self._zeros3.copy()  # ensure user can mutate after call
        if tau_ext is None:
            tau_ext = self._zeros3  # safe shared read-only

        eta0 = self.eta  # references (updated at end)
        nu0 = self.nu

        k_eta = self._k_eta
        k_nu = self._k_nu
        eta_tmp = self._eta_tmp
        nu_tmp = self._nu_tmp

        # -- k1 --
        self._dynamics_into(eta0, nu0, tau, tau_ext, k_eta[0], k_nu[0])

        # -- k2 --
        eta_tmp[:] = eta0 + 0.5 * dt * k_eta[0]
        nu_tmp[:] = nu0 + 0.5 * dt * k_nu[0]
        self._dynamics_into(eta_tmp, nu_tmp, tau, tau_ext, k_eta[1], k_nu[1])

        # -- k3 --
        eta_tmp[:] = eta0 + 0.5 * dt * k_eta[1]
        nu_tmp[:] = nu0 + 0.5 * dt * k_nu[1]
        self._dynamics_into(eta_tmp, nu_tmp, tau, tau_ext, k_eta[2], k_nu[2])

        # -- k4 --
        eta_tmp[:] = eta0 + dt * k_eta[2]
        nu_tmp[:] = nu0 + dt * k_nu[2]
        self._dynamics_into(eta_tmp, nu_tmp, tau, tau_ext, k_eta[3], k_nu[3])

        # Combine increments (in-place update to existing state arrays)
        eta0 += (dt / 6.0) * (k_eta[0] + 2 * k_eta[1] + 2 * k_eta[2] + k_eta[3])
        nu0 += (dt / 6.0) * (k_nu[0] + 2 * k_nu[1] + 2 * k_nu[2] + k_nu[3])

    def get_states(self) -> tp.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Returns a copy of the current states: :math:`\eta, \nu`
        """
        return self.eta.copy(), self.nu.copy()
