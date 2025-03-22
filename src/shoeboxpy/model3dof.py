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
        if tau_ext is None:
            tau_ext = np.zeros(3)

        # Kinematics
        eta_dot = self.J(eta) @ nu

        # Coriolis/centripetal effects
        C_total = self.C_RB(nu) + self.C_A(nu)

        # For a planar vessel, there are no hydrostatic restoring forces (set to zero)
        g_rest = np.zeros(3)

        # Compute right-hand side (forces minus damping and Coriolis effects)
        rhs = tau + tau_ext + g_rest - self.D @ nu - C_total @ nu

        # Solve for acceleration in body frame
        nu_dot = self.invM_eff @ rhs

        return eta_dot, nu_dot

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
            tau = np.zeros(3)
        if tau_ext is None:
            tau_ext = np.zeros(3)

        eta0 = self.eta
        nu0 = self.nu

        # -- k1 --
        k1_eta, k1_nu = self.dynamics(eta0, nu0, tau, tau_ext)

        # -- k2 --
        eta_temp = eta0 + 0.5 * dt * k1_eta
        nu_temp = nu0 + 0.5 * dt * k1_nu
        k2_eta, k2_nu = self.dynamics(eta_temp, nu_temp, tau, tau_ext)

        # -- k3 --
        eta_temp = eta0 + 0.5 * dt * k2_eta
        nu_temp = nu0 + 0.5 * dt * k2_nu
        k3_eta, k3_nu = self.dynamics(eta_temp, nu_temp, tau, tau_ext)

        # -- k4 --
        eta_temp = eta0 + dt * k3_eta
        nu_temp = nu0 + dt * k3_nu
        k4_eta, k4_nu = self.dynamics(eta_temp, nu_temp, tau, tau_ext)

        # Combine increments
        self.eta = eta0 + (dt / 6.0) * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        self.nu = nu0 + (dt / 6.0) * (k1_nu + 2 * k2_nu + 2 * k3_nu + k4_nu)

    def get_states(self) -> tp.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Returns a copy of the current states: :math:`\eta, \nu`
        """
        return self.eta.copy(), self.nu.copy()
