import numpy as np
import numpy.typing as npt
import typing as tp

import shoeboxpy.utils as utils
from shoeboxpy.utils import skew


class Shoebox:
    r"""
    6-DOF rectangular "shoebox" vessel with:
      - Rigid-body mass & inertia (diagonal) from geometry (L, B, T).
      - Added mass (diagonal) from user-chosen dimensionless factors.
      - Linear damping (diagonal).
      - Simple linear restoring in roll & pitch for small angles (metacentric method).
      - **Coriolis & centripetal** effects from both rigid-body and added mass.

    States:

    .. math::
        \eta = [x, y, z, \phi, \theta, \psi]
        \nu = [u, v, w, p, q, r]

    The dynamics are:

    .. math::
        \dot{\eta} = J(\eta)\nu
    .. math::
        (M_{RB} + M_A)\dot{\nu} + (C_{RB}(\nu) + C_A(\nu))\nu + D\nu =
        \tau + \tau_{\mathrm{ext}} + g_{\mathrm{restoring}}(\eta).


    :param L: Length of the shoebox (m)
    :param B: Width of the shoebox (m)
    :param T: Height of the shoebox (m)
    :param rho: Density of the fluid :math:`(kg/m^3)`
    :param alpha_*, beta_*: Added mass & damping coefficients (dimensionless)
    :param GM_phi, GM_theta: Metacentric heights for roll & pitch
    :param g: Gravitational acceleration :math:`(m/s^2)`
    :param eta0: Initial :math:`[x, y, z, \phi, \theta, \psi]`
    :param nu0: Initial :math:`[u, v, w, p, q, r]`
    """

    def __init__(
        self,
        L: float,
        B: float,
        T: float,
        rho: float = 1000.0,
        # Added mass coefficients
        alpha_u: float = 0.1,
        alpha_v: float = 0.2,
        alpha_w: float = 1.0,
        alpha_p: float = 0.1,
        alpha_q: float = 0.1,
        alpha_r: float = 0.1,
        # Damping factors
        beta_u: float = 0.05,
        beta_v: float = 0.05,
        beta_w: float = 0.05,
        beta_p: float = 0.05,
        beta_q: float = 0.05,
        beta_r: float = 0.05,
        # Restoring parameters
        GM_phi: float = 0.0,    # metacentric height in roll
        GM_theta: float = 0.0,  # metacentric height in pitch
        g: float = 9.81,        # gravitational acceleration
        # Initial states
        eta0: npt.NDArray[np.float64] = np.zeros(6),
        nu0: npt.NDArray[np.float64] = np.zeros(6),
    ):
        r"""
        Initialize the shoebox model.

        :param L: Length of the shoebox (m)
        :param B: Width of the shoebox (m)
        :param T: Height of the shoebox (m)
        :param rho: Density of the fluid :math:`(kg/m^3)`
        :param alpha_*, beta_*: Added mass & damping coefficients (dimensionless)
        :param GM_phi, GM_theta: Metacentric heights for roll & pitch
        :param g: Gravitational acceleration :math:`(m/s^2)`
        :param eta0: Initial :math:`[x, y, z, \phi, \theta, \psi]`
        :param nu0: Initial :math:`[u, v, w, p, q, r]`
        :return: None
        """
        # 1) Rigid-body mass from volume (the code uses full L*B*T)
        self.m = rho * L * B * T
        print(f"Mass: {self.m:.2f} kg")

        # 2) Moments of inertia (uniform box, diagonal)
        Ix = (1.0 / 12.0) * self.m * (B**2 + T**2)
        Iy = (1.0 / 12.0) * self.m * (L**2 + T**2)
        Iz = (1.0 / 12.0) * self.m * (L**2 + B**2)

        self.MRB = np.diag([self.m, self.m, self.m, Ix, Iy, Iz])

        # 3) Added mass (diagonal)
        self.MA = np.diag([
            alpha_u * self.m,
            alpha_v * self.m,
            alpha_w * self.m,
            alpha_p * Ix,
            alpha_q * Iy,
            alpha_r * Iz,
        ])
        self.M_eff = self.MRB + self.MA

        # 4) Linear damping (diagonal)
        self.D = np.diag([
            beta_u * self.m,
            beta_v * self.m,
            beta_w * self.m,
            beta_p * Ix,
            beta_q * Iy,
            beta_r * Iz,
        ])

        self.invM_eff = np.linalg.inv(self.M_eff)

        # 5) Restoring in roll & pitch
        self.GM_phi = GM_phi
        self.GM_theta = GM_theta
        self.g = g

        # Store states
        self.eta = eta0.astype(float)
        self.nu  = nu0.astype(float)

    def J(self, eta):
        r"""
        Computes the 6x6 transformation matrix `J` that maps body velocities
        to the inertial frame velocities :math:`[\dot{x}, \dot{y}, \dot{y}, \dot{\phi}, \dot{\theta}, \dot{\psi}]`.
        """
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]

        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth  = np.cos(theta)
        sth  = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Rotation for linear part (body->inertial)
        R11 = cth * cpsi
        R12 = sphi * sth * cpsi - cphi * spsi
        R13 = cphi * sth * cpsi + sphi * spsi

        R21 = cth * spsi
        R22 = sphi * sth * spsi + cphi * cpsi
        R23 = cphi * sth * spsi - sphi * cpsi

        R31 = -sth
        R32 = sphi * cth
        R33 = cphi * cth

        R_lin = np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])

        eps = 1e-9
        tth  = sth / max(cth, eps)
        scth = 1.0 / max(cth, eps)

        T_ang = np.array([
            [1.0, sphi*tth,  cphi*tth],
            [0.0, cphi,      -sphi   ],
            [0.0, sphi*scth, cphi*scth]
        ])

        return np.block([
            [R_lin, np.zeros((3,3))],
            [np.zeros((3,3)), T_ang]
        ])

    def C_RB(self, nu: np.ndarray) -> np.ndarray:
        r"""
        Rigid-body Coriolis/centripetal matrix for diagonal :math:`M_{RB}`.
        (assuming CG at origin, no product of inertia).
        """
        u, v, w, p, q, r = nu

        m  = self.m
        Ix = self.MRB[3,3]
        Iy = self.MRB[4,4]
        Iz = self.MRB[5,5]

        # Build the 6x6 in block form:
        # top-left = 0
        # top-right = -m * S([p,q,r])
        # bottom-left = -m * S([u,v,w])
        # bottom-right = - S(I * [p,q,r])
        C = np.zeros((6,6))

        v_b = np.array([u, v, w])
        w_b = np.array([p, q, r])
        Iw_b = np.array([Ix*p, Iy*q, Iz*r])  # diagonal inertia times w

        C[:3, :3] = 0.0
        C[:3, 3:] = -m * skew(w_b)
        C[3:, :3] = -skew(m*v_b)
        C[3:, 3:] = -skew(Iw_b)

        return C

    def C_A(self, nu: np.ndarray) -> np.ndarray:
        r"""
        Added-mass Coriolis/centripetal matrix for diagonal :math:`M_A`.
        """
        u, v, w, p, q, r = nu

        Xudot = self.MA[0,0]
        Yvdot = self.MA[1,1]
        Zwdot = self.MA[2,2]
        Kpdot = self.MA[3,3]
        Mqdot = self.MA[4,4]
        Nrdot = self.MA[5,5]

        # Similar block structure, using "added mass" equivalents
        C = np.zeros((6,6))

        v_b = np.array([u, v, w])
        w_b = np.array([p, q, r])

        # linear part: M_A,lin * v_b
        # rotational part: M_A,rot * w_b
        Mlin_v = np.array([Xudot*u, Yvdot*v, Zwdot*w])
        Mrot_w = np.array([Kpdot*p, Mqdot*q, Nrdot*r])

        # top-left = 0
        # top-right = - skew( M_A,lin v_b )
        # bottom-left = - skew( M_A,lin v_b )
        # bottom-right = - skew( M_A,rot w_b )
        C[:3, :3] = 0.0
        C[:3, 3:] = - skew(Mlin_v)
        C[3:, :3] = - skew(Mlin_v)
        C[3:, 3:] = - skew(Mrot_w)

        return C

    def restoring_forces(self, eta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Compute roll/pitch restoring moment in BODY frame for small angles:

        .. math::
            K = - m g GM_{\phi}  \phi \\
            M = - m g GM_{\theta}  \theta

        :param eta: :math:`[x, y, z, \phi, \theta, \psi]`
        :return: Restoring forces :math:`[0, 0, 0, K, M, 0]`
        """
        phi = eta[3]
        theta = eta[4]

        K_rest = -self.m * self.g * self.GM_phi * phi
        M_rest = -self.m * self.g * self.GM_theta * theta

        return np.array([0.0, 0.0, 0.0, K_rest, M_rest, 0.0])

    def dynamics(
        self,
        eta: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        tau_ext: npt.NDArray[np.float64],
    ) -> tp.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Returns the time derivatives :math:`(\dot{\eta}, \dot{\nu})` for the 6DOF model:

        .. math::
            \dot{\eta} = J(\eta)\nu
            (M_{RB} + M_A)\dot{\nu}
             + (C_{RB}(\nu) + C_A(\nu))\nu
             + D\nu
             = \tau + \tau_{\mathrm{ext}} + g_{\mathrm{restoring}}(\eta)

        """
        # 1) Kinematics
        J_ = self.J(eta)
        eta_dot = J_ @ nu

        # 2) Coriolis/centripetal
        C_RB_ = self.C_RB(nu)
        C_A_  = self.C_A(nu)
        C_ = C_RB_ + C_A_

        # 3) Restoring
        g_rest = self.restoring_forces(eta)

        # 4) Sum up
        rhs = tau + tau_ext + g_rest - self.D @ nu - C_ @ nu

        # Solve for nu_dot
        nu_dot = self.invM_eff @ rhs

        return eta_dot, nu_dot

    def step(self, tau=None, tau_ext=None, dt=0.01):
        r"""
        Advance :math:`(\eta, \nu)` by one time step dt using 4th-order Runge-Kutta.
        """
        if tau is None:
            tau = np.zeros(6)
        if tau_ext is None:
            tau_ext = np.zeros(6)

        # Current state
        eta0 = self.eta
        nu0 = self.nu

        # -- k1 --
        k1_eta, k1_nu = self.dynamics(eta0, nu0, tau, tau_ext)

        # -- k2 --
        eta_temp = eta0 + 0.5 * dt * k1_eta
        nu_temp  = nu0  + 0.5 * dt * k1_nu
        k2_eta, k2_nu = self.dynamics(eta_temp, nu_temp, tau, tau_ext)

        # -- k3 --
        eta_temp = eta0 + 0.5 * dt * k2_eta
        nu_temp  = nu0  + 0.5 * dt * k2_nu
        k3_eta, k3_nu = self.dynamics(eta_temp, nu_temp, tau, tau_ext)

        # -- k4 --
        eta_temp = eta0 + dt * k3_eta
        nu_temp  = nu0  + dt * k3_nu
        k4_eta, k4_nu = self.dynamics(eta_temp, nu_temp, tau, tau_ext)

        # Combine
        self.eta = eta0 + (dt / 6.0) * (k1_eta + 2.0*k2_eta + 2.0*k3_eta + k4_eta)
        self.nu  = nu0  + (dt / 6.0) * (k1_nu  + 2.0*k2_nu  + 2.0*k3_nu  + k4_nu)

    def get_states(self):
        return self.eta.copy(), self.nu.copy()