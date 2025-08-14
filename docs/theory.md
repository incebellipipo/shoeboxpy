## 1. Geometry & Rigid‑Body Inertia

Symbols:

| Symbol | Meaning | Units (example) |
|--------|---------|-----------------|
| $L$ | Length | $m$ |
| $B$ | Beam / width | $m$ |
| $T$ | Height / depth | $m$ |
| $\rho$ | Fluid density | $kg\,m^3$ (1000 for water) |
| $m$ | Mass | $kg$ |
| $I_x,I_y,I_z$ | Principal inertias about body axes | $kg \, m^2$ |

### Mass

For a full solid of uniform density:

$$
m = \rho L B T.
$$

### Moments of Inertia (through centroid)

For a rectangular prism (axes aligned with edges):

$$
I_x = \tfrac{1}{12} m (B^2 + T^2),\\
I_y = \tfrac{1}{12} m (L^2 + T^2),\\
I_z = \tfrac{1}{12} m (L^2 + B^2).\
$$

Rigid‑body mass-inertia matrix (no products of inertia, CG at origin):

$$
\mathbf{M}-{\mathrm{RB}} = \mathrm{diag}(m,m,m, I_x, I_y, I_z).
$$

## 2. Added Mass & Damping

### Added Mass

Diagonal approximation:

$$
\mathbf{M}_{\mathrm{A}} = \mathrm{diag}(X_{\dot u}, Y_{\dot v}, Z_{\dot w}, K_{\dot p}, M_{\dot q}, N_{\dot r}).
$$

Often parameterised by fractions of the corresponding rigid‑body terms (e.g. $X_{\dot u} = \alpha_u m$, $K_{\dot p}=\alpha_p I_x$, etc.).

Effective mass:

$$
\mathbf{M}_{\mathrm{eff}} = \mathbf{M}_{\mathrm{RB}} + \mathbf{M}_{\mathrm{A}}.
$$

### Linear Damping

Coefficients collected in a diagonal matrix (coefficients only):

$$
\mathbf{D} = \mathrm{diag}(d_u, d_v, d_w, d_p, d_q, d_r).
$$

Applied force:

$$
\mathbf{D}\nu = [d_u u,\ d_v v,\ d_w w,\ d_p p,\ d_q q,\ d_r r]^\top
$$

(sign convention embedded in $d_i$).

## 3. Kinematics

State vectors:

$$
\eta = [x, y, z, \phi, \theta, \psi]^\top, \qquad
\nu = [u, v, w, p, q, r]^\top.
$$

Kinematic relation:

$$
\dot{\eta} = \mathbf{J}(\eta)\nu.
$$

Block structure:

$$
  \mathbf{J}(\eta)=
  \begin{bmatrix}
    R_{\mathrm{lin}}(\phi,\theta,\psi) & \mathbf{0} \\
    \mathbf{0} & T_{\mathrm{ang}}(\phi,\theta)
  \end{bmatrix}
$$

with standard Z-Y-X rotation composition $R_{\mathrm{lin}} = R_z(\psi)R_y(\theta)R_x(\phi)$ and Euler‑rate mapping $T_{\mathrm{ang}}$.

## 4. Dynamics

Full 6‑DOF equation (body frame):

$$
(\mathbf{M}_{\mathrm{RB}} + \mathbf{M}_{\mathrm{A}})\dot{\nu} + (\mathbf{C}_{\mathrm{RB}}(\nu) + \mathbf{C}_{\mathrm{A}}(\nu))\nu + \mathbf{D}\nu
  = \tau + \tau_{\mathrm{ext}} + \mathbf{g}_{\mathrm{restoring}}(\eta).
$$

### Coriolis / Centripetal Terms

Let $\mathbf{v} = [u,v,w]^\top$, $\omega = [p,q,r]^\top$, and $S(\cdot)$ the skew operator.

Rigid‑body:

$$
\mathbf{C}_{\mathrm{RB}}(\nu) = \begin{bmatrix} \mathbf{0} & -m S(\omega) \\ -m S(\mathbf{v}) & - S(\mathbf{I}\omega) \end{bmatrix},\quad
\mathbf{I}\omega = [I_x p, I_y q, I_z r].
$$

Added mass (partition $\mathbf{M}_{\mathrm{A}}$ into linear / rotational blocks):

$$
\mathbf{C}_{\mathrm{A}}(\nu) = \begin{bmatrix} \mathbf{0} & - S(\mathbf{M}_{\mathrm{A,lin}}\mathbf{v}) \\ - S(\mathbf{M}_{\mathrm{A,lin}}\mathbf{v}) & - S(\mathbf{M}_{\mathrm{A,rot}}\omega) \end{bmatrix}.
$$

### Damping Force Vector

$$
\mathbf{D}\nu = [d_u u,\ d_v v,\ d_w w,\ d_p p,\ d_q q,\ d_r r]^\top.
$$

### Hydrostatics (Heave)

Small displacement approximation:
$ F_z = - \rho g L B\, z. $

### Restoring (Roll / Pitch)

Small‑angle moments:

$$
\mathbf{g}_{\mathrm{restoring}}(\eta) = \begin{bmatrix}
0 \\ 0 \\ 0 \\ - m g\, \mathrm{GM}_{\phi} \, \phi \\ - m g \, \mathrm{GM}_{\theta} \, \theta \\ 0
\end{bmatrix}.
$$

## 5. Time Integration

System:

$$
\dot{\eta} = \mathbf{J}(\eta)\nu,
$$

$$
\mathbf{M}_{\mathrm{eff}} \dot{\nu} &+ (\mathbf{C}_{\mathrm{RB}} + \mathbf{C}_{\mathrm{A}})\nu + \mathbf{D}\nu = \tau + \tau_{\mathrm{ext}} + \mathbf{g}_{\mathrm{restoring}}(\eta).
$$

Implemented with classical 4th‑order Runge-Kutta (RK4) over step $\Delta t$:

1. Evaluate derivatives at current state.
2. Two midpoint evaluations (using half steps).
3. Final endpoint evaluation.
4. Weighted combination to advance $(\eta,\nu)$.
