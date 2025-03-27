## Geometry and Rigid‐Body Inertia

Let:
- $L$ = length of the box,
- $B$ = width (beam),
- $T$ = height/depth,
- $\rho$ = fluid density (e.g. 1000 kg/m³ for water).

### Mass

Computing the **mass** of the rectangular prism (assuming it is fully of density $\rho$) as
$$
  m = \rho LBT.
$$


### Moments of Inertia

For a uniform **rectangular prism** of mass $m$ with dimensions $(L,B,T)$, the **principal moments of inertia** about its center (aligned with the box axes) are:

$$
  I_x = \frac{1}{12}m\bigl(B^2 + T^2\bigr),
  \quad
  I_y = \frac{1}{12}m\bigl(L^2 + T^2\bigr),
  \quad
  I_z = \frac{1}{12}m\bigl(L^2 + B^2\bigr).
$$

Hence, the **rigid‐body mass–inertia matrix** $\mathbf{M}_{\mathrm{RB}}$ (assuming no products of inertia and center of gravity at the body origin) is:

$$
  \mathbf{M}_{\mathrm{RB}}
  =
  \mathrm{diag}(m,m,m,I_x,I_y,I_z).
$$

## Added Mass and Damping

### Added Mass

**Diagonal added mass** approximates hydrodynamic effects. Each entry is a fraction of the corresponding rigid‐body mass or inertia:

$$
  \mathbf{M}_{\mathrm{A}}
  =
  \mathrm{diag}\bigl(
    X_{\dot{u}},Y_{\dot{v}},Z_{\dot{w}},K_{\dot{p}},M_{\dot{q}},N_{\dot{r}}
  \bigr).
$$

$$
X_{\dot{u}} = \alpha_um,
\quad
Y_{\dot{v}} = \alpha_vm,
\quad
Z_{\dot{w}} = \alpha_wm,
\quad
K_{\dot{p}} = \alpha_pI_x,
\quad
M_{\dot{q}} = \alpha_qI_y,
\quad
N_{\dot{r}} = \alpha_rI_z.
$$

Hence the **effective mass** is

$$
  \mathbf{M}_{\mathrm{eff}}
  =
  \mathbf{M}_{\mathrm{RB}}
  +
  \mathbf{M}_{\mathrm{A}}.
$$

### Linear Damping

For each DOF, you have a **linear damping coefficient**, also forming a **diagonal** matrix $\mathbf{D}$. Each diagonal entry is typically some dimensionless factor times $m$ or the inertia:

$$
  \mathbf{D}
  =
  \mathrm{diag}\bigl(
    d_u,d_v,d_w,d_p,d_q,d_r
  \bigr).
$$


## Kinematics

$$
\eta
=
\begin{bmatrix}
x ,
y ,
z ,
\phi ,
\theta ,
\psi
\end{bmatrix} ^\top
,\quad
\nu
=
\begin{bmatrix}
u ,
v ,
w ,
p ,
q ,
r
\end{bmatrix}^\top,
$$

where $\eta$ is position/orientation in an inertial frame, and $\nu$ is velocity in the body frame. The **kinematic** relation is:

$$
  \dot{\eta}
  =
  \mathbf{J}(\eta)\nu,
$$

where $\mathbf{J}(\eta)$ is a **6×6** block‐diagonal matrix:

$$
\mathbf{J}(\eta)
=
\begin{bmatrix}
R_{\mathrm{lin}}(\phi,\theta,\psi) & \mathbf{0}_{3\times3}\\
\mathbf{0}_{3\times3} & T_{\mathrm{ang}}(\phi,\theta)
\end{bmatrix}.
$$

- $R_{\mathrm{lin}}(\phi,\theta,\psi)$ is the standard rotation matrix $\mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$ for the linear velocities.
- $T_{\mathrm{ang}}(\phi,\theta)$ maps $(p,q,r)$ to the Euler‐angle rates $(\dot{\phi},\dot{\theta},\dot{\psi})$.


## Dynamics

### Full Equation

The **6‐DOF** body‐frame dynamic equation is:

$$
  (\mathbf{M}_{\mathrm{RB}} + \mathbf{M}_{\mathrm{A}})\dot{\nu}
  +
  \bigl(\mathbf{C}_{\mathrm{RB}}(\nu) + \mathbf{C}_{\mathrm{A}}(\nu)\bigr)\nu
  +
  \mathbf{D}\nu
  =
  \tau
  +
  \tau_{\mathrm{ext}}
  +
  \mathbf{g}_{\mathrm{restoring}}(\eta).
$$

where:

- $\tau$ and $\tau_{\mathrm{ext}}$ are **control** and **external** forces/moments in the body frame.
- $\mathbf{g}_{\mathrm{restoring}}(\eta)$ is your simple roll/pitch restoring.

### Coriolis and Centripetal

You include **rigid‐body** and **added‐mass** Coriolis/centripetal terms:

1. **Rigid‐Body Coriolis** $\mathbf{C}_{\mathrm{RB}}(\nu)$

   For diagonal $\mathbf{M}_{\mathrm{RB}}$ and center of gravity at the origin:

   $$
   \mathbf{C}_{\mathrm{RB}}(\nu)
   =
   \begin{bmatrix}
   \mathbf{0} & -mS(\omega) \\
   -mS(\mathbf{v}) & -S(\mathbf{I}\omega)
   \end{bmatrix},
   $$
   where:
   - $\mathbf{v}=[u,v,w]^\top$,
   - $\omega=[p,q,r]^\top$,
   - $\mathbf{I}\omega = [I_xp, I_yq, I_zr]$,
   - $S(\cdot)$ is the skew‐symmetric operator.

2. **Added‐Mass Coriolis** $\mathbf{C}_{\mathrm{A}}(\nu)$

   With diagonal $\mathbf{M}_{\mathrm{A}}=\mathrm{diag}(X_{\dot{u}}, \dots, N_{\dot{r}})$, we have an analogous structure:

   $$
   \mathbf{C}_{\mathrm{A}}(\nu)
   =
   \begin{bmatrix}
   \mathbf{0} & -S(\mathbf{M}_{\mathrm{A,lin}}\mathbf{v}) \\
   -S(\mathbf{M}_{\mathrm{A,lin}}\mathbf{v}) & -S(\mathbf{M}_{\mathrm{A,rot}}\omega)
   \end{bmatrix}.
   $$

### Damping

Linear damping:

$$
\mathbf{D}\nu
=
\begin{bmatrix}
d_uu,
d_vv,
d_ww,
d_pp,
d_qq,
d_rr
\end{bmatrix}^\top.
$$

### Restoring (Roll/Pitch)

For small angles, the restoring moment is:

$$
\mathbf{g}_{\mathrm{restoring}}(\eta)
=
\begin{bmatrix}
0 ,
0 ,
0 ,
-mg(\mathrm{GM}_{\phi})\phi ,
-mg(\mathrm{GM}_{\theta})\theta ,
0
\end{bmatrix},
$$
where $\mathrm{GM}_\phi,\mathrm{GM}_\theta$ are **metacentric heights** in roll/pitch.

## Integration

Numerically, you solve the system:

$$
\begin{aligned}
&\dot{\eta} = \mathbf{J}(\eta)\nu,\\
&\mathbf{M}_{\mathrm{eff}}\dot{\nu}
  + \bigl(\mathbf{C}_{\mathrm{RB}}+\mathbf{C}_{\mathrm{A}}\bigr)\nu
  + \mathbf{D}\nu
  = \tau + \tau_{\mathrm{ext}} + \mathbf{g}_{\mathrm{restoring}}(\eta).
\end{aligned}
$$

Your code does a **4th‐order Runge–Kutta** step over each timestep $\Delta t$:

1. Compute $(\dot{\eta},\dot{\nu})$ at the current state $(\eta,\nu)$.
2. Evaluate intermediate steps $(\eta + \frac{1}{2}k_1,\nu + \frac{1}{2}k_1)$, etc.
3. Combine via RK4.

This yields updated states $\eta(t+\Delta t)$ and $\nu(t+\Delta t)$.

## Summary

1. **Mass & Inertia** of a rectangular prism:
   $$
   m = \rho L B T,
   \quad
   I_x = \frac{1}{12}m(B^2 + T^2),
   \quad
   I_y = \frac{1}{12}m(L^2 + T^2),
   \quad
   I_z = \frac{1}{12}m(L^2 + B^2).
   $$
   $$
   \mathbf{M}_{\mathrm{RB}} = \mathrm{diag}(m,m,m,I_x,I_y,I_z).
   $$
   $$
   \mathbf{M}_{\mathrm{A}} = \mathrm{diag}(X_{\dot{u}},Y_{\dot{v}},Z_{\dot{w}},K_{\dot{p}},M_{\dot{q}},N_{\dot{r}}).
   $$

2. **Kinematics**:
   $$
   \dot{\eta} = \mathbf{J}(\eta)\nu,
   $$
   with $\eta=[x,y,z,\phi,\theta,\psi]$ in inertial frame, $\nu=[u,v,w,p,q,r]$ in body frame.

3. **Dynamics**:
   $$
   (\mathbf{M}_{\mathrm{RB}} + \mathbf{M}_{\mathrm{A}})\dot{\nu}
     + \Bigl(\mathbf{C}_{\mathrm{RB}}(\nu)+\mathbf{C}_{\mathrm{A}}(\nu)\Bigr)\nu
     + \mathbf{D}\nu
   =
   \tau + \tau_{\mathrm{ext}} + \mathbf{g}_{\mathrm{restoring}}(\eta).
   $$

4. **Restoring** (small‐angle roll/pitch):
   $$
   \mathbf{g}_{\mathrm{restoring}}(\eta)
   =
   \begin{bmatrix}
   0 ,
   0 ,
   0 ,
   -mg\mathrm{GM}_{\phi}\phi ,
   -mg\mathrm{GM}_{\theta}\theta ,
   0
   \end{bmatrix}.
   $$