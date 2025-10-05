import numpy as np
from shoeboxpy.model3dof import Shoebox
from shoeboxpy.animate import animate_history
import matplotlib.pyplot as plt


# Initialize the 3-DOF shoebox model (surge, sway, yaw)
shoebox = Shoebox(
    L=1.0,  # Length (m)
    B=0.3,  # Breadth (m)
    T=0.03,  # Draft (m)
    eta0=np.array([0.0, 0.0, 0.0]),  # Initial [x, y, psi]
    nu0=np.zeros(3),  # Initial [u, v, r]
)

dt = 0.01
T = np.arange(0, 10, dt)
pose_history = np.zeros((len(T), 3))  # [x, y, psi]
velocity_history = np.zeros((len(T), 3))  # [u, v, r]

# Simulate for 10 seconds with a time step of 0.01 seconds
for i, t in enumerate(T):
    # Apply control forces in surge, sway and yaw
    shoebox.step(tau=np.array([2.0, 0.5, 0.1]), dt=dt)
    position, velocity = shoebox.get_states()
    pose_history[i] = position  # Store the current state
    velocity_history[i] = velocity  # Store the current velocities

# Plot final on a matplotlib figure. Left: position (x,y), Right: velocities (u,v,r)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(T, pose_history[:, 0], label='x (m)')
plt.plot(T, pose_history[:, 1], label='y (m)')
plt.legend(); plt.title('Position')
plt.xlabel('Time (s)')

plt.subplot(1, 2, 2)
plt.plot(T, velocity_history[:, 0], label='u (m/s)')
plt.plot(T, velocity_history[:, 1], label='v (m/s)')
plt.plot(T, velocity_history[:, 2], label='r (rad/s)')
plt.legend(); plt.title('Velocities')
plt.xlabel('Time (s)')

plt.tight_layout()

# Animate using the 3-DOF pose history (x, y, psi)
animate_history(pose_history, dt=dt, L=1.0, B=0.3, T=0.03)