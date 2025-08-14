import numpy as np
from shoeboxpy.model6dof import Shoebox
from shoeboxpy.animate import animate_history
import matplotlib.pyplot as plt


# Initialize the shoebox model
shoebox = Shoebox(
    L=1.0,  # Length (m)
    B=0.3,  # Width (m)
    T=0.03,  # Height (m)
    eta0=np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]),  # Initial position and orientation
    nu0=np.zeros(6),  # Initial velocities
    GM_phi=0.2,  # Metacentric height in roll
    GM_theta=0.2,  # Metacentric height in pitch
)

dt = 0.01
T = np.arange(0, 10, dt)
pose_history = np.zeros((len(T), 6))  # Initialize pose history
velocity_history = np.zeros((len(T), 6))  # Initialize velocity history

# Simulate for 10 seconds with a time step of 0.01 seconds
for i, t in enumerate(T):
    shoebox.step(tau=np.array([2.0, 0.5, 0.0, 0.0, 0.0, 0.1]), dt=dt)  # Apply control forces
    position, velocity = shoebox.get_states()
    pose_history[i] = position  # Store the current state
    velocity_history[i] = velocity  # Store the current velocities

# Plot final on a matplotlib figure. left side of the plot shows the position, right side shows the velocity.
# Plot final on a matplotlib figure. left side of the plot shows the position, right side shows the velocity, and bottom shows the XY plot.
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(T, pose_history[:, 0], label='x (m)')
plt.plot(T, pose_history[:, 1], label='y (m)')
plt.plot(T, pose_history[:, 2], label='z (m)')
plt.title('Position over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid()
plt.legend()
plt.subplot(1, 3, 2)

plt.plot(T, velocity_history[:, 0], label='u (m/s)')
plt.plot(T, velocity_history[:, 1], label='v (m/s)')
plt.plot(T, velocity_history[:, 2], label='w (m/s)')
plt.title('Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(pose_history[:, 0], pose_history[:, 1])
plt.title('X-Y Plot')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()




animate_history(pose_history, dt=dt, L=1.0, B=0.3, T=0.2)  # Animate the results