import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


def animate_history(
    pos_history: np.ndarray, dt: float, L: float = 1.0, B: float = 0.5, T: float = 0.3
):
    """
    Animates the vessel motion from its position history with moving axis ticks.

    :param pos_history: np.ndarray
        Array of states. Each row is assumed to be either:
          - [x, y, z, phi, theta, psi] (6 elements), or
          - [x, y, z] (3 elements) if no rotation is desired.
    :param dt: float
        Time step between frames in seconds.
    :param L: float, optional
        Length of the vessel (default 1.0 m).
    :param B: float, optional
        Width of the vessel (default 0.5 m).
    :param T: float, optional
        Height of the vessel (default 0.3 m).
    """
    n_frames = pos_history.shape[0]

    # Set up the figure and 3D axis.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Initial axis limits (they will be updated each frame).
    margin = L * 2
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)

    # Create an object for the trace.
    (trace_line,) = ax.plot([], [], [], "b-", lw=2)

    # Use a mutable container to store the vessel Poly3DCollection.
    vessel_poly_obj = [None]

    # This list will store the absolute positions for the trace.
    trace_data = []

    def init():
        trace_line.set_data([], [])
        trace_line.set_3d_properties([])
        return (trace_line,)

    def animate(i):
        nonlocal trace_data

        # Extract the current state.
        # If six columns are provided, use columns 0:3 for position and 3:6 for Euler angles.
        if pos_history.shape[1] >= 6:
            pos = pos_history[i, :3]
            angles = pos_history[i, 3:6]
        else:
            pos = pos_history[i, :3]
            angles = [0, 0, 0]

        # Append current absolute position to trace data.
        trace_data.append(pos)
        trace_data_np = np.array(trace_data)

        # Update the trace (in absolute coordinates).
        trace_line.set_data(trace_data_np[:, 0], trace_data_np[:, 1])
        trace_line.set_3d_properties(trace_data_np[:, 2])

        # Update axis limits to keep the vessel at the center.
        ax.set_xlim(pos[0] - margin, pos[0] + margin)
        ax.set_ylim(pos[1] - margin, pos[1] + margin)
        ax.set_zlim(pos[2] - margin, pos[2] + margin)

        # Remove previous vessel drawing if it exists.
        if vessel_poly_obj[0] is not None:
            vessel_poly_obj[0].remove()

        # Define the 8 vertices of the vessel's box (centered at the origin).
        vertices = np.array(
            [
                [-L / 2, -B / 2, -T / 2],
                [L / 2, -B / 2, -T / 2],
                [L / 2, B / 2, -T / 2],
                [-L / 2, B / 2, -T / 2],
                [-L / 2, -B / 2, T / 2],
                [L / 2, -B / 2, T / 2],
                [L / 2, B / 2, T / 2],
                [-L / 2, B / 2, T / 2],
            ]
        )

        # Rotate the vertices according to the Euler angles.
        r = R.from_euler("xyz", angles)
        rotated_vertices = r.apply(vertices)
        # Translate the vessel so that its center is at the absolute position.
        translated_vertices = rotated_vertices + pos

        # Define the 6 faces of the box using the translated vertices.
        faces = [
            [translated_vertices[j] for j in [0, 1, 2, 3]],  # bottom
            [translated_vertices[j] for j in [4, 5, 6, 7]],  # top
            [translated_vertices[j] for j in [0, 1, 5, 4]],  # side
            [translated_vertices[j] for j in [2, 3, 7, 6]],  # side
            [translated_vertices[j] for j in [1, 2, 6, 5]],  # side
            [translated_vertices[j] for j in [4, 7, 3, 0]],  # side
        ]

        # Create a new Poly3DCollection for the vessel and add it to the axis.
        vessel_poly = Poly3DCollection(
            faces, facecolors="cyan", edgecolors="black", alpha=0.5
        )
        ax.add_collection3d(vessel_poly)

        # Store the current vessel artist.
        vessel_poly_obj[0] = vessel_poly

        return trace_line, vessel_poly

    # Create the animation.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=dt * 1000, blit=False
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Vessel Animation with Moving Axis Ticks")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Generate dummy history data: 200 frames of a circular path with rotation.
    t = np.linspace(0, 2 * np.pi, 200)
    x = 5 * np.cos(t)
    y = 5 * np.sin(t)
    z = 0.5 * np.sin(2 * t)
    phi = 0.2 * np.sin(t)  # roll
    theta = 0.2 * np.cos(t)  # pitch
    psi = t  # yaw
    history = np.column_stack([x, y, z, phi, theta, psi])

    animate_history(history, dt=0.05, L=2.0, B=1.0, T=0.5)
