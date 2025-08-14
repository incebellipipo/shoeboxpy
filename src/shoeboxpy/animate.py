"""Animation utilities.

This module contains a (now optimized) routine for animating a vessel's
6-DoF (or 3-DoF position only) history. The original implementation recreated
matplotlib artists and performed Python list -> ndarray conversions every
frame, which caused quadratic time growth and excessive CPU load for small
``dt`` (high frame counts / high FPS).

Main performance improvements:
* Pre-allocate trace arrays instead of appending + copying each frame.
* Keep a single ``Poly3DCollection`` instance and update its vertices in-place
  (``set_verts``) instead of removing/adding each frame.
* Optional frame decimation via ``max_fps`` so very small ``dt`` values do not
  request more frames than a typical display can show.
* Optional static axes (``follow=False``) to avoid per-frame axis limit updates.
* Avoid ``scipy`` ``Rotation`` object construction per frame by using a tiny
  local Euler->matrix implementation (keeps dependency optional at runtime).
* Optional ``show`` flag so tests / scripts can build an animation without
  blocking on GUI.

Note: 3D blitting is still unreliable across backends, so ``blit=False``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple

__all__ = ["animate_history"]


def _euler_xyz_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return rotation matrix for XYZ (roll, pitch, yaw) intrinsic sequence.

    Equivalent to Rz(psi) * Ry(theta) * Rx(phi) when using right-handed coords.
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # Rx
    R_x = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
    # Ry
    R_y = np.array([[ctheta, 0, stheta], [0, 1, 0], [-stheta, 0, ctheta]])
    # Rz
    R_z = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def animate_history(
    pos_history: np.ndarray,
    dt: float,
    L: float = 1.0,
    B: float = 0.5,
    T: float = 0.3,
    *,
    follow: bool = True,
    max_fps: float = 60.0,
    show: bool = True,
    static_margin_factor: float = 2.0,
) -> animation.FuncAnimation:
    """Animate a 3D vessel trajectory (position + optional Euler angles).

    Parameters
    ----------
    pos_history : (N,3) or (N,6) ndarray
        Columns 0:3 are x,y,z. If 6 columns, 3:6 are Euler angles (phi,theta,psi).
    dt : float
        Simulation time step between successive ``pos_history`` rows (seconds).
    L, B, T : float
        Vessel dimensions (length, beam/width, height/thickness).
    follow : bool, default True
        If True, axes move to keep vessel centered. If False, a static bounding
        box that encloses the whole trajectory is used (faster for large N).
    max_fps : float, default 60
        Cap the displayed frames per second. If 1/dt > max_fps frames are
        requested, frames are decimated (skipping) to not exceed this.
    show : bool, default True
        Call ``plt.show()`` at the end. Set False for scripts/tests.
    static_margin_factor : float, default 2.0
        When ``follow=False`` the static axis cube extends this multiple of the
        vessel length beyond the min/max trajectory extents.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The created animation object (can be saved with ``anim.save``).
    """

    if pos_history.ndim != 2 or pos_history.shape[1] not in (3, 6):
        raise ValueError("pos_history must have shape (N,3) or (N,6)")
    if dt <= 0:
        raise ValueError("dt must be positive")

    n_total = pos_history.shape[0]

    # Frame decimation factor if dt smaller than display refresh interval target.
    desired_interval_ms = max(1000.0 / max_fps, 1.0)  # at least 1 ms
    frame_interval_ms = dt * 1000.0
    skip = int(np.ceil(desired_interval_ms / frame_interval_ms)) if frame_interval_ms < desired_interval_ms else 1
    indices = np.arange(0, n_total, skip, dtype=int)
    n_frames = len(indices)

    # Split positions & angles (angles may be zeros if not provided).
    pos = pos_history[:, :3]
    if pos_history.shape[1] == 6:
        angles = pos_history[:, 3:6]
    else:
        angles = np.zeros_like(pos)

    # Precompute static axis limits if follow disabled.
    margin_geom = L * static_margin_factor
    if not follow:
        mins = pos.min(axis=0) - margin_geom
        maxs = pos.max(axis=0) + margin_geom

    # Base vertices of vessel centered at origin.
    base_vertices = np.array(
        [
            [-L / 2, -B / 2, -T / 2],
            [L / 2, -B / 2, -T / 2],
            [L / 2, B / 2, -T / 2],
            [-L / 2, B / 2, -T / 2],
            [-L / 2, -B / 2, T / 2],
            [L / 2, -B / 2, T / 2],
            [L / 2, B / 2, T / 2],
            [-L / 2, B / 2, T / 2],
        ],
        dtype=float,
    )
    # Face index lists into the (8,3) vertex array.
    face_indices = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # side
        [2, 3, 7, 6],  # side
        [1, 2, 6, 5],  # side
        [4, 7, 3, 0],  # side
    ]

    # Pre-allocate trace arrays (avoid list append + copy each frame).
    trace_x = np.empty(n_frames)
    trace_y = np.empty(n_frames)
    trace_z = np.empty(n_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Vessel animation")

    if follow:
        # Start with a reasonable cube around first position.
        p0 = pos[0]
        ax.set_xlim(p0[0] - margin_geom, p0[0] + margin_geom)
        ax.set_ylim(p0[1] - margin_geom, p0[1] + margin_geom)
        ax.set_zlim(p0[2] - margin_geom, p0[2] + margin_geom)
    else:
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    (trace_line,) = ax.plot([], [], [], "b-", lw=1.5)

    # Create initial vessel poly (will update verts in-place each frame).
    initial_faces = [[base_vertices[j] for j in face] for face in face_indices]
    vessel_poly = Poly3DCollection(
        initial_faces, facecolors="cyan", edgecolors="black", alpha=0.5
    )
    ax.add_collection3d(vessel_poly)

    def init():  # noqa: D401
        trace_line.set_data([], [])
        trace_line.set_3d_properties([])
        return trace_line, vessel_poly

    def _frame(k: int) -> Tuple[object, object]:  # returns artists for FuncAnimation
        i = indices[k]
        p = pos[i]
        a = angles[i]

        # Update trace slices.
        trace_x[k] = p[0]
        trace_y[k] = p[1]
        trace_z[k] = p[2]
        trace_line.set_data(trace_x[: k + 1], trace_y[: k + 1])
        trace_line.set_3d_properties(trace_z[: k + 1])

        # Axis follow (only if enabled).
        if follow:
            ax.set_xlim(p[0] - margin_geom, p[0] + margin_geom)
            ax.set_ylim(p[1] - margin_geom, p[1] + margin_geom)
            ax.set_zlim(p[2] - margin_geom, p[2] + margin_geom)

        # Rotation matrix.
        Rm = _euler_xyz_matrix(a[0], a[1], a[2])
        rotated = base_vertices @ Rm.T + p  # (8,3)
        # Build face list for set_verts.
        faces = [[rotated[j] for j in face] for face in face_indices]
        vessel_poly.set_verts(faces)

        return trace_line, vessel_poly

    anim = animation.FuncAnimation(
        fig,
        _frame,
        init_func=init,
        frames=n_frames,
        interval=max(frame_interval_ms * skip, desired_interval_ms),
        blit=False,
        repeat=False,
    )

    if show:
        plt.show()

    return anim


if __name__ == "__main__":  # pragma: no cover - manual use example
    t = np.linspace(0, 20, 2000)
    x = 5 * np.cos(0.3 * t)
    y = 5 * np.sin(0.3 * t)
    z = 0.5 * np.sin(0.6 * t)
    phi = 0.2 * np.sin(0.5 * t)
    theta = 0.2 * np.cos(0.4 * t)
    psi = 0.5 * t
    hist = np.column_stack([x, y, z, phi, theta, psi])
    animate_history(hist, dt=t[1] - t[0], L=2.0, B=1.0, T=0.5)
