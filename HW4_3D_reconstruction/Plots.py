import matplotlib.pyplot as plt
import numpy as np

class Plots:
    def __init__(self, figsize=(10, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        # self.ax.set_xlim([-10, 10])
        # self.ax.set_ylim([-10, 10])
        # self.ax.set_zlim([-10, 10])
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        # self.ax.set_aspect('equal')

    @staticmethod
    def validate_inputs(R: np.ndarray, t: np.ndarray):
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must be of shape (3, 3).")
        if t.shape not in [(3,), (3, 1)]:
            raise ValueError("Translation vector t must be of shape (3,) or (3, 1).")

    def plot_camera(self, R: np.ndarray, t: np.ndarray, scale: float = 1, label: str = "") -> None:
        Plots.validate_inputs(R, t)
        t = t.reshape(-1)  # Ensure it's a flat array
        camera_center = R.T @ -t

        # Axes in the camera frame (local frame)
        x_axis = R.T @ np.array([1, 0, 0]) * scale
        y_axis = R.T @ np.array([0, 1, 0]) * scale
        z_axis = R.T @ np.array([0, 0, 1]) * scale

        # Plot the camera origin
        self.ax.scatter(*camera_center, c='r', marker='o')
        self.ax.text(*camera_center + 0.1, label, color='blue')

        # Plot the axes
        self.ax.quiver(*camera_center, *x_axis, color='r', arrow_length_ratio=0.1)
        self.ax.quiver(*camera_center, *y_axis, color='g', arrow_length_ratio=0.1)
        self.ax.quiver(*camera_center, *z_axis, color='b', arrow_length_ratio=0.1)

    def plot_points(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points array must have shape (N, 3).")
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=5)

    def set_axes_equal(self):
        limits = np.array([self.ax.get_xlim3d(),
                           self.ax.get_ylim3d(),
                           self.ax.get_zlim3d(),])

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

        self.ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        self.ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        self.ax.set_zlim3d([origin[2] - radius, origin[2] + radius])