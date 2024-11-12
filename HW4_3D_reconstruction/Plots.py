import numpy as np
import matplotlib.pyplot as plt

class Plots():

    @staticmethod
    def plot_camera(ax, R:np.ndarray, t:np.ndarray, scale:float = 1, label:str = "") -> None:
        """
        Plots the 3D coordinate frame of a camera described by rotation R and translation t
        The rotation and translation describe the camera coordinates in reference frame at origin
        Hence the camera center would be the inverse of R, t (we need to go from the camera frame to ref frame)

        Args:
            ax      : plt axes to plot the coordinate frame on
            R [3, 3]: 3D rotation matrix
            t [3, 1]: 3D translation vector
            scale   : scale factor for the quivers
            label   : label for the coordinate axis drawn

        Returns:
            None
        """
        camera_center = R.T @ -t
        
        # Axes in the camera frame (local frame)
        x_axis = R.T @ np.array([1, 0, 0]) * scale
        y_axis = R.T @ np.array([0, 1, 0]) * scale
        z_axis = R.T @ np.array([0, 0, 1]) * scale
        
        # Convert arrays to dtype=object to avoid the deprecation warning
        camera_center = np.array(camera_center).reshape(-1)  # Ensure it's a flat array
        x_axis = np.array(x_axis).reshape(-1)
        y_axis = np.array(y_axis).reshape(-1)
        z_axis = np.array(z_axis).reshape(-1)
        
        # Plot the camera origin
        ax.scatter(*camera_center, c='r', marker='o')
        ax.text(*camera_center, label, color='blue')
        
        # Plot the axes
        ax.quiver(*camera_center, *x_axis, color='r', arrow_length_ratio=0.1)
        ax.quiver(*camera_center, *y_axis, color='g', arrow_length_ratio=0.1)
        ax.quiver(*camera_center, *z_axis, color='b', arrow_length_ratio=0.1)

        # Set plot limits and labels
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')