import cv2
import matplotlib.pyplot as plt
import numpy as np

class Homography:
    def __init__(self) -> None:
        self.num_points = None
        self.src_points = None
        self.des_points = None

    def calculate_homography(self, points_x:np.ndarray, points_x_dash:np.ndarray) -> np.ndarray:
        """
        Calculates the homography based on the matching points (features) using h = (A' A)^-1 (A' b)

        """
        self.src_points = points_x
        self.des_points = points_x_dash

        A = np.zeros((2*self.num_points, 8))
        b = np.zeros(self.num_points*2)

        for i in range(self.num_points):
            x, y = points_x[i]
            x_dash, y_dash = points_x_dash[i]

            A[2*i] = [x, y, 1, 0, 0, 0, -x*x_dash, -y*x_dash]
            A[2*i+1] = [0, 0, 0, x, y, 1, -x*y_dash, -y*y_dash]

            b[2*i:2*i+2] = points_x_dash[i]

        H = np.append(np.linalg.inv(A.T@A)@(A.T@b), 1).reshape(3, 3)
        return H