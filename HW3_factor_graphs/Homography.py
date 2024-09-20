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
        H, mask = cv2.findHomography(points_x, points_x_dash, cv2.RANSAC, 5.0)
        
        return H