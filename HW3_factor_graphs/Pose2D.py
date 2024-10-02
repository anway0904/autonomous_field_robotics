import numpy as np

class Pose2D():
    @staticmethod
    def get_pose_from_homography(H:np.ndarray):
        
        H /= H[2,2]

        tx = H[0,2]
        ty = H[1,2]
        theta_rad = np.arctan2(H[1,0], H[0,0])

        return (tx, ty, theta_rad)
    
    @staticmethod
    def get_initial_pose_estimates(H_array:np.ndarray):
        poses = []
        x = y = theta = 0

        for H in H_array:
            dx, dy, dtheta = Pose2D.get_pose_from_homography(H)
            x += dx
            y += dy
            theta += dtheta
            poses.append((x, y, theta))

        return poses
    
    @staticmethod
    def get_trajectory():
        pass