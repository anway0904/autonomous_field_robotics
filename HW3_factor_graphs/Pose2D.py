import numpy as np

class Pose2D():
    @staticmethod
    def get_pose_from_homography(H:np.ndarray):
        U, _, Vt = np.linalg.svd(H)
        rotation_mat = U @ np.diag([1, 1, np.linalg.det(U)]) @ Vt

        tx = H[0,2]
        ty = H[1,2]

        theta_rad = np.arctan2(rotation_mat[1,1], rotation_mat[2,1])

        return (tx, ty, theta_rad)
    
    @staticmethod
    def get_initial_pose_estimates(H_array:np.ndarray):
        poses = []
        x = y = 0

        for H in H_array:
            dx, dy, theta = Pose2D.get_pose_from_homography(H)
            x += dx
            y += dy
            poses.append((x, y, theta))

        return poses