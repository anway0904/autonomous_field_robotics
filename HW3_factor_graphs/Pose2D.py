import numpy as np

class Pose2D():
    @staticmethod
    def get_pose_from_homography(H:np.ndarray,
                                 T:np.ndarray):
        # Perform SVD on H
        U, S, Vt = np.linalg.svd(H)

        # Construct the diagonal matrix A
        A = np.diag([S[0], S[1], 1])

        # Solve for translation
        translation_mat = np.linalg.solve(A, U[:, 2])

        # Calculate the rotation matrix
        rotation_mat = U @ np.diag([1, 1, np.linalg.det(U)]) @ Vt
        
        # Normalize to homogeneous co-ordinates
        translation_mat/=translation_mat[-1]

        # Divide by the scaling factor to de-normalize translation
        tx = translation_mat[0]/T[0,0]
        ty = translation_mat[1]/T[1,1]
        theta_rad = np.atan2(rotation_mat[1,1], rotation_mat[2,1])

        return (tx, ty, theta_rad)