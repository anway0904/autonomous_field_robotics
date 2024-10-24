import cv2
import numpy as np

class Homography:
    @staticmethod
    def is_good(H:np.ndarray, 
                inliers:np.ndarray,
                inlier_thresh:int,
                src_points:np.ndarray, 
                dst_points:np.ndarray) -> tuple[bool, float]:
        
        num_inliers = np.count_nonzero(inliers)

        if num_inliers < inlier_thresh:
                return False, None
        
        inliers_idx = np.where(inliers == 1)
        src = src_points[inliers_idx[0]]
        dst = dst_points[inliers_idx[0]]

        src_transformed = cv2.perspectiveTransform(src, H)
        errors = np.linalg.norm(src_transformed.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)

        if np.mean(errors) > 5:
            return False, None
             
        return True, 1000*np.mean(errors)/(num_inliers**2)