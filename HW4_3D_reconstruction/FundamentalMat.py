import sympy as sp
import numpy as np
import cv2
import time

class FundamentalMat():
    @staticmethod
    def normalize_points(points:np.ndarray):
        '''
        Normalizes the array points such as the centroid is at origin of the coordinates
        and RMS distance of points from the origin is sqrt(2)

        Args:
            points [N, 1, 2]: points to normalize

        Returns:
            normalized_points [N, 2]: 
            T [3, 3]                : normalization matrix
        '''
        points = points.reshape(-1, 2)

        centroid = np.mean(points, axis=0)
        translated_points = points - centroid

        rms_distance = np.sqrt(np.mean(np.sum(translated_points**2, axis=1)))
        scale_factor = np.sqrt(2) / rms_distance
        normalized_points = translated_points * scale_factor

        T = np.array([
            [scale_factor, 0, -scale_factor * centroid[0]],
            [0, scale_factor, -scale_factor * centroid[1]],
            [0, 0, 1]
        ])

        return normalized_points, T
    
    @staticmethod
    def seven_point_algorithm(src_points:np.ndarray,
                              dst_points:np.ndarray):
        """
        Randomly selects 7 correspondences from the matches

        Args:
            src_points [N, 1, 2]: source points
            dst_points [N, 1, 2]: destination points

        
        """
        num_points = src_points.shape[0]

        N = np.inf
        p = 0.99
        sample_count = 0
        max_inliers = -np.inf

        while (sample_count < N):
            indices = np.random.choice(num_points, 7, replace=False)
            src_7_points = src_points[indices].reshape(7, 2)
            dst_7_points = dst_points[indices].reshape(7, 2)

            F1, F2 = FundamentalMat.get_null_space_generators(src_7_points,dst_7_points)
            F_array = FundamentalMat.solve_for_alpha_and_F(F1, F2)

            for F in F_array:
                inlier_mask, num_inliers = FundamentalMat.evaluate_F(src_points,dst_points, F)
                
                if num_inliers > max_inliers:
                    best_F = F
                    best_inlier_mask = inlier_mask
                    max_inliers = num_inliers

            e = 1 - (num_inliers/num_points)
            N = np.log(1 - 0.99)/np.log(1-(1-e)**7)
            sample_count += 1
            # print("TIME: ", (time.perf_counter() - start_time)*1000)
        
        return best_F, best_inlier_mask, src_points[best_inlier_mask.ravel()], dst_points[best_inlier_mask.ravel()]


    @staticmethod
    def get_null_space_generators(src_points:np.ndarray,
                                  dst_points:np.ndarray):
        """
        Computes the right null-space of 7x9 matrix A of form Af = 0
        Since the matrix A is of rank 7, we get f1, f2 that form basis of this null space
        The required fundamental matrix can be written as a linear combination of f1, f2
        >>> F = (alpha)*F1 + (1-alpha)*F2 (F1, F2 are matrices of vectors f1, f2)
        
        The null space is calculated using SVD of A
        >>> A = U*D*Vt
        last two columns of Vt are vectors f1, f2

        Args:
            src_points [7, 2]: source points
            dst_points [7, 2]: destination points

        Returns:
            F1 [3, 3]: generator 1
            F2 [3, 3]: generator 2

        """
        # A = np.zeros((7, 9))
        # for i in range(7):
        #     x1, y1 = src_points[i]
        #     x2, y2 = dst_points[i]
        #     A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

        x1, y1 = src_points[:, 0], src_points[:, 1]
        x2, y2 = dst_points[:, 0], dst_points[:, 1]
        A = np.column_stack([
            x1 * x2, x1 * y2, x1,
            y1 * x2, y1 * y2, y1,
            x2, y2, np.ones_like(x1)
        ])
        _, _, V = np.linalg.svd(A)
        F1 = V[-1].reshape(3, 3)
        F2 = V[-2].reshape(3, 3)

        return F1, F2
    
    @staticmethod
    def solve_for_alpha_and_F(F1:np.ndarray,
                              F2:np.ndarray):
        """
        Once the generators of the null space of A are known, we can enforce
        >>> det(F) = 0
        >>> det(alpha*F1 + (1-alpha)*F2) = 0

        This gives rise to a cubic polynomial in alpha. Using sympy to solve it
        we can compute either 1 or 3 of Fundamental matrices

        Args:
            F1 [3, 3]: generator 1
            F2 [3, 3]: generator 2

        Returns:
            F_array [3]: A list of computed fundamental matrices
        """
        F1_ = sp.Matrix(F1)
        F2_ = sp.Matrix(F2)

        alpha = sp.symbols('alpha')

        # Define F(alpha) as a linear combination of F1 and F2
        F_alpha = (alpha * F1_) + ((1 - alpha) * F2_)

        alpha_vals_real = sp.solveset(F_alpha.det(), alpha, domain=sp.S.Reals)

        F_array = []
        for alpha_val in alpha_vals_real:
            F:np.ndarray = (alpha_val * F1) + ((1-alpha_val)*F2)
            F_array.append(F.astype(np.float32))

        return F_array
    
    @staticmethod
    def estimate_F_svd( 
					   T_src:np.ndarray,
					   T_dst:np.ndarray,
					   src_points: np.ndarray,
					   dst_points: np.ndarray):
		
        num_points = src_points.shape[0]

        A = np.zeros((num_points, 9))
        for i in range(num_points):
            x1, y1 = src_points[i]
            x2, y2 = dst_points[i]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

        _, _, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)

        # Enforce rank-2 constraint
        U_F, S_F, V_F = np.linalg.svd(F)
        S_F[-1] = 0
        F = U_F @ np.diag(S_F) @ V_F

        # Denormalize
        F = T_src.T @ F @ T_dst
            
        return F
    
    @staticmethod
    def evaluate_F(src_points:np.ndarray,
                   dst_points:np.ndarray,
                   F:np.ndarray):
        """
        Given a current estimate of F from RANSAC, this function measures
        how closely the matched pair satisfies the epipolar geometry using Sampson dist
        and returns the number of inliers from the correspondences

        Sampson dist: refer equation 11.9 on page 287 of the bible (Hartley Zisserman)

        Args:
            src_points [N, 1, 2]: source points
            dst_points [N, 1, 2]: destination points
            F [3, 3]            : fundamental matrix
        
        Returns:
            inlier_mask [N]      : Boolean array masking inliers from src and dst points
            src_inliers [M, 1, 2]: The M src points (inliers)
            dst_inliers [M, 1, 2]: The M dst points (inliers)
        """
        F_xs  = cv2.perspectiveTransform(src_points, F)
        Ft_xd = cv2.perspectiveTransform(dst_points, F.T)

        denominator = F_xs [:,:,0]**2 + F_xs [:,:,1]**2 + \
                      Ft_xd[:,:,0]**2 + Ft_xd[:,:,1]**2
        
        xd_F_xs = np.sum(np.multiply(dst_points, F_xs), axis=2) + 1

        sampson_dist = np.divide(xd_F_xs**2, denominator)
        
        inlier_mask = (np.sqrt(sampson_dist) <= 1.5)
        num_inliers = np.count_nonzero(inlier_mask)
        
        return inlier_mask, num_inliers
