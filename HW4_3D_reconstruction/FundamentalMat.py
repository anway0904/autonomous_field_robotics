import sympy as sp
import numpy as np

class FundamentalMat():
    @staticmethod
    def normalize_points(points:np.ndarray):
        """
        Normalizes the array points such as the centroid is at origin of the coordinates
        and RMS distance of points from the origin is sqrt(2)

        Args:
            points [N, 1, 2]: points to normalize

        Returns:
            normalized_points [N, 2]: 
            T [3, 3]                : normalization matrix
        """
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
        indices = np.random.randint(0, num_points, 7)
        src_7_points = src_points[indices].reshape(7, 2)
        dst_7_points = dst_points[indices].reshape(7, 2)

        F1, F2 = FundamentalMat.get_null_space_generators(src_7_points,
                                                          dst_7_points)
        
        return(FundamentalMat.solve_cubic_polynomial(F1, F2))

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
        A = np.zeros((7, 9))
        for i in range(7):
            x1, y1 = src_points[i]
            x2, y2 = dst_points[i]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

        _, _, V = np.linalg.svd(A)
        F1 = V[-1].reshape(3, 3)
        F2 = V[-2].reshape(3, 3)

        return F1, F2
    
    @staticmethod
    def solve_cubic_polynomial(F1:np.ndarray,
                               F2:np.ndarray):
        """
        Once the generators of the null space of A are known, we can enforce
        >>> det(F) = 0
        >>> det(alpha*F1 + (1-alpha)*F2) = 0

        This gives rise to a cubic polynomial in alpha. Using sympy to solve it

        Args:
            F1 [3, 3]: generator 1
            F2 [3, 3]: generator 2

        Returns:
            alpha [3]: roots of polynomial
        """
        F1_ = sp.Matrix(F1)
        F2_ = sp.Matrix(F2)

        alpha = sp.symbols('alpha')

        # Define F(alpha) as a linear combination of F1 and F2
        F_alpha = (alpha * F1_) + ((1 - alpha) * F2_)

        alpha_vals_real = sp.solveset(F_alpha.det(), alpha, domain=sp.S.Reals)

        for alpha_val in alpha_vals_real:
            F = (alpha_val * F1) + ((1-alpha_val)*F2)

        return F.astype(np.float32)

