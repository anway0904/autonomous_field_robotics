import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class CvHelper():
	def __init__(self):
		self.cv_container = {}
		self.pt_container = {}
		self.K = None

	def set_intrinsic_matrix(self,
						  	 fx:float,
						  	 fy:float,
							 cx:float,
							 cy:float):
		
		self.K = np.array([[fx, 0, cx],
					 	   [0, fy, cy],
						   [0,  0,  1]])
	
	def get_essential_mat(self,
					   	  F:np.ndarray):
		return self.K.T @ F @ self.K
	
	def get_projection_matrix(self,
						   	  E:np.ndarray,
							  src_points:np.ndarray,
							  dst_points:np.ndarray):
		
		_, R, t, inlier_mask = cv2.recoverPose(E, src_points, dst_points, self.K)
	
		P = self.K @ np.hstack((R, t))

		src_inliers = src_points[inlier_mask.ravel() != 0]
		dst_inliers = dst_points[inlier_mask.ravel() != 0]

		return P, src_inliers, dst_inliers
	
	def normalize_points(self,
					  	 points:np.ndarray, 
					     img_shape:np.ndarray):
		# height, width = img_shape

		# tx = sx = width//2
		# ty = sy = height//2

		
		# T = np.array([[ 1/sx,   0,    -tx/sx], 
        #       		  [  0,   1/sy,   -ty/sy], 
        #       		  [  0,    0,        1  ]])
		
		# points_norm = cv2.perspectiveTransform(points, T).reshape(-1,2)
		points = points.reshape(-1, 2)

		# Step 1: Compute the centroid of the points
		centroid = np.mean(points, axis=0)

		# Step 2: Translate points so that the centroid is at the origin
		translated_points = points - centroid

		# Step 3: Compute the RMS distance of the points from the origin
		rms_distance = np.sqrt(np.mean(np.sum(translated_points**2, axis=1)))

		# Step 4: Scale points so that RMS distance is sqrt(2)
		scale_factor = np.sqrt(2) / rms_distance
		normalized_points = translated_points * scale_factor

		# Step 5: Construct the normalization matrix
		T = np.array([
			[scale_factor, 0, -scale_factor * centroid[0]],
			[0, scale_factor, -scale_factor * centroid[1]],
			[0, 0, 1]
		])
		return normalized_points, T

	def estimate_F_svd(self, 
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

	def get_fundamental_mat_8_pt_norm(self,
										img_shape:np.ndarray,
										src_points:np.ndarray,
										dst_points:np.ndarray):
					
		src_points_norm, T_src = self.normalize_points(src_points, img_shape)
		dst_points_norm, T_dst = self.normalize_points(dst_points, img_shape)

		F = self.estimate_F_svd(T_src, T_dst, src_points_norm, dst_points_norm)
		
		return F, None, None, None
	
	def get_fundamental_mat_8_pt_ransac(self,
								   img_shape:np.ndarray,
								   src_points:np.ndarray,
								   dst_points:np.ndarray,
								   ransac_iter: int = 1000,
								   ransac_thresh: float = 1.0):
		
		num_points = src_points.shape[0]
		
		best_inliers = 0
		best_inlier_mask = None

		for _ in range(ransac_iter):
			random_8_idx = np.random.randint(0, num_points, 8)
			src_8_points = src_points[random_8_idx] # points_src
			dst_8_points = dst_points[random_8_idx] # points_dst
			
			src_8_points_norm, T_src = self.normalize_points(src_8_points, img_shape)
			dst_8_points_norm, T_dst = self.normalize_points(dst_8_points, img_shape)

			F = self.estimate_F_svd(T_src, T_dst, src_8_points_norm, dst_8_points_norm)
			
			distances = self.compute_sampson_distance(F, src_points, dst_points)
			inlier_mask = distances < ransac_thresh
			num_inliers = np.sum(inlier_mask)

			if num_inliers > best_inliers:
				best_inliers = num_inliers
				best_inlier_mask = inlier_mask
				print(f"Found better solution with {num_inliers}/{num_points} inliers")

		# Final refinement using all inliers
		src_inliers = src_points[best_inlier_mask]
		dst_inliers = dst_points[best_inlier_mask]

		src_inliers_norm, T_src = self.normalize_points(src_inliers, img_shape)
		dst_inliers_norm, T_dst = self.normalize_points(dst_inliers, img_shape)
		F_final = self.estimate_F_svd(T_src, T_dst, src_inliers_norm, dst_inliers_norm)

		return F_final, best_inlier_mask, src_inliers, dst_inliers
	
	def get_fundamental_mat(self,
						 	src_points:np.ndarray,
							dst_points:np.ndarray):
		F, inlier_mask = cv2.findFundamentalMat(src_points, dst_points, method=cv2.RANSAC)
		
		src_inliers = src_points[inlier_mask.ravel() == 1]
		dst_inliers = dst_points[inlier_mask.ravel() == 1]

		return F, inlier_mask, src_inliers, dst_inliers
	
	def draw_epipolar_lines(self,
							src_inliers:np.ndarray,
							dst_inliers:np.ndarray,
							F:np.ndarray,
							img_src:np.ndarray,
							img_dst:np.ndarray,
							max_lines:int):
		
		_, ax = plt.subplots(1, 2, figsize = (10, 20))

		_, width, _ = img_src.shape
		lines_dst = cv2.computeCorrespondEpilines(src_inliers, 1, F).reshape(-1, 3)
		lines_src = cv2.computeCorrespondEpilines(dst_inliers, 2, F).reshape(-1, 3)

		count = 0
		for _, (line_src, line_dst) in enumerate(zip(lines_src, lines_dst)):
			if count >= max_lines:
				break
			color = tuple(np.random.rand(1,3).ravel())

			a, b, c = line_dst
			x_0, y_0 = 0, -c/b
			x_1, y_1 = width, -(c + a*width)/b

			ax[1].plot([x_0, x_1], [y_0, y_1], c = color, alpha = 0.4)

			a, b, c = line_src
			x_0, y_0 = 0, -c/b
			x_1, y_1 = width, -(c + a*width)/b

			ax[0].plot([x_0, x_1], [y_0, y_1], c = color, alpha = 0.4)

			count += 1

		ax[1].scatter(dst_inliers[:min(max_lines, lines_dst.shape[0]), 0, 0], 
					  dst_inliers[:min(max_lines, lines_dst.shape[0]), 0, 1], marker = "o", c = 'r', alpha=0.8)
		
		ax[0].scatter(src_inliers[:min(max_lines, lines_src.shape[0]), 0, 0], 
					  src_inliers[:min(max_lines, lines_src.shape[0]), 0, 1], marker = "o", c = 'r', alpha=0.8)

		ax[1].imshow(img_dst)
		ax[1].axis("off")

		ax[0].imshow(img_src)
		ax[0].axis("off")

		plt.show()

	def update_cv_container(self, src_idx, dst_idx, F = None, E = None, P = None):
		self.cv_container[(src_idx, dst_idx)] = {"F": F, "E":E, "P":P}

	def update_pt_container(self, src_idx, dst_idx, src_points, dst_points):
		self.pt_container[(src_idx, dst_idx)] = {"src":src_points, "dst":dst_points}
