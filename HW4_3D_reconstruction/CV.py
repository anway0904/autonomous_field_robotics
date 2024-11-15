import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

class CvHelper():
	def __init__(self):
		self.cv_container = {}
		self.pt_container = {}
		self.K = None

	def set_intrinsic_matrix(self,
						  	 fx:float,
						  	 fy:float,
							 cx:float,
							 cy:float) -> None:
		
		"""
		Set the parmeters of the intrinsic matrix of the camera.
		
		Args:
			fx (float): focal length of the camera in x
			fy (float): focal length of the camera in y
			cx (float): center of projection in x
			cy (float): center of projection in y

		Returns:
			None : 
		"""
		self.K = np.array([[fx, 0, cx],
					 	   [0, fy, cy],
						   [0,  0,  1]])
	
	def get_essential_mat(self,
					   	  F:np.ndarray):
		
		"""
		Computes the Essential matrix from the Fundamental matrix using the intrinsic camera matrix.

		The Essential matrix (E) is obtained as: E = K.T @ F @ K,
		where K is the intrinsic matrix of the camera.

		Args:
			F (numpy.ndarray): Fundamental matrix of shape [3, 3].

		Returns:
			numpy.ndarray: Essential matrix of shape [3, 3].
		"""

		E = self.K.T @ F @ self.K

		return E
	
	def get_essential_mat_cv(self,
						     src_points:np.ndarray,
							 dst_points:np.ndarray):
		"""
		Additional function to compute the essential matrix using the cv2 function.
		This function gives better results as compared to first computing the fundamental matrix 
		and then computing the essential matrix as E = K.T @ F @ K


		"""
		E, inlier_mask = cv2.findEssentialMat(src_points, dst_points, self.K)
		src_inliers = src_points[inlier_mask.ravel() != 0]
		dst_inliers = dst_points[inlier_mask.ravel() != 0]

		return E, src_inliers, dst_inliers
	
	def get_extrinsic_and_projection_matrix(self,
						   	  				E:np.ndarray,
							  				src_points:np.ndarray,
							  				dst_points:np.ndarray):
		"""
		Recovers the camera pose (extrinsic matrix) from the Essential matrix.

		This function estimates the camera's rotation and translation (extrinsic parameters)
		with respect to the world frame, given the Essential matrix. It also computes the 
		projection matrix from the recovered extrinsic matrix. Additionally, the function 
		refines the inlier source and destination points after computing the extrinsic matrix.

		Args:
			E (numpy.ndarray): Essential matrix of shape [3, 3].
			src_points (numpy.ndarray): Inlier source points of shape [N, 1, 2], obtained after computing the Fundamental matrix.
			dst_points (numpy.ndarray): Inlier destination points of shape [N, 1, 2], obtained after computing the Fundamental matrix.

		Returns:
			P (numpy.ndarray): Projection matrix of shape [3, 4].
			C (numpy.ndarray): Extrinsic matrix of shape [3, 4].
			T (numpy.ndarray): Homogeneous extrinsic matrix of shape [4, 4].
			src_inliers (numpy.ndarray): Refined inlier source points of shape [N, 1, 2].
			dst_inliers (numpy.ndarray): Refined inlier destination points of shape [N, 1, 2].
		"""

		_, R, t, inlier_mask = cv2.recoverPose(E, src_points, dst_points, self.K)
	
		P = self.K @ np.hstack((R, t))
		C = np.hstack((R, t))
		T = np.vstack((C, [0,0,0,1]))

		src_inliers = src_points[inlier_mask.ravel() != 0]
		dst_inliers = dst_points[inlier_mask.ravel() != 0]

		return P, C, T, src_inliers, dst_inliers
	
	def get_fundamental_mat(self,
						 	src_points:np.ndarray,
							dst_points:np.ndarray):
		
		"""
		
		"""
		F, inlier_mask = cv2.findFundamentalMat(src_points, dst_points, method=cv2.RANSAC,
										  ransacReprojThreshold=1.0)
		
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

			ax[1].plot([x_0, x_1], [y_0, y_1], c = color, alpha = 0.3)

			a, b, c = line_src
			x_0, y_0 = 0, -c/b
			x_1, y_1 = width, -(c + a*width)/b

			ax[0].plot([x_0, x_1], [y_0, y_1], c = color, alpha = 0.3)

			count += 1

		ax[1].scatter(dst_inliers[:min(max_lines, lines_dst.shape[0]), 0, 0], 
					  dst_inliers[:min(max_lines, lines_dst.shape[0]), 0, 1], s = 5, marker = "o", c = 'g', alpha=0.8)
		
		ax[0].scatter(src_inliers[:min(max_lines, lines_src.shape[0]), 0, 0], 
					  src_inliers[:min(max_lines, lines_src.shape[0]), 0, 1], s = 5, marker = "o", c = 'g', alpha=0.8)

		ax[1].imshow(img_dst)
		ax[1].axis("off")

		ax[0].imshow(img_src)
		ax[0].axis("off")

		plt.show()

	def update_cv_container(self, src_idx, dst_idx, F = None, E = None, P = None, C = None, T = None, PTS = None):
		self.cv_container[(src_idx, dst_idx)] = {"F": F, "E":E, "P":P, "C":C, "T":T, "PTS": PTS}

	def update_pt_container(self, src_idx, dst_idx, src_points, dst_points):
		self.pt_container[(src_idx, dst_idx)] = {"src":src_points, "dst":dst_points}
