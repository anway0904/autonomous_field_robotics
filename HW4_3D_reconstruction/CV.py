import cv2
import numpy as np
import matplotlib.pyplot as plt

class CvHelper():
	@staticmethod
	def get_fundamental_mat(src_points:np.ndarray,
							dst_points:np.ndarray):
		F, inlier_mask = cv2.findFundamentalMat(src_points, dst_points, method=cv2.RANSAC)
		src_inliers = src_points[inlier_mask.ravel() == 1]
		dst_inliers = dst_points[inlier_mask.ravel() == 1]

		return F, inlier_mask, src_inliers, dst_inliers
	 
	@staticmethod
	def draw_epipolar_lines(src_inliers:np.ndarray,
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
		for idx, (line_src, line_dst) in enumerate(zip(lines_src, lines_dst)):
			if count >= max_lines:
				break
			color = tuple(np.random.rand(1,3).ravel())

			a, b, c = line_dst
			x_0, y_0 = 0, -c/b
			x_1, y_1 = width, -(c + a*width)/b

			ax[1].plot([x_0, x_1], [y_0, y_1], c = color, alpha = 1)

			a, b, c = line_src
			x_0, y_0 = 0, -c/b
			x_1, y_1 = width, -(c + a*width)/b

			ax[0].plot([x_0, x_1], [y_0, y_1], c = color, alpha = 1)

			count += 1

			ax[1].scatter(dst_inliers[idx,0,0], dst_inliers[idx,0,1], marker = "o", c = 'r', alpha=0.8)
			ax[0].scatter(src_inliers[idx,0,0], src_inliers[idx,0,1], marker = "o", c = 'r', alpha=0.8)

		ax[1].imshow(img_dst)
		ax[1].axis("off")

		ax[0].imshow(img_src)
		ax[0].axis("off")

		plt.show()