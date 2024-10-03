import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

class Features():
    def __init__(self) -> None:
        self.detector = None

    def create_sift(self, 
                    n_octave_layers = 5,
                    contrast_threshold:float = 0.025, #0.01
                    edge_threshold:int = 10, #30
                    sigma_:float = 0.9) -> None: #0.06
        
        # self.detector = cv2.SIFT_create(nOctaveLayers = n_octave_layers,
        #                                 contrastThreshold = contrast_threshold,
        #                                 edgeThreshold = edge_threshold,
        #                                 sigma = sigma_)
        
        self.detector = cv2.SIFT_create(nOctaveLayers = 7)
        
        # self.detector = cv2.SIFT_create()
        
    
    def detect_keypoints(self, 
                        images:list,
                        num_images:int,
                        plot:bool) -> tuple[list]:
        
        keypoints_arr = []
        descriptors_arr = []

        for img in range(num_images):
            keypoints, descriptors = self.detector.detectAndCompute(images[img], None)
            keypoints_arr.append(keypoints)
            descriptors_arr.append(descriptors)

        if plot:
            subplot_rows = math.ceil(math.sqrt(num_images))
            subplot_cols = math.ceil(num_images/subplot_rows)

            _, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(22, 10))
            ax = ax.flatten()

            for img in range(subplot_rows*subplot_cols):
                if img < num_images:
                    ax[img].imshow(cv2.drawKeypoints(images[img], keypoints_arr[img], None, 
                                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
                    ax[img].set_title(f"Image {img+1}")

                ax[img].axis("off")
        
            plt.tight_layout()
            plt.show()

        return keypoints_arr, descriptors_arr
    
    def match_features(self, 
                       image_1:np.ndarray,
                       image_2:np.ndarray,
                       keypoint_1:cv2.KeyPoint,
                       keypoint_2:cv2.KeyPoint, 
                       descriptor_1:np.ndarray,
                       descriptor_2:np.ndarray,
                       match_threshold:float,
                       plot:bool):

        """
        Match features in image 1 and image 2
        """
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptor_1, descriptor_2, k=2)
        
        good_matches = []
        src_points = []
        dst_points = []
        
        for m,n in matches:
            if m.distance < match_threshold * n.distance:
                good_matches.append([m])
                src_points.append(keypoint_1[m.queryIdx].pt)
                dst_points.append(keypoint_2[m.trainIdx].pt)
        
        src_points = np.float32(src_points).reshape(-1,1,2)
        dst_points = np.float32(dst_points).reshape(-1,1,2)

        if plot:
            plt.figure()
            matches_img = cv2.drawMatchesKnn(image_1,keypoint_1,
                                            image_2,keypoint_2,
                                            good_matches, None)
        
            plt.imshow(matches_img)
            plt.axis('off')
            plt.tight_layout()

        return src_points, dst_points, good_matches
    
    def draw_inliers(self, 
                     mask:np.ndarray,
                     matches:np.ndarray,
                     image_1:np.ndarray, 
                     image_2:np.ndarray,
                     keypoints_1:np.ndarray,
                     keypoints_2:np.ndarray):
        
        plt.figure()
        matches_img = cv2.drawMatchesKnn(image_1,keypoints_1,
                                         image_2,keypoints_2,
                                         matches, None,
                                         matchesMask=mask,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.imshow(matches_img)
        plt.axis('off')
        plt.title("Inliers after RANSAC")
        plt.tight_layout()

        
    def normalize(self,
                  x:np.ndarray,
                  x_dash:np.ndarray,
                  shape:np.ndarray):
        
        height, width, _ = shape

        t_x = t_x_dash = s_x = s_x_dash = width//2
        t_y = t_y_dash = s_y = s_y_dash = height//2
        
        x_norm = np.divide(np.subtract(x, [[[t_x, t_y]]]), [[[s_x, s_y]]])
        x_dash_norm = np.divide(np.subtract(x_dash, [[[t_x_dash, t_y_dash]]]), [[[s_x_dash, s_y_dash]]])

        # De-normalization matrix 
        T = np.array([[ 1/s_x,   0,    -t_x/s_x], 
                      [  0,   1/s_y,   -t_y/s_y], 
                      [  0,    0,        1   ]])
        
        T_dash = np.array([[ 1/s_x_dash,      0,     -t_x_dash/s_x_dash], 
                           [     0,      1/s_y_dash, -t_y_dash/s_y_dash], 
                           [     0,           0,              1        ]])
        
        return x_norm, x_dash_norm, T, T_dash