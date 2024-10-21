import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

class SIFT():
    def __init__(self) -> None:
        self.detector = None

    def create_detector(self, 
                        n_octave_layers = 5,
                        contrast_threshold:float = 0.2,
                        edge_threshold:int = None,
                        sigma_:float = None) -> None:
        
        # self.detector = cv2.SIFT_create(nOctaveLayers = n_octave_layers,
        #                                 contrastThreshold = contrast_threshold,
        #                                 edgeThreshold = edge_threshold,
        #                                 sigma = sigma_)
        
        self.detector = cv2.SIFT_create()
        
    
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
                    # ax[img].set_title(f"Image {img+1}")

                ax[img].axis("off")
        
            plt.tight_layout()
            plt.show()

        return keypoints_arr, descriptors_arr
    
    def normalize_keypoints(self,
                            src_points:np.ndarray,
                            dst_points:np.ndarray,
                            shape:np.ndarray):
        
        height, width = shape

        t_x = t_x_dash = s_x = s_x_dash = width//2
        t_y = t_y_dash = s_y = s_y_dash = height//2
        
        src_norm = np.divide(np.subtract(src_points, [[[t_x, t_y]]]), [[[s_x, s_y]]])
        dst_norm = np.divide(np.subtract(dst_points, [[[t_x_dash, t_y_dash]]]), [[[s_x_dash, s_y_dash]]])

        # De-normalization matrix 
        T = np.array([[ 1/s_x,   0,    -t_x/s_x], 
                      [  0,   1/s_y,   -t_y/s_y], 
                      [  0,    0,        1   ]])
        
        return src_norm, dst_norm, T