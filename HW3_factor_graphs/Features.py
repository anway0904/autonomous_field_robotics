import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

class Features():
    def __init__(self) -> None:
        self.detector = None

    def create_sift(self, 
                    n_features:int = 1000,
                    contrast_threshold:float = 0.005,
                    edge_threshold:int = 30,
                    sigma:float = 0.6) -> None:
        
        self.detector = cv2.SIFT_create(nfeatures = n_features,
                                        contrastThreshold = contrast_threshold,
                                        edgeThreshold = edge_threshold,
                                        sigma = sigma)
        
    
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

            _, ax = plt.subplots(subplot_rows, subplot_cols)
            ax = ax.flatten()

            for img in range(num_images):
                ax[img].imshow(cv2.drawKeypoints(images[img], keypoints_arr[img], None))
                ax[img].axis("off")
                ax[img].set_title(f"Image {img+1}")
        
            plt.tight_layout()
            plt.show()

        return keypoints_arr, descriptors_arr
    
    def match_features(self, 
                       image_1:np.ndarray,
                       image_2:np.ndarray,
                       keypoint_1:cv2.KeyPoint,
                       keypoint_2:cv2.KeyPoint, 
                       descriptor_1:np.ndarray,
                       descriptor_2:np.ndarray):

        """
        Match features in image 1 and image 2
        For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used, 
        its related parameters etc. For algorithms like SIFT, the following can be used:
        """
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        plt.figure()
        matches = matcher.knnMatch(descriptor_1, descriptor_2, k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        good_matches = []
        
        # ratio test
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.85*n.distance:
                matchesMask[i]=[1,0]
                good_matches.append(m)
        
        draw_params = dict(matchColor = (255,255,100), matchesMask = matchesMask)
        matches_img = cv2.drawMatchesKnn(image_1,keypoint_1,
                                         image_2,keypoint_2,
                                         matches, None,
                                         **draw_params)
                
        plt.imshow(matches_img)
        plt.axis('off')
        plt.tight_layout()


        height_1, width_1 = image_1.shape
        height_2, width_2 = image_2.shape
        
        # Extract pixel coordinates of the matched keypoints
        points_x      = np.float32([keypoint_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        points_x_dash = np.float32([keypoint_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        # for i, match in enumerate(good_matches):
        #     img1_idx = match.queryIdx
        #     img2_idx = match.trainIdx

        #     # Get the pixel coordinates of the matched keypoints
        #     (x1, y1) = keypoint_1[img1_idx].pt  # keypoint in image1
        #     (x2, y2) = keypoint_2[img2_idx].pt  # keypoint in image2

        #     # Normalize coordinates
        #     x1_norm = x1 / width_1
        #     y1_norm = y1 / height_1
        #     x2_norm = x2 / width_2
        #     y2_norm = y2 / height_2

        #     # Store the normalized coordinates. y and x reversed to match with indexing beginning top left (as np array)
        #     points_x.append([x1_norm, y1_norm])
        #     points_x_dash.append([x2_norm, y2_norm])
            
        return points_x, points_x_dash