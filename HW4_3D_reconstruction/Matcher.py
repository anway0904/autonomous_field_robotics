import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

class Matcher():
    def __init__(self, 
                 imgs:list[np.ndarray], 
                 keypoints:list[cv2.KeyPoint],
                 descriptors:list[np.ndarray]):
        
        self.imgs = imgs
        self.keypoints = keypoints
        self.descriptors = descriptors
        
        self.MATCH_THRESH = 0.8

    def match_features(self,
                       src_idx:int,
                       dst_idx:int,
                       plot:bool):

        """
        Match features in image 1 and image 2

        """
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(self.descriptors[src_idx], self.descriptors[dst_idx], k = 2)
        
        good_matches = []
        src_points = []
        dst_points = []
        
        for m,n in matches:
            if m.distance < self.MATCH_THRESH * n.distance:
                good_matches.append([m])
                src_points.append(self.keypoints[src_idx][m.queryIdx].pt)
                dst_points.append(self.keypoints[dst_idx][m.trainIdx].pt)
        
        src_points = np.float32(src_points).reshape(-1,1,2)
        dst_points = np.float32(dst_points).reshape(-1,1,2)

        if plot:
            plt.figure(figsize=(8,8))
            matches_img = cv2.drawMatchesKnn(self.imgs[src_idx],self.keypoints[src_idx],
                                             self.imgs[dst_idx],self.keypoints[dst_idx],
                                             good_matches, None)
        
            plt.imshow(matches_img)
            plt.axis('off')
            plt.tight_layout()

        return src_points, dst_points, good_matches
    
    def draw_inliers(self, 
                     mask:np.ndarray,
                     matches:np.ndarray,
                     src_idx:int,
                     dst_idx:int):
        
        plt.figure(figsize=(8,8))
        matches_img = cv2.drawMatchesKnn(self.imgs[src_idx],self.keypoints[src_idx],
                                         self.imgs[dst_idx],self.keypoints[dst_idx],
                                         matches, None,
                                         matchesMask=mask,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.imshow(matches_img)
        plt.axis('off')
        plt.title("Inliers after RANSAC")
        plt.tight_layout()

    