import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, permutations

def non_temporal_matching(mosaic :object,
                          features:object,
                          homography:classmethod,
                          keypoints_list:np.ndarray,
                          descriptor_list:np.ndarray,
                          pivot:int,
                          match_thresh:float,
                          ransac_thresh:float,
                          inlier_thresh:int,
                          plot:bool) -> dict:
    
    imgs_per_pivot = pivot
    matches_dict = {} # {(src_idx, des_idx) : {H: [], error: }}

    for ref_img_start_idx in range(0, mosaic.num_imgs_mosaic, 2*imgs_per_pivot):
        
        low_iterator_start = ref_img_start_idx - 1 
        high_iterator_start = ref_img_start_idx + imgs_per_pivot

        for src_img_idx in range(ref_img_start_idx, ref_img_start_idx+imgs_per_pivot, 1):    
            for des_img_idx in range(low_iterator_start, low_iterator_start-imgs_per_pivot, -1):
                if des_img_idx < 0:
                    break
                
                if abs(src_img_idx - des_img_idx) == 1:
                    continue

                src_points, dst_points, _ = features.match_features(mosaic.mosaic_imgs_gray[src_img_idx], 
                                                                    mosaic.mosaic_imgs_gray[des_img_idx],
                                                                    keypoints_list[src_img_idx],
                                                                    keypoints_list[des_img_idx],
                                                                    descriptor_list[src_img_idx],
                                                                    descriptor_list[des_img_idx],
                                                                    match_threshold=match_thresh,
                                                                    plot=False)
        
                H, inliers = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransac_thresh)

                
                good, covar = homography.is_good(H, inliers, inlier_thresh, src_points, dst_points)
                if not good:
                    print(f"{src_img_idx} ---> {des_img_idx}: No match!")
                    continue
                
                if plot:
                    plt.figure()
                    w = cv2.warpPerspective(mosaic.mosaic_imgs_gray[src_img_idx], H, (mosaic.img_shape[1], mosaic.img_shape[0]))
                    plt.imshow(cv2.addWeighted(mosaic.mosaic_imgs_gray[des_img_idx], 0.5, w, 0.5, 10))
                    plt.title(f"src = {src_img_idx}  dst = {des_img_idx}")       

                print(f"{src_img_idx} ---> {des_img_idx}: Match! ({covar = })")
                matches_dict[(src_img_idx, des_img_idx)] = {"H": H, "error": covar}


            for des_img_idx in range(high_iterator_start, high_iterator_start+imgs_per_pivot, 1):
                if des_img_idx > (mosaic.num_imgs_mosaic - 1):
                    break
                
                if abs(src_img_idx - des_img_idx) == 1:
                    continue

                src_points, dst_points, _ = features.match_features(mosaic.mosaic_imgs_gray[src_img_idx], 
                                                                    mosaic.mosaic_imgs_gray[des_img_idx],
                                                                    keypoints_list[src_img_idx],
                                                                    keypoints_list[des_img_idx],
                                                                    descriptor_list[src_img_idx],
                                                                    descriptor_list[des_img_idx],
                                                                    match_threshold=match_thresh,
                                                                    plot=False)
        
                H, inliers = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransac_thresh)

                
                good, covar = homography.is_good(H, inliers, inlier_thresh, src_points, dst_points)
                if not good:
                    print(f"{src_img_idx} ---> {des_img_idx}: No match! ({np.count_nonzero(inliers)})")
                    continue
                
                if plot:
                    plt.figure()
                    w = cv2.warpPerspective(mosaic.mosaic_imgs_gray[src_img_idx], H, (mosaic.img_shape[1], mosaic.img_shape[0]))
                    plt.imshow(cv2.addWeighted(mosaic.mosaic_imgs_gray[des_img_idx], 0.5, w, 0.5, 10))
                    plt.title(f"src = {src_img_idx}  dst = {des_img_idx}")       

                print(f"{src_img_idx} ---> {des_img_idx}: Match! ({covar = })")
                matches_dict[(src_img_idx, des_img_idx)] = {"H": H, "error": covar}

    return matches_dict

def non_temporal_matching_bf(mosaic :object,
                             features:object,
                             homography:classmethod,
                             keypoints_list:np.ndarray,
                             descriptor_list:np.ndarray,
                             pivot:int,
                             match_thresh:float,
                             ransac_thresh:float,
                             inlier_thresh:int,
                             plot:bool) -> dict:

    matches_dict = {}
   
    for src_img_idx, des_img_idx in list(combinations(range(mosaic.num_imgs_mosaic), 2)):
        if abs(src_img_idx - des_img_idx) == 1:
            continue
        
        if (des_img_idx, src_img_idx) in matches_dict.keys():
            continue

        src_points, dst_points, _ = features.match_features(mosaic.mosaic_imgs_gray[src_img_idx], 
                                                            mosaic.mosaic_imgs_gray[des_img_idx],
                                                            keypoints_list[src_img_idx],
                                                            keypoints_list[des_img_idx],
                                                            descriptor_list[src_img_idx],
                                                            descriptor_list[des_img_idx],
                                                            match_threshold=match_thresh,
                                                            plot=False)
        
        H, inliers = cv2.estimateAffinePartial2D(src_points, dst_points,None,cv2.RANSAC, ransac_thresh)
        H = np.append(H, [[0, 0, 1]], axis=0)
        good, covar = homography.is_good(H, inliers, inlier_thresh, src_points, dst_points)

        if not good:
            # Try reversing the matching (des --> src)
            H, inliers = cv2.estimateAffinePartial2D(dst_points, src_points, None, cv2.RANSAC, ransac_thresh)
            H = np.append(H, [[0, 0, 1]], axis=0)
            good, covar = homography.is_good(H, inliers, inlier_thresh, src_points, dst_points)

            if not good:
                print(f"{src_img_idx} ---> {des_img_idx}: No match! ({np.count_nonzero(inliers)})")
                continue
        
        if plot:
            plt.figure()
            w = cv2.warpPerspective(mosaic.mosaic_imgs_gray[src_img_idx], H, (mosaic.img_shape[1], mosaic.img_shape[0]))
            plt.imshow(cv2.addWeighted(mosaic.mosaic_imgs_gray[des_img_idx], 0.5, w, 0.5, 10))
            plt.title(f"src = {src_img_idx}  dst = {des_img_idx}")       

        print(f"{src_img_idx} ---> {des_img_idx}: Match! ({covar = })")
        matches_dict[(src_img_idx, des_img_idx)] = {"H": H, "error": covar}

    return matches_dict