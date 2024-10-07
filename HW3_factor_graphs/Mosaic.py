import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math

class Mosaic():
    def __init__(self) -> None:
        self.num_imgs_mosaic = None
        self.mosaic_imgs = None
        self.mosaic_imgs_gray = None
        self.clahe_images = None
        self.img_shape = None
        self.T = None
        self.canvas_shape = None
        
    def read_imgs_from_folder(self, path:str, resize_factor:float, show_imgs:bool, reverse:bool = False) -> np.ndarray:
        """
        Read images from a folder and resize them according to the resize factor.

        """
        images = []
        gray_images = []
        for image in sorted(os.listdir(path), reverse=reverse):
            original_image = (cv2.cvtColor(cv2.imread(os.path.join(path, image)), cv2.COLOR_BGR2RGB))
            resized_image = cv2.resize(original_image, (int(original_image.shape[1]*resize_factor),
                                                        int(original_image.shape[0]*resize_factor)))
            
            gray_img = cv2.normalize(cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY), 
                                     None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(resized_image)
            gray_images.append(gray_img)

        self.mosaic_images = images
        self.num_imgs_mosaic = len(images)
        self.mosaic_imgs_gray = gray_images

        self.img_shape = resized_image.shape
        self.mosaic_imgs_gray = gray_images

        if show_imgs:
            subplot_rows = math.ceil(math.sqrt(self.num_imgs_mosaic))
            subplot_cols = math.ceil(self.num_imgs_mosaic/subplot_rows)

            _, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(22, 10))
            ax = ax.flatten()
            for img in range(subplot_rows*subplot_cols):
                if img < self.num_imgs_mosaic:
                    ax[img].imshow(self.mosaic_imgs_gray[img], cmap="gray")
                    ax[img].set_title(f"Image {img+1}")
                
                ax[img].axis("off")
            
            plt.tight_layout()
            plt.show()

    def apply_clahe(self):
        clahe_imgs = []
        clahe = cv2.createCLAHE(clipLimit=3)
        for img in self.mosaic_imgs_gray:
            clahe_imgs.append(clahe.apply(img))

        self.clahe_images = clahe_imgs


    def apply_radiometric_correction(self):
        f = np.array(self.mosaic_imgs_gray)
        f_dash_xy = np.mean(f, axis=0)
        f_std = np.std(f, axis=0)

        min_1 = 1/f_dash_xy
        min_2 = 255/(f_dash_xy + (4 * f_std))

        R = np.mean(f)*np.minimum(min_1, min_2)

        self.radio_corrected_imgs = np.multiply(self.mosaic_imgs_gray, R).astype(np.uint8)


    def calculate_min_max_coordinates(self, homographies:np.ndarray) -> tuple[np.ndarray]:
        """
        Transform the corners of the image to find the maximum and the minimum limits after the transformation is applied.

        """
        img_y, img_x, _ = self.img_shape

        img_corners = np.array([[0, 0], [0, img_y], [img_x, img_y], [img_x, 0]], dtype=np.float32).reshape(-1, 1, 2)
        
        min_x = min_y = np.inf
        max_x = max_y = -np.inf

        for i, H in enumerate(homographies):
            transformed_corners = cv2.perspectiveTransform(np.copy(img_corners), H)

            transformed_corners_x = transformed_corners.ravel()[::2]
            transformed_corners_y = transformed_corners.ravel()[1::2]

            min_x = min(min_x, *transformed_corners_x)
            max_x = max(max_x, *transformed_corners_x)
            min_y = min(min_y, *transformed_corners_y)
            max_y = max(max_y, *transformed_corners_y)

        return min_x, min_y, max_x, max_y 
    
    def get_homography_wrt_center(self, central_img_idx:int, homographies:np.ndarray):
        """
        Transform the homographies with respect to the central image for a better result

        """
        homographies_wrt_center = []
        for i in range(self.num_imgs_mosaic):
            H = np.identity(3)        
            if i < central_img_idx:
                for h_idx in range(i, central_img_idx):
                    H @= homographies[h_idx]

            elif i > central_img_idx:
                for h_idx in range(i-1, central_img_idx-1, -1):
                    H @= np.linalg.inv(homographies[h_idx])

            homographies_wrt_center.append(H)

        return homographies_wrt_center
    
    def warp_images(self, H:np.ndarray, min_x:int, min_y:int, max_x:int, max_y:int, plot:bool) -> np.ndarray:
        """
        Warp all the images of the mosaic on a large canvas 
        
        """
        # Translation matrix to fit the panaroma in the canvas
        self.T = np.array([[1, 0, abs(min_x)],
                           [0, 1, abs(min_y)],
                           [0, 0,       1   ]])

        warped_images = []
        self.canvas_shape = (int(max_x-min_x), int(max_y-min_y))

        for i in range(self.num_imgs_mosaic):
            warped_img = cv2.warpPerspective(np.copy(self.mosaic_images[i]), self.T @ H[i], self.canvas_shape)
            warped_images.append(warped_img)

        if plot: 
            for i in range(self.num_imgs_mosaic):
                plt.figure()
                plt.imshow(warped_images[i], cmap="gray")
                plt.axis("off")

        return warped_images
    
    def transform_img_corners(self, H:np.ndarray) -> np.ndarray:
        img_y, img_x, _ = self.img_shape
        img_corners = np.array([[0, 0], [0, img_y], [img_x, img_y], [img_x, 0]], dtype=np.float32).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(img_corners, self.T @ H)

        return transformed_corners.reshape((-1, 2))
    
    def show_panorama(self, 
                      warped_images:list, 
                      homographies:np.ndarray,
                      fig_size=(12, 10)):
        
        panorama = np.zeros_like(warped_images[0])
        canvas_mask = np.copy(panorama[:,:,0]).astype(np.uint8)
        alpha = 0.
        gamma = 20

        plt.figure(figsize=fig_size)

        for i in range(len(warped_images)):
            transformed_corners = self.transform_img_corners(homographies[i])
            warp_mask = np.zeros_like(warped_images[0][:,:,0], dtype=np.uint8)
            cv2.fillPoly(warp_mask, [transformed_corners.astype(int)], 255)

            overlap_mask = cv2.bitwise_and(warp_mask, canvas_mask)
            not_overlap_mask = cv2.bitwise_not(overlap_mask)
            overlap_idx = np.where(overlap_mask>0)
            
            panorama[overlap_idx] = (alpha)*panorama[overlap_idx] + ((1-alpha)*warped_images[i][overlap_idx]) + gamma
            
            panorama = cv2.bitwise_or(panorama, 
                                    cv2.bitwise_and(warped_images[i],
                                                    np.repeat(not_overlap_mask[:,:,np.newaxis], 3, axis=2)))
            canvas_mask = cv2.bitwise_or(canvas_mask, warp_mask)

        plt.imshow(panorama)
        plt.axis('off')
        plt.show()
