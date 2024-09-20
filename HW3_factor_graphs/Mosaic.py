import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class Mosaic():
    def __init__(self) -> None:
        self.num_imgs_mosaic = None
        self.mosaic_imgs = None
        self.mosaic_imgs_gray = None
        self.img_shape = None
        self.T = None
        self.canvas_shape = None
        
    def read_imgs_from_folder(self, path:str, resize_factor:float = 1) -> np.ndarray:
        """
        Read images from a folder and resize them according to the resize factor.

        """
        images = []
        gray_images = []
        for image in sorted(os.listdir(path)):
            original_image = (cv2.cvtColor(cv2.imread(os.path.join(path, image)), cv2.COLOR_BGR2RGB))
            resized_image = cv2.resize(original_image, (int(original_image.shape[1]*resize_factor),
                                                        int(original_image.shape[0]*resize_factor)))
            gray_img = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

            images.append(resized_image)
            gray_images.append(gray_img)

        self.mosaic_images = images
        self.num_imgs_mosaic = len(images)
        self.mosaic_imgs_gray = gray_images

        self.img_shape = resized_image.shape
        self.mosaic_imgs_gray = gray_images

    def calculate_min_max_coordinates(self, homographies:np.ndarray) -> tuple[np.ndarray]:
        """
        Transform the corners of the image to find the maximum and the minimum limits after the transformation is applied.

        """
        img_y, img_x, _ = self.img_shape
        central_img_idx = self.num_imgs_mosaic//2

        img_corners = np.array([[0, 0], [0, img_y], [img_x, img_y], [img_x, 0]], dtype=np.float32).reshape(-1, 1, 2)

        transformed_corners_first = cv2.perspectiveTransform(np.copy(img_corners), homographies[0])
        transformed_corners_last = cv2.perspectiveTransform(np.copy(img_corners), homographies[-1])

        transformed_corners_x = np.concat((transformed_corners_first.ravel(), transformed_corners_last.ravel()))[::2]
        transformed_corners_y = np.concat((transformed_corners_first.ravel(), transformed_corners_last.ravel()))[1::2]

        min_x = min(transformed_corners_x)
        max_x = max(transformed_corners_x)
        min_y = min(transformed_corners_y)
        max_y = max(transformed_corners_y)

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
    
    def warp_images(self, H:np.ndarray, min_x:int, min_y:int, max_x:int, max_y:int) -> np.ndarray:
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

        return warped_images
    
    def transform_img_corners(self, H:np.ndarray) -> np.ndarray:
        img_y, img_x, _ = self.img_shape
        img_corners = np.array([[0, 0], [0, img_y], [img_x, img_y], [img_x, 0]], dtype=np.float32).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(np.copy(img_corners), self.T @ H)

        return transformed_corners.reshape((-1, 2))