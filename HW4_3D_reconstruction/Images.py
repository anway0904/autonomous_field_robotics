import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math

class Images():
    def __init__(self):
        self.num_imgs  = None
        self.rgb  = None
        self.gray = None
        self.shape = None
    
    def read_imgs_from_folder(self, path:str, resize_factor:float, show_imgs:bool, reverse:bool = False) -> np.ndarray:
        """
        Read images from a folder, convert to grayscale and resize them according to the resize factor.
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

        self.rgb = images
        self.num_imgs = len(images)
        self.gray = gray_images
        self.shape = gray_img.shape


        if show_imgs:
            subplot_rows = math.ceil(math.sqrt(self.num_imgs))
            subplot_cols = math.ceil(self.num_imgs/subplot_rows)

            _, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(22, 10))
            ax = ax.flatten()
            for img in range(subplot_rows*subplot_cols):
                if img < self.num_imgs:
                    ax[img].imshow(self.rgb[img])
                
                ax[img].axis("off")
            
            plt.tight_layout()
            plt.show()