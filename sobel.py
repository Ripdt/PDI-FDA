import numpy as np
from convolution import conv2d_sharpening

def sobel_sharpening(img : np.ndarray) -> np.ndarray:
    kernel_sobel_vertical = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    kernel_sobel_horizontal = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))

    img_sobel_1 = conv2d_sharpening(img, kernel_sobel_vertical)
    img_sobel_2 = conv2d_sharpening(img, kernel_sobel_horizontal)
    
    return np.hypot(img_sobel_1, img_sobel_2)
