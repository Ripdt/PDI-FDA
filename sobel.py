import numpy as np
from convolution import conv2d_sharpening

def sobel_sharpening(img : np.ndarray) -> np.ndarray:
    kernel_sobel_1 = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    kernel_sobel_2 = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))

    img_sobel_1 = conv2d_sharpening(img, kernel_sobel_1)
    img_sobel_2 = conv2d_sharpening(img, kernel_sobel_2)

    return np.abs(img_sobel_1)+np.abs(img_sobel_2)