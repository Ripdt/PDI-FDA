import cv2
import numpy as np

from noise import salt_n_pepper
from gauss import gauss_filter
from convolution import conv2d_sharpening


img = cv2.imread('res/big-brain-img.png', 0)

cv2.imshow('no-modification', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_noise = salt_n_pepper(img)

cv2.imshow('askpy-noise', img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gauss = gauss_filter(img=img_noise);
img_gauss = gauss_filter(img=img_gauss);
img_gauss = gauss_filter(img=img_gauss);

cv2.imshow('gauss', img_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel_sobel_1 = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
kernel_sobel_2 = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))

img_sobel_1 = conv2d_sharpening(img_gauss, kernel_sobel_1)
img_sobel_2 = conv2d_sharpening(img_gauss, kernel_sobel_2)

img_sobel = np.abs(img_sobel_1)+np.abs(img_sobel_2)

cv2.imshow('sobel', gauss_filter(img=img_sobel))
cv2.waitKey(0)
cv2.destroyAllWindows()