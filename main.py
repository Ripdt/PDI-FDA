import cv2
import numpy as np

from noise import salt_n_pepper
from gauss import gauss_filter
from sobel import sobel_sharpening_optmized

# img = cv2.imread('res/lemur.jpg', 0)
img = cv2.imread('res/small-brain-img.png', 0)

cv2.imshow('no-modification', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_noise = salt_n_pepper(img, noise_ratio=0.05)

cv2.imshow('askpy-noise', img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gauss = gauss_filter(img=img_noise)
for i in range(5):
    img_gauss = gauss_filter(img=img_gauss)

img_gauss_sobel = sobel_sharpening_optmized(img_gauss)

cv2.imshow('sobel-gauss', img_gauss_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
