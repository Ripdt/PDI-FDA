import cv2
import numpy as np

from noise import salt_n_pepper
from gauss import gauss_filter
from sobel import sobel_sharpening

from fda import fda_filter

from sharpening import sharp_img

# img = cv2.imread('res/lemur.jpg', 0)
img = cv2.imread('res/small-brain-img.png', 0)
# img = cv2.imread('res/big-brain-img.png', 0)

# =========== init =========== 
cv2.imshow('no-modification', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_noise = salt_n_pepper(img, noise_ratio=.05)

cv2.imshow('askpy-noise', img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========== smoothing =========== 
img_gauss = gauss_filter(img=img_noise)
for i in range(4):
    img_gauss = gauss_filter(img=img_gauss)

img_fda = fda_filter(img_noise)
for i in range(2):
    img_fda = fda_filter(img_fda)

# =========== borders =========== 
img_gauss_sobel = sobel_sharpening(img_gauss)
img_fda_sobel = sobel_sharpening(img_fda)

cv2.imshow('sobel-gauss', img_gauss_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('sobel-fda', img_fda_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========== masking =========== 
img_sharp_gauss = sharp_img(img, img_gauss, alpha=.1)

cv2.imshow('sharpened-gauss', img_sharp_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()


img_sharp_fda = sharp_img(img, img_fda, alpha=.1)

cv2.imshow('sharpened-fda', img_sharp_fda)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========== borders =========== 
img_gauss_sharp_sobel = sobel_sharpening(img_sharp_gauss)
img_fda_sharp_sobel = sobel_sharpening(img_sharp_fda)

cv2.imshow('sobel-sharp-gauss', img_gauss_sharp_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('sobel-sharp-fda', img_fda_sharp_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
