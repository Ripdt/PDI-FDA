import cv2
import numpy as np

from noise import salt_n_pepper
from gauss import gauss_filter
from sobel import sobel_sharpening

img = cv2.imread('res/small-brain-img.png', 0)
# img = cv2.imread('res/circle.jpg', 0)
# img = cv2.imread('res/lemur.jpg', 0)

cv2.imshow('no-modification', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_noise = salt_n_pepper(img)

cv2.imshow('askpy-noise', img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
TODO:
    - fix sobel: i think the result is wrong (or the image chosen isnt good)
    - implement fda (with lookup table)
    - implement PSNR and MSE
    - implement unsharp masking and highboost filtering
'''

# gauss being executed 20 times is why i think sobel is wrong (._. )
# img_gauss = gauss_filter(img=img_noise);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);

# img_gauss = gauss_filter(img=img_noise);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);
# img_gauss = gauss_filter(img=img_gauss);

# cv2.imshow('gauss', img_gauss)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_sobel_gauss = sobel_sharpening(img)
print(img_sobel_gauss.dtype)

cv2.imshow('sobel', img_sobel_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()