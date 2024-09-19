import cv2
import numpy as np

from noise import salt_n_pepper
from gauss import gauss_filter
from sobel import sobel_sharpening, sobel_sharpening_optmized
from fda import fda_filter
from sharpening import sharp_img
from mse import calculate_mse
from psnr import calculate_psnr

from os import mkdir
from os.path import exists
from shutil import rmtree

directory = './results'
if exists(directory):
    rmtree(directory)

def show_and_save_img(img : np.ndarray, filename : str) -> None:
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(directory + '/' + filename + '.png', img)

mkdir(directory)
    
# =========== init =========== 
img = cv2.imread('res/small-brain-img.png', 0)

show_and_save_img(img, filename='no-modification')

# =========== noise =========== 
img_noise = salt_n_pepper(img, noise_ratio=.05)

show_and_save_img(img_noise, filename='noise')

# =========== smoothing =========== 
img_gauss = gauss_filter(img=img_noise)
for i in range(4):
    img_gauss = gauss_filter(img=img_gauss)

img_fda = fda_filter(img_noise)
for i in range(5):
    img_fda = fda_filter(img_fda)

show_and_save_img(img_gauss, 'gauss')
show_and_save_img(img_fda, 'fda')

# =========== borders =========== 
img_gauss_sobel = sobel_sharpening_optmized(img_gauss)
img_fda_sobel = sobel_sharpening_optmized(img_fda)

show_and_save_img(img_gauss_sobel, 'sobel-gauss')
show_and_save_img(img_fda_sobel, 'sobel-fda')

# =========== masking =========== 
img_sharp_gauss = sharp_img(img, img_gauss, alpha=.1)
img_sharp_fda = sharp_img(img, img_fda, alpha=.1)

show_and_save_img(img_sharp_gauss, 'sharpened-gauss')
show_and_save_img(img_sharp_fda, 'sharpened-fda')

# =========== borders =========== 
img_gauss_sharp_sobel = sobel_sharpening_optmized(img_sharp_gauss)
img_fda_sharp_sobel = sobel_sharpening_optmized(img_sharp_fda)

show_and_save_img(img_gauss_sharp_sobel, 'sobel-sharp-gauss')
show_and_save_img(img_fda_sharp_sobel, 'sobel-sharp-fda')

# =========== metrics =========== 
mse_gauss = calculate_mse(img, img_sharp_gauss)
psnr_gauss = calculate_psnr(img, img_sharp_gauss)

mse_fda = calculate_mse(img, img_sharp_fda)
psnr_fda = calculate_psnr(img, img_sharp_fda)

print()
print('MSE - \tGAUSS: ' + str(mse_gauss))
print('MSE - \tFDA: ' + str(mse_fda))
print()
print('PSNR - \tGAUSS: ' + str(psnr_gauss))
print('PSNR - \tFDA: ' + str(psnr_fda))
print()
