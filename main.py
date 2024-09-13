import cv2
import numpy as np

def add_salt_and_pepper_noise(image : np.ndarray, noise_ratio=0.02) -> np.ndarray:
    noisy_image = image.copy()
    h, w = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
 
    for _ in range(noisy_pixels):
        row, col= np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = 0
        else:
            noisy_image[row, col] = 255
 
    return noisy_image

img = cv2.imread('res/small-brain-img.png', 0)

cv2.imshow('no-modification', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(type(img))

img_noise = add_salt_and_pepper_noise(img)

cv2.imshow('askpy-noise', img_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

