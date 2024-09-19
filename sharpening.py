from numpy import ndarray, clip, uint8

def sharp_img(img : ndarray, smoothed_img : ndarray, alpha=1) -> ndarray:
    img_mask = img - smoothed_img
    img_sharpened = img + alpha * img_mask
    return clip(img_sharpened, 0, 255).astype(uint8)