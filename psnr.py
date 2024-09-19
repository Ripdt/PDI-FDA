import numpy as np
from mse import calculate_mse

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    mse = calculate_mse(original, compressed)
    if mse == 0: # imagens iguais
        return float('inf') # PSNR infinito
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)