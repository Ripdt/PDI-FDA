import numpy as np

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float
    mse = calculate_mse(original, compressed)
    if mse == 0: # imagens iguais
        return float('inf') # PSNR infinito
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr