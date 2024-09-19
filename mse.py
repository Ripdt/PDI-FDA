import numpy as np

def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    # Diferença entre as imagens
    err = np.sum((original.astype(float) - compressed.astype(float)) ** 2)
    # Cálculo MSE
    err /= float(original.shape[0] * original.shape[1])
    return err