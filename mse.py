import numpy as np

def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    # Diferença entre as imagens
    erro = original.astype(np.float64) - compressed.astype(np.float64)
    # Cálculo MSE
    mse = np.mean(erro ** 2)
    return mse