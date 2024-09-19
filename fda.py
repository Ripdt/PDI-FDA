import numpy as np
from convolution import conv2d_with_lookup_table

from math import exp,log

def lookup_table(lmbda : int = 1, delta : float =.5) -> np.ndarray:
    lut = np.zeros(shape=256, dtype=np.float64)
    if lmbda < 1 or delta < 0 or delta > .5:
        return lut

    euler = np.e
    for i in range(0, 255):
        euler_powered = np.pow(euler, -(np.pow(i, 1/5)/lmbda)/5)
        lut[i] = (1 - np.pow(euler, (-8*delta*euler_powered))) / 8

    return lut

def fda_filter(img : np.ndarray) -> np.ndarray:
    return conv2d_with_lookup_table(
        img=img, 
        lookup_table=lookup_table(),
        k_height=3,
        k_width=3
    )