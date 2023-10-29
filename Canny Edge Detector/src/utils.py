import numpy as np
import cv2 

DER_KERNEL = np.array([1, 0, -1])
DER_KERNEL_FLIP = np.flip(DER_KERNEL)


def convert2gray(img : np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def get1D_gaussian(size : int):
    return cv2.getGaussianKernel(ksize=size,sigma=0)

def conv_1d(img : np.ndarray, kernel : np.ndarray):
    assert(kernel.shape[0] % 2 == 1)
    pad_size = (kernel.shape[0] - 1)/2

    
    



