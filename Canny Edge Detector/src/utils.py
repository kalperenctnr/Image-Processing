import numpy as np
import cv2 

DER_KERNEL = 1/2 * np.array([1, 0, -1])


def convert2gray(img : np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def get1D_gaussian(size : int):
    return cv2.getGaussianKernel(ksize=size,sigma=0)

def conv_1d(img : np.ndarray, kernel : np.ndarray, axis : int):
    assert(kernel.shape[0] % 2 == 1)
    assert(kernel.shape[0] <= img.shape[0])
    
    kernel_flip = kernel[::-1]
    pad_size = int((kernel_flip.shape[0] - 1)/2)
    pad_mat = np.zeros((img.shape[axis], pad_size))
    conv_img = np.zeros_like(img)
    
    if axis == 0:   
        padded_img = np.hstack([pad_mat, img, pad_mat])
        for i in range(conv_img.shape[0]):
            for j in range(conv_img.shape[1]):
                conv_img[i,j] = np.dot(padded_img[i, j:j+kernel_flip.shape[0]], kernel_flip)
    elif axis == 1:
        padded_img = np.vstack([pad_mat.T, img, pad_mat.T])    
        for i in range(conv_img.shape[0]):
                for j in range(conv_img.shape[1]):
                    conv_img[i,j] = np.dot(padded_img[i:i+kernel_flip.shape[0], j], kernel_flip)
    
    return conv_img

# a = np.arange(8)[:, None]
# k = np.asarray([1, 1, 1])
# b = np.arange(8)
# print(conv_1d(a, DER_KERNEL, 1))

# s = np.convolve(b, DER_KERNEL)
# print(s)

gaus = get1D_gaussian(5)

img = cv2.imread(r"C:\Users\K.A.C\Desktop\ITU Dersler\Image Processing\Image-Processing\Canny Edge Detector\images\Lenna.png")
img_gray = convert2gray(img)
gaus_der = conv_1d(gaus, DER_KERNEL, 1)
img_gaus_der_x = conv_1d(img_gray, gaus_der, 0)
img_gaus_der_x_gaus = conv_1d(img_gaus_der_x, gaus, 1)

cv2.imshow("wfad", img_gaus_der_x_gaus)
cv2.waitKey(-1)

    

    
    



