import numpy as np
import cv2 

DER_KERNEL = 1/2 * np.array([1, 0, -1])
GAUSS_KERNEL = 1/4 * np.array([1, 2, 1])

def convert2gray(img : np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def get1D_gaussian(sigma : int):
    k = np.arange(-2*sigma, 2*sigma+1, dtype=np.float64)
    temp = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((k ** 2)) / (2 * (sigma ** 2)))
    return temp / np.sum(np.absolute(temp))

def get1D_derGaussian(sigma : int):
    k = np.arange(-2*sigma, 2*sigma+1)
    temp = k * 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((k ** 2)) / (2 * (sigma ** 2)))

    return temp / np.sum(np.absolute(temp))

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

def threshold(img : np.ndarray, min_val : int, max_val : int):
    mask_max = (img > max_val) * 1 
    mask = ((img < max_val) & (img > min_val))
    mask_final = np.zeros_like(mask_max, dtype=np.uint8)
    for i in range(img.shape[0]-3):
        for j in range(img.shape[1]-3):
            if mask[i+1, j+1]:
                mask_final[i+1, j+1] = 1 *(np.sum(mask_max[i:i+3, j:j+3]) >= 1)
    
    mask_final = mask_final + mask_max
    
    return mask_final

def nms(theta : np.ndarray, G : np.ndarray):
    G_nms = np.zeros_like(G)
    for i in range(1, theta.shape[0]-1):
        for j in range(1, theta.shape[1]-1):
            u, d = 255, 255
            #angle 0
            if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180) or (-22.5 <= theta[i,j] < 0) or (-180 <= theta[i,j] < -157.5):
                u = G[i, j+1]
                d = G[i, j-1]
            #angle 45
            elif (22.5 <= theta[i,j] < 67.5) or (-157.5 <= theta[i,j] < -122.5):
                u = G[i+1, j-1]
                d = G[i-1, j+1]
            #angle 90
            elif (67.5 <= theta[i,j] < 112.5) or (-122.5 <= theta[i,j] < -67.5):
                u = G[i+1, j]
                d = G[i-1, j]
            #angle 135
            elif (112.5 <= theta[i,j] < 157.5) or (-67.5 <= theta[i,j] < -22.5):
                u = G[i-1, j-1]
                d = G[i+1, j+1]
                
            
            if u < G[i, j] and d < G[i, j]:
                G_nms[i, j] = G[i, j]
                              
    return G_nms


def sobel_filters(img):
    sobel_x = conv_1d(img, GAUSS_KERNEL, 1)
    sobel_x = conv_1d(sobel_x, DER_KERNEL, 0)

    sobel_y = conv_1d(img, GAUSS_KERNEL, 0)
    sobel_y = conv_1d(sobel_y, DER_KERNEL, 1)
    
    G = np.sqrt(sobel_x**2 + sobel_y**2 )
    G = G / G.max() * 255
    theta = np.rad2deg(np.arctan2(sobel_y, sobel_x))
    
    return G, theta

def Canny(img : np.ndarray,  min_val : int, max_val : int):
    G, theta = sobel_filters(img)
    G_nms = nms(theta, G)
    mask = threshold(G_nms, 50, 200)
    G_th = G_nms * mask
    
    return G_th
    


