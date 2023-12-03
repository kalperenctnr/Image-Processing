import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

DER_KERNEL = 1/2 * np.array([1, 0, -1])
GAUSS_KERNEL = 1/4 * np.array([1, 2, 1])

# utilizing the opencv function to convert to grayscale
def convert2gray(img : np.ndarray):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# generate the gaussian kernel as na approximation of the gaussian distribution
def get1D_gaussian(sigma : int):
    k = np.arange(-2*sigma, 2*sigma+1, dtype=np.float64)
    temp = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((k ** 2)) / (2 * (sigma ** 2)))
    return temp / np.sum(np.absolute(temp))

def conv_1d(img : np.ndarray, kernel : np.ndarray, axis : int):
    assert(kernel.shape[0] % 2 == 1)
    assert(kernel.shape[0] <= img.shape[0])
    
    kernel_flip = kernel[::-1] # conv flip
    pad_size = int((kernel_flip.shape[0] - 1)/2) # padding to preserve the size
    pad_mat = np.zeros((img.shape[axis], pad_size))
    conv_img = np.zeros_like(img)
    
    if axis == 0:   
        padded_img = np.hstack([pad_mat, img, pad_mat])
        for i in range(conv_img.shape[0]):
            for j in range(conv_img.shape[1]):
                conv_img[i,j] = np.dot(padded_img[i, j:j+kernel_flip.shape[0]], kernel_flip) # after the flip conv is like cross correlation
    elif axis == 1:
        padded_img = np.vstack([pad_mat.T, img, pad_mat.T])    
        for i in range(conv_img.shape[0]):
                for j in range(conv_img.shape[1]):
                    conv_img[i,j] = np.dot(padded_img[i:i+kernel_flip.shape[0], j], kernel_flip)
    
    return conv_img

# Check if all pixels are a fit
def Fit(img_part : np.ndarray, kernel : np.ndarray, kernel_sum : int) -> bool:
    assert(img_part.shape == kernel.shape)
    temp = img_part * kernel 
    return temp.sum() == kernel_sum

# Chekc if there is any hit
def Hit(bg_part : np.ndarray, kernel : np.ndarray) -> bool:
    assert(bg_part.shape == kernel.shape)
    return (bg_part * kernel).sum() >= 1

# Erode the image by the fit rule
def erode(img_binary : np.ndarray, kernel : np.ndarray, polarity : bool = True) -> np.ndarray:
    assert(kernel.shape[0] == kernel.shape[1])
    n = kernel.shape[0]
    check_pixel = polarity * 1
    target = 1 - check_pixel
    kernel_sum = kernel.sum()
    
    output = img_binary.copy()
    for i in range(int(n/2), img_binary.shape[0]-(int(n/2) +1)):
        for j in range(int(n/2), img_binary.shape[1]-(int(n/2)+1)):
            if img_binary[i, j] == check_pixel and not Fit(img_binary[i- int(n/2):i+int(n/2) +1, j-int(n/2):j+int(n/2) + 1], kernel, kernel_sum):
                output[i, j] = target
                
    return output

# Dilate the image by the hit rule
def dilate(img_binary : np.ndarray, kernel : np.ndarray, polarity : bool = True) -> np.ndarray:
    assert(kernel.shape[0] == kernel.shape[1])
    n = kernel.shape[0]
    check_pixel = 1 - polarity * 1
    target = 1 - check_pixel
    
    output = img_binary.copy()
    for i in range(int(n/2), img_binary.shape[0]-(int(n/2) +1)):
        for j in range(int(n/2), img_binary.shape[1]-(int(n/2)+1)):
            if img_binary[i, j] == check_pixel and Hit(img_binary[i- int(n/2):i+int(n/2) +1, j-int(n/2):j+int(n/2) + 1], kernel):
                    output[i, j] = target
    return output

# We can maximize the inter-class variance instead of minimizing intra-class for bimodal 
# for otsu to function properly 
# the histogram should have a bimodal distribution with 
# a deep and sharp valley between the two peaks                                
def Otsu(img : np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(img, bins=img.max()-img.min() + 1,range=(img.min(), img.max()))

    pixel_values = np.arange(img.min(), img.max()+1)

    w_0 = np.cumsum(hist)
    w_1 = w_0[-1] - w_0

    expectation_0 = np.cumsum(hist * pixel_values)
    expectation_1 = expectation_0[-1] - expectation_0

    m_0 =  expectation_0 / (w_0 + 1e-6)
    m_1 = expectation_1 / (w_1 + 1e-6)

    var = w_0 * w_1 * (m_0 - m_1) ** 2
    th = pixel_values[np.argmax(var)]

    img_binary = np.zeros_like(img)
    img_binary[img >= th] = 1
    # img_binary = 1 - img_binary
    
    return img_binary, th

# 8-connected nb
def check_neighbours(pad_img, j, i):
    assert(j > 0 and j < pad_img.shape[1] and i > 0 and i < pad_img.shape[0])
    neighbour_labels = set()
    if pad_img[i, j-1] > 0:
        neighbour_labels.add(pad_img[i, j-1])
    if pad_img[i-1, j-1] > 0:
        neighbour_labels.add(pad_img[i-1, j-1])
    if pad_img[i-1, j] > 0:
        neighbour_labels.add(pad_img[i-1, j])
    if pad_img[i-1, j+1] > 0:
        neighbour_labels.add(pad_img[i-1, j+1])
    
    neighbour_labels = sorted(neighbour_labels)
    return neighbour_labels

def pad(img):
    v = np.zeros((1, img.shape[1]))
    h = np.zeros((img.shape[0]+2, 1))
    
    img = np.vstack([v, img, v])
    img = np.hstack([h, img, h])
    
    return img

def local_otsu(img : np.ndarray, size : tuple) -> np.ndarray:
    assert(img.shape[0] >= size[0] and img.shape[1] >= size[1])
    # _, th = Otsu(img)
    # print(th)
    # print("-------")
    output = np.zeros_like(img)
    
    prev_i, prev_j = 0, 0
    for i in range(size[0], img.shape[0]+size[0], size[0]):
        prev_j = 0
        for j in range(size[1], img.shape[1]+size[1], size[1]):
            output[prev_i:min(i, img.shape[0]), prev_j:min(j, img.shape[1])], th_local = Otsu(img[prev_i:min(i, img.shape[0]), prev_j:min(j, img.shape[1])])
            if np.abs(th_local - np.mean(img[prev_i:min(i, img.shape[0]), prev_j:min(j, img.shape[1])])) < 6: # this assusmes a scaled grayscale image
                output[prev_i:min(i, img.shape[0]), prev_j:min(j, img.shape[1])] = 1
            prev_j = j
        prev_i = i
    
    return output

# 2 stage CCA with union find algo
def CCA(img):
    pad_img = pad(img)
    output_img = np.zeros_like(pad_img)

    eq_list_aug = {}
    label = 1
    for i in range(1, pad_img.shape[0]-1):
        for j in range(1, pad_img.shape[1]-1):
            if pad_img[i, j] == 1:
                label_set = check_neighbours(output_img, j, i)
                
                if len(label_set) == 0:
                    output_img[i, j] = label
                    eq_list_aug[label] = label
                    label += 1
                elif len(label_set) == 1:
                    output_img[i, j] = label_set[0]
                    
                else:
                    output_img[i, j] = label_set[0]
                    for other_label in label_set:
                        eq_list_aug[other_label] = eq_list_aug[eq_list_aug[label_set[0]]]
                        
    for i in range(1, pad_img.shape[0]-1):
        for j in range(1, pad_img.shape[1]-1):
            if pad_img[i, j] == 1:
                 output_img[i, j] = eq_list_aug[output_img[i, j]]
    
    label_set = set()  
    for _, label_ in eq_list_aug.items():
       label_set.add(label_)              
      
    return output_img, label_set
