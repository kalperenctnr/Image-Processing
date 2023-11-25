import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

# for otsu to function properly 
# the histogram should have a bimodal distribution with 
# a deep and sharp valley between the two peaks
path = r"C:\Users\aselsan\Desktop\ML\Image Processing\bird images\bird 2.jpg"
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY ) 
img = cv.GaussianBlur(img, (7,7), 0)

hist, bin_edges = np.histogram(img, bins=img.max()-img.min() + 1,range=(img.min(), img.max()))

# plt.bar(bin_edges[:-1], hist)
# plt.show()
pixel_values = np.arange(img.min(), img.max()+1)

w_0 = np.cumsum(hist)
w_1 = w_0[-1] - w_0

expectation_0 = np.cumsum(hist * pixel_values)
expectation_1 = expectation_0[-1] - expectation_0

m_0 =  expectation_0 / (w_0 + 1e-6)
m_1 = expectation_1 / (w_1 + 1e-6)

var = w_0 * w_1 * (m_0 - m_1) ** 2
th = pixel_values[np.argmax(var)]
print(th)
img_binary = np.zeros_like(img)
img_binary[img >= th] = 1
img_binary = 1 - img_binary

# plt.imshow(img_binary, cmap='gray')
# plt.show()



n = 9
dilate_opr = np.ones((n, n), dtype=np.uint8)

output = img_binary.copy()
for i in range(int(n/2), img_binary.shape[0]-(int(n/2) +1)):
    for j in range(int(n/2), img_binary.shape[1]-(int(n/2)+1)):
        if img_binary[i, j] == 0:
            if np.sum(img_binary[i- int(n/2):i+int(n/2) +1, j-int(n/2):j+int(n/2) + 1] * dilate_opr) >= 1:
                output[i, j] = 1

img_binary = output

# plt.imshow(output*255, cmap='gray')
# plt.show()
n = 5
erode_opr = np.ones((n, n), dtype=np.uint8)

output = img_binary.copy()
for i in range(int(n/2), img_binary.shape[0]-(int(n/2) +1)):
    for j in range(int(n/2), img_binary.shape[1]-(int(n/2)+1)):
        if img_binary[i, j] == 1:
            if np.sum(img_binary[i- int(n/2):i+int(n/2) +1, j-int(n/2):j+int(n/2) + 1] * erode_opr) < n**2:
                output[i, j] = 0

img_binary = output

plt.imshow(img_binary, cmap='gray')
plt.show()