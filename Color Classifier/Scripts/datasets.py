import pandas as pd
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import os

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class ColorImageDataset(Dataset):
    def __init__(self, img_labels : pd.DataFrame, dataset_dir : str, transform=None) -> None:
        self.img_labels = img_labels
        self.dataset_dir = dataset_dir
        self.transfor = transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index) -> None:
        img_path = os.path.join(self.dataset_dir, self.img_labels.iloc[index, 0])
        label = self.img_labels.iloc[index, 1]
        img = cv.imread(img_path) # opencv reads in bgr order
        mu = np.mean(img.reshape(3, -1), axis=1, dtype=np.float64)/255 # take the mean in every chanel for the first RGB moment (normalized)
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
        img_hsv = img_hsv.astype(np.float64)
        img_hsv[:, :, 0] = np.cos(np.pi * img_hsv[:, :, 0]/180)
        ## hist = cv.calcHist([img_hsv], channels=[0, 1], mask=None, histSize=[8, 8], ranges=[[-1, 1], [0, 256]])
        hist_h = np.histogram(img_hsv[:, :, 0], bins=8, range=[-1, 1])
        hist_s = np.histogram(img_hsv[:, :, 1], bins=8, range=[0, 256])

        hist_h = softmax(np.asarray(hist_h[0], dtype=np.float64))
        hist_s = softmax(np.asarray(hist_s[0], dtype=np.float64))
        return np.concatenate([mu, hist_h, hist_s]), label
