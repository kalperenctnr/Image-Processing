import pandas as pd
import cv2 as cv
import numpy as np
import os


def create_csv_dataset(images_path : str, csv_name : str, csv_path : str):
    labels = []
    img_paths = []
    image_folders = os.listdir(images_path)

    for folder in image_folders:
        image_files = os.listdir(images_path+folder+"/")

        for image_file in image_files:
            img_paths.append(folder +"/" + image_file)
            labels.append(folder)
            
    df = pd.DataFrame(zip(img_paths, labels), columns=["Image", "Label"])
    df.to_csv(csv_path+csv_name+'.csv', header=False, index=False)
    
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def generate_latent_variables(img : np.ndarray):
    mu = np.mean(img.reshape(3, -1), axis=1, dtype=np.float64)/255 # take the mean in every chanel for the first RGB moment (normalized)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
    img_hsv = img_hsv.astype(np.float64)
    img_hsv[:, :, 0] = np.cos(np.pi * img_hsv[:, :, 0]/180)
    ## hist = cv.calcHist([img_hsv], channels=[0, 1], mask=None, histSize=[8, 8], ranges=[[-1, 1], [0, 256]])
    hist_h = np.histogram(img_hsv[:, :, 0], bins=8, range=[-1, 1])
    hist_s = np.histogram(img_hsv[:, :, 1], bins=8, range=[0, 256])

    hist_h = softmax(np.asarray(hist_h[0], dtype=np.float64))
    hist_s = softmax(np.asarray(hist_s[0], dtype=np.float64))

    return np.concatenate([mu, hist_h, hist_s]).reshape(1, -1)