
import cv2
import numpy as np
import random

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def filter__unsharp_mask(img):
    img_original = img.copy()
    img_processed = unsharp_mask(
        img_original,
        amount=random.uniform(0.0,100.0),
        sigma=random.uniform(0.0,10.0),
        threshold=random.uniform(0,256))
    return img_processed

def filter__removeShadow(img):
    img_original = img.copy()
    dilated_img = cv2.dilate(img_original, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img_original, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def filter__threshold(img):
    img_original = img.copy()
    _, img_processed = cv2.threshold(
        img_original,
        random.randint(1,128),
        255,
        random.choice([cv2.THRESH_OTSU, cv2.THRESH_BINARY])
    )
    return img_processed
    
def filter__adaptiveThreshold(img):
    img_original = img.copy()
    img_processed = cv2.adaptiveThreshold(
        img_original,
        random.randint(1,128),
        random.choice([cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]),
        cv2.THRESH_BINARY_INV,
        1+random.randint(1,5)*2,
        random.randint(1,8)
    )
    return img_processed
    
def filter__linear_brightness(img):
    alpha = random.uniform(0.0, 5.0)
    beta = random.uniform(0.0, 5.0)
    
    img_original = img.copy()
    img_processed = cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    return img_processed
    
#Gamma correction

def filter__gamma_correction(img):
    gamma = random.uniform(0.0, 3.0)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    img_original = img.copy()
    img_processed = cv2.LUT(img_original, lookUpTable)
    return img_processed
    

