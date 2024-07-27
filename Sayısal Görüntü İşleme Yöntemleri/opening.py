import numpy as np
import cv2

def erode(image, kernel):
    img_height, img_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    eroded_image = np.zeros_like(image)
    
    for i in range(img_height):
        for j in range(img_width):
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]
            if image.ndim == 2:
                eroded_image[i, j] = np.min(roi[kernel == 1])
            else:
                for k in range(image.shape[2]):
                    eroded_image[i, j, k] = np.min(roi[kernel == 1, k])
    
    return eroded_image

def dilate(image, kernel):
    img_height, img_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    dilated_image = np.zeros_like(image)
    
    for i in range(img_height):
        for j in range(img_width):
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]
            if image.ndim == 2:
                dilated_image[i, j] = np.max(roi[kernel == 1])
            else:
                for k in range(image.shape[2]):
                    dilated_image[i, j, k] = np.max(roi[kernel == 1, k])
    
    return dilated_image

def opening(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded = erode(image, kernel)
    opened = dilate(eroded, kernel)
    return opened

def closing(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = dilate(image, kernel)
    closed = erode(dilated, kernel)
    return closed
