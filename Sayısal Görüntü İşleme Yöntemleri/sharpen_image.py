import numpy as np

def sharpen_image(image):

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    height, width, _ = image.shape
    sharpened_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for k in range(3):  
                sharpened_pixel = np.sum(image[i - 1:i + 2, j - 1:j + 2, k] * kernel)
                sharpened_image[i, j, k] = np.clip(sharpened_pixel, 0, 255)

    return sharpened_image
