import numpy as np

def decrease_contrast(image, factor):
    height, width, _ = image.shape
    contrast_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            for k in range(3):  
                new_value = int(factor * (image[i, j, k] - 128) + 128)
                contrast_image[i, j, k] = np.clip(new_value, 0, 255)

    return contrast_image