import numpy as np

def histogram_equalization(image):

    height, width = image.shape

    histogram = np.zeros(256, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1

    histogram /= (height * width)

    cdf = np.cumsum(histogram)

    transformation_function = np.round(cdf * 255).astype(np.uint8)

    equalized_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            equalized_image[i, j] = transformation_function[image[i, j]]

    return equalized_image
