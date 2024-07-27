import numpy as np

def apply_blur_filter(image):
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9


    height, width, _ = image.shape

    filtered_image = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for k in range(3): 

                convolution = np.sum(image[i - 1:i + 2, j - 1:j + 2, k] * kernel)

                filtered_image[i, j, k] = np.clip(convolution, 0, 255)

    return filtered_image
