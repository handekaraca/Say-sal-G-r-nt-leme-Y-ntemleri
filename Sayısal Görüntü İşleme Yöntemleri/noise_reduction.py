import numpy as np

def noise_reduction(image):
    height, width, _ = image.shape
    denoised_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            denoised_image[i, j] = np.median(image[i - 1:i + 2, j - 1:j + 2].reshape(-1, 3), axis=0)

    return denoised_image
