import numpy as np

def apply_sobel_filter(image):

    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_kernel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    height, width = image.shape

    filtered_image_x = np.zeros_like(image)
    filtered_image_y = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):

            convolution_x = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_kernel_x)
            convolution_y = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_kernel_y)

            filtered_image_x[i, j] = np.clip(convolution_x, 0, 255)
            filtered_image_y[i, j] = np.clip(convolution_y, 0, 255)

    gradient_magnitude = np.sqrt(filtered_image_x**2 + filtered_image_y**2)

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    return gradient_magnitude.astype(np.uint8)
