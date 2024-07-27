import numpy as np

def gaussian_function(x, y, sigma):

    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

def create_gaussian_kernel(size, sigma):

    kernel = np.zeros((size, size))

    center = size // 2

    for i in range(size):
        for j in range(size):

            distance_x = i - center
            distance_y = j - center

            kernel[i, j] = gaussian_function(distance_x, distance_y, sigma)


    kernel /= np.sum(kernel)

    return kernel

def apply_gaussian_filter(image, kernel_size, sigma):

    kernel = create_gaussian_kernel(kernel_size, sigma)

    height, width, _ = image.shape

    filtered_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            for k in range(3):  
                convolution = 0
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        pixel_value = image[min(max(i + m - kernel_size // 2, 0), height - 1),
                                            min(max(j + n - kernel_size // 2, 0), width - 1), k]
                        convolution += pixel_value * kernel[m, n]
                filtered_image[i, j, k] = np.clip(convolution, 0, 255)

    return filtered_image
