import numpy as np

def grayscale_conversion(image):
    height, width, _ = image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            blue = image[i, j, 0]
            green = image[i, j, 1]
            red = image[i, j, 2]
            grayscale_pixel = 0.299 * red + 0.587 * green + 0.114 * blue
            grayscale_pixel = np.clip(grayscale_pixel, 0, 255)  # Piksel değerini 0 ile 255 arasında kısıtla
            grayscale_image[i, j] = np.uint8(grayscale_pixel)  # Sonucu uint8 türüne dönüştür

    return grayscale_image

