import numpy as np

def resize_image(image, new_width, new_height):
    # Get the height and width of the original image
    height, width, _ = image.shape

    # Create an empty numpy array to store the resized image
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate scaling factors
    scale_x = width / new_width
    scale_y = height / new_height

    # Loop through each pixel of the resized image
    for i in range(new_height):
        for j in range(new_width):
            # Find the corresponding pixel position in the original image
            original_x = int(j * scale_x)
            original_y = int(i * scale_y)
            
            # Assign the pixel value from the original image to the resized image
            resized_image[i, j] = image[original_y, original_x]

    return resized_image
