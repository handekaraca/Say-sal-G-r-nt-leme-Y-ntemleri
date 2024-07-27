import numpy as np
from grayscale_conversion import grayscale_conversion

def detect_edges(image, threshold1, threshold2):
    np.seterr(over='ignore')
    # Convert the image to grayscale
    grayscale_image = grayscale_conversion(image)

    # Get the height and width of the image
    height, width = grayscale_image.shape

    # Create an empty numpy array to store the edge-detected image
    edge_image = np.zeros((height, width), dtype=np.uint8)

    # Apply Canny edge detection algorithm
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Compute gradient intensity and direction
            dx = grayscale_image[i + 1, j] - grayscale_image[i - 1, j]
            dy = grayscale_image[i, j + 1] - grayscale_image[i, j - 1]
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            gradient_direction = np.arctan2(dy, dx) * (180 / np.pi)
            
            # Quantize gradient direction to nearest 45 degrees
            quantized_direction = np.round(gradient_direction / 45) * 45

            # Apply non-maximum suppression
            if ((quantized_direction == 0 or quantized_direction == 180) and 
                (gradient_magnitude > grayscale_image[i, j - 1]) and 
                (gradient_magnitude > grayscale_image[i, j + 1])):
                edge_image[i, j] = gradient_magnitude
            elif ((quantized_direction == 45 or quantized_direction == 225) and 
                  (gradient_magnitude > grayscale_image[i - 1, j + 1]) and 
                  (gradient_magnitude > grayscale_image[i + 1, j - 1])):
                edge_image[i, j] = gradient_magnitude
            elif ((quantized_direction == 90 or quantized_direction == 270) and 
                  (gradient_magnitude > grayscale_image[i - 1, j]) and 
                  (gradient_magnitude > grayscale_image[i + 1, j])):
                edge_image[i, j] = gradient_magnitude
            elif ((quantized_direction == 135 or quantized_direction == 315) and 
                  (gradient_magnitude > grayscale_image[i - 1, j - 1]) and 
                  (gradient_magnitude > grayscale_image[i + 1, j + 1])):
                edge_image[i, j] = gradient_magnitude

    # Apply double thresholding and edge tracking by hysteresis
    for i in range(height):
        for j in range(width):
            if edge_image[i, j] < threshold1:
                edge_image[i, j] = 0
            elif edge_image[i, j] > threshold2:
                edge_image[i, j] = 255
            else:
                # Check 8-connected neighbors
                neighbors = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                             (i, j - 1),                 (i, j + 1),
                             (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
                for neighbor_i, neighbor_j in neighbors:
                    if 0 <= neighbor_i < height and 0 <= neighbor_j < width and edge_image[neighbor_i, neighbor_j] == 255:
                        edge_image[i, j] = 255
                        break
                else:
                    edge_image[i, j] = 0

    return edge_image
