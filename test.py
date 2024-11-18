import cv2
import numpy as np
from collections import Counter

def is_vivid_color(pixel, threshold=30):
    r, g, b = pixel
    if max(abs(r - g), abs(g - b), abs(b - r)) > threshold:
        return True
    return False

def dominant_vivid_color(screenshot, k=5):
    # Reshape the image to a list of pixels
    pixels = screenshot.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Filter out grayscale pixels
    vivid_pixels = np.array([pixel for pixel in pixels if is_vivid_color(pixel)])
    
    if len(vivid_pixels) == 0:
        return None  # No vivid colors found

    # Define k-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Apply k-means clustering
    _, labels, centers = cv2.kmeans(vivid_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    # Find the most vivid color (with maximum saturation)
    vivid_color = max(centers, key=lambda c: np.std(c))

    return vivid_color

# Example usage:
screenshot = cv2.imread('test3.png')
vivid_color = dominant_vivid_color(screenshot, k=5)
print('Most vivid color:', vivid_color)

vivid_color_image = np.zeros((100, 100, 3), dtype=np.uint8)
vivid_color_image[:] = vivid_color

cv2.imshow('Most Vivid and Prevalent Color', vivid_color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()