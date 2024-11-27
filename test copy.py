import numpy as np
import cv2
import dxcam
import time

# Initialize the camera to capture the entire screen
cam = dxcam.create(output_idx=0, output_color="BGR")

# Function to extract colors based on a color extraction method
def color_extraction(image_list, color_func, led_number, use_constant_color=False, constant_color=None):
    if use_constant_color and constant_color:
        return [constant_color for _ in range(led_number)]
    return [color_func(image_list[n]) for n in range(led_number)]

# Function to reduce darkness in the image (helps enhance visibility)
def reduce_darkness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    mask = (s < 20) & (v < 220) | (v < 50) | ((10 <= h) & (h <= 50) & (s < 50))
    image[mask] = [0, 0, 0]
    return image

# Function to get the dominant color from an image using k-means clustering
def dominant_color(screenshot, k=3):
    pixels = screenshot.reshape((-1, 3)).astype(np.float32)
    if pixels.shape[0] == 0:  # If the image is empty, return a default color
        return [0, 0, 0]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    return max(centers, key=lambda c: np.std(c))

# Function to process a screenshot and divide it into sections
def process_screenshot(scale_factor=0.0625):
    # Grab the entire screen using dxcam
    screenshot = cam.grab()
    
    # If no screenshot is captured, return None
    if screenshot is None:
        return None, None
    
    # Resize and reduce darkness for better processing
    screenshot_resized = cv2.resize(screenshot, (0, 0), fx=scale_factor, fy=scale_factor)
    screenshot_vivid = reduce_darkness(screenshot_resized)
    
    return screenshot_vivid, screenshot

def color_string(color_list):
    """
    Returns color string completed with redundant zeros.
    As a result output string for esp32 has the same length every time.
    :param color_list:  list of colors extracted in previous steps
    :param scale:       parameter that specifies if color scale is reduced from 0-255 to 0-99
    :return:            Color string
    """

    color_list_hex = []
    for color in color_list:
        color_hex = []
        color_hex.append(hex(color[0])[2:])
        if len(color_hex[0]) < 2:
            color_hex[0] = '0' + color_hex[0]
        color_hex.append(hex(color[1])[2:])
        if len(color_hex[1]) < 2:
            color_hex[1] = '0' + color_hex[1]
        color_hex.append(hex(color[2])[2:])
        if len(color_hex[2]) < 2:
            color_hex[2] = '0' + color_hex[2]
        color_list_hex.append(color_hex)

    flattened_color_list_hex = [channel for color in color_list_hex for channel in color]
    long_string = ''.join(flattened_color_list_hex)

    return long_string

# Main function to generate LED colors
def color_gen(led_span=1, h_leds=18, w_leds=32, h=1440, w=2560, use_constant_color=False, constant_color=None, method=dominant_color):
   
    # Grab and process the full screen image
    resized_screenshot_bgr, full_screenshot_bgr = process_screenshot()
    start_time = time.time()
    
    # If no screenshot is captured, exit
    if resized_screenshot_bgr is None:
        print("Failed to capture screenshot")
        return
    
    # Divide the image into 3 sections: upper, left, right
    h_divider = 8
    w_divider = 16

    u_box = resized_screenshot_bgr[:resized_screenshot_bgr.shape[0] // h_divider, :]
    l_box = resized_screenshot_bgr[resized_screenshot_bgr.shape[0] // h_divider:, :resized_screenshot_bgr.shape[1] // w_divider]
    r_box = resized_screenshot_bgr[resized_screenshot_bgr.shape[0] // h_divider:, resized_screenshot_bgr.shape[1] - resized_screenshot_bgr.shape[1] // w_divider:]
    
    u_box_f = full_screenshot_bgr[:h // h_divider, :]
    l_box_f = full_screenshot_bgr[h // h_divider:, :w // w_divider]
    r_box_f = full_screenshot_bgr[h // h_divider:, w - w // w_divider:]

    # Split the sections into image lists for LED color extraction
    l_image_list = [l_box[n * l_box.shape[0] // (h_leds//led_span):(n + 1) * l_box.shape[0] // (h_leds//led_span)] for n in range((h_leds//led_span))]
    u_image_list = [u_box[:, n * u_box.shape[1] // w_leds:(n + 1) * u_box.shape[1] // w_leds] for n in range(w_leds)]
    r_image_list = [r_box[n * r_box.shape[0] // (h_leds//led_span):(n + 1) * r_box.shape[0] // (h_leds//led_span)] for n in range((h_leds//led_span))]
    
    # Extract colors from the divided sections
    l_colors = color_extraction(l_image_list, method, (h_leds//led_span), use_constant_color, constant_color)
    u_colors = color_extraction(u_image_list, method, w_leds, use_constant_color, constant_color)
    r_colors = color_extraction(r_image_list, method, (h_leds//led_span), use_constant_color, constant_color)

    # Create a black image to display the LED colors
    black_image = np.zeros((full_screenshot_bgr.shape[0], full_screenshot_bgr.shape[1], 3), dtype=np.uint8)
    black_image[0:u_box_f.shape[0], 0:u_box_f.shape[1]] = u_box_f
    black_image[u_box_f.shape[0]:black_image.shape[0], 0:l_box_f.shape[1]] = l_box_f
    black_image[u_box_f.shape[0]:black_image.shape[0], black_image.shape[1] - r_box_f.shape[1]:black_image.shape[1]] = r_box_f

    # Apply extracted colors to the black image
    for i, color in enumerate(l_colors):
        region = black_image[u_box_f.shape[0] + l_box_f.shape[0] - (i + 1) * l_box_f.shape[0] // (h_leds//led_span):
                             u_box_f.shape[0] + l_box_f.shape[0] - i * l_box_f.shape[0] // (h_leds//led_span),
                             l_box_f.shape[1]:l_box_f.shape[1] * 2]
        region[:] = color

    for i, color in enumerate(r_colors):
        region = black_image[u_box_f.shape[0] + (i) * r_box_f.shape[0] // (h_leds//led_span):
                             u_box_f.shape[0] + (i+1) * r_box_f.shape[0] // (h_leds//led_span),
                             black_image.shape[1] - 2* r_box_f.shape[1]:black_image.shape[1] - r_box_f.shape[1]]
        region[:] = color

    for i, color in enumerate(u_colors):
        region = black_image[u_box_f.shape[0]:2*u_box_f.shape[0],
                             i * u_box_f.shape[1] // w_leds:(i + 1) * u_box_f.shape[1] // w_leds]
        region[:] = color

    # Output execution time
    print(f"Execution Time: {time.time() - start_time:.4f} seconds")

    # Show the black image with applied colors
    cv2.imshow("Black Image", black_image)
    cv2.waitKey(1)

    l_color_string = color_string(l_colors)
    u_color_string = color_string(u_colors)
    r_color_string = color_string(r_colors)

    return l_color_string + u_color_string + r_color_string

# Run the function
while True:
    print(color_gen())


 