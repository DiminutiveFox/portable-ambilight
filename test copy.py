import numpy as np
import cv2
import dxcam
import time

# Initialize the camera to capture the entire screen
cam = dxcam.create(output_idx=0, output_color="BGR")

def hsv_to_rgb(h, s, v):
    # Special case for the bin with V < 64
    if v == 0:
        return np.array([0, 0, 0])
    
    # Convert a single HSV value to RGB
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return rgb / 255.0  # Normalize to [0, 1] for matplotlib

def create_hsv_histogram(hsv_image):

    # Separate the pixels with V < 64
    low_v_mask = hsv_image[:, :, 2] < 64
    low_v_pixels = hsv_image[low_v_mask]
    high_v_pixels = hsv_image[~low_v_mask]
    
    # Define bin edges for H, S, and V channels for the remaining pixels
    h_bins = np.linspace(0, 179, 17)  # 16 bins for Hue (0-178)
    s_bins = np.linspace(0, 256, 5)   # 4 bins for Saturation (0-255)
    v_bins = np.linspace(64, 256, 5)  # 4 bins for Value (64-255)
    
    # Compute the 3D histogram for pixels with V >= 64
    high_v_hist, _ = np.histogramdd(
        high_v_pixels.reshape(-1, 3),
        bins=(h_bins, s_bins, v_bins)
    )
    
    # Combine the histograms
    hist = np.zeros((17, 4, 4))  # 17 H bins to include the separate bin for low V
    hist[0, 0, 0] = low_v_pixels.shape[0]  # All low V pixels go into the first bin
    hist[1:, :, :] = high_v_hist
    
    return hist, (h_bins, s_bins, v_bins), hsv_image.shape[0] * hsv_image.shape[1]

def get_top_n_bins(hist, bin_edges, total_pixels, top_n=5):
    # Flatten the histogram
    hist_flat = hist.flatten()
    
    # Get the indices of the top n bins
    top_indices = np.argsort(hist_flat)[-top_n:][::-1]
    
    # Get the corresponding frequencies
    top_frequencies = hist_flat[top_indices]
    
    # Convert flat indices to multi-dimensional indices
    top_bins = np.unravel_index(top_indices, hist.shape)
    
    # Map bin indices to bin centers
    h_bin_centers = np.concatenate(([0], (bin_edges[0][:-1] + bin_edges[0][1:]) / 2))
    s_bin_centers = (bin_edges[1][:-1] + bin_edges[1][1:]) / 2
    v_bin_centers = np.concatenate(([0], (bin_edges[2][:-1] + bin_edges[2][1:]) / 2))
    
    top_hsv_values = [(h_bin_centers[h], s_bin_centers[s], v_bin_centers[v]) for h, s, v in zip(*top_bins)]
    
    # Calculate the percentage of total pixels
    top_percentages = (top_frequencies / total_pixels) * 100
    
    return top_frequencies, top_hsv_values, top_percentages

def color_extraction(image_list, color_func):
    return [color_func(image_list[n]) for n in range(len(image_list))]

def histogram_selection(image):
    hist, bin_edges, total_pixels = create_hsv_histogram(image)
    # top_frequencies, top_hsv_values, top_percentages = get_top_n_bins(hist, bin_edges, total_pixels, top_n=5)
    dominant_bin_position, dominant_color = find_dominant_bin(hist, total_pixels, bin_edges)

    return dominant_color

def hsv_to_rgb_dominant(h, s, v):
    # Convert a single HSV value to RGB
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return rgb

def find_dominant_bin(hist, total_pixels, bin_edges):
    # Flatten the histogram
    hist_flat = hist.flatten()
    
    # Get the indices of the bins sorted by frequency (highest to lowest)
    sorted_indices = np.argsort(hist_flat)[::-1]
    
    # Get the positions of the top two bins
    most_frequent_bin_idx = sorted_indices[0]
    second_most_frequent_bin_idx = sorted_indices[1]
    
    # Get the frequency of the most frequent and second most frequent bins
    most_frequent_count = hist_flat[most_frequent_bin_idx]
    second_most_frequent_count = hist_flat[second_most_frequent_bin_idx]
    
    # Get the indices of the bins corresponding to the black pixels (V < 64)
    black_bin_idx = 0  # Black pixels are assigned to the first bin
    
    # Get the frequency of black pixels (V < 64)
    black_count = hist_flat[black_bin_idx]
    
    # Calculate the percentage of black pixels in the image
    black_percentage = (black_count / total_pixels) * 100
    
    # Map bin indices to bin centers
    h_bin_centers = np.concatenate(([0], (bin_edges[0][:-1] + bin_edges[0][1:]) / 2)).astype(int)
    s_bin_centers = ((bin_edges[1][:-1] + bin_edges[1][1:]) / 2).astype(int)
    v_bin_centers = np.concatenate(([0], (bin_edges[2][:-1] + bin_edges[2][1:]) / 2)).astype(int)
    
    # Get the HSV values of the most frequent and second most frequent bins
    def get_hsv_from_idx(bin_idx):
        h, s, v = np.unravel_index(bin_idx, hist.shape)
        return h_bin_centers[h], s_bin_centers[s], v_bin_centers[v]
    
    # Get the RGB value of the most frequent bin
    most_frequent_hsv = get_hsv_from_idx(most_frequent_bin_idx)
    most_frequent_rgb = hsv_to_rgb_dominant(*most_frequent_hsv)
    
    # Get the RGB value of the second most frequent bin
    second_most_frequent_hsv = get_hsv_from_idx(second_most_frequent_bin_idx)
    second_most_frequent_rgb = hsv_to_rgb_dominant(*second_most_frequent_hsv)
    
    # Check the conditions
    if most_frequent_count == black_count and black_percentage < 100:
        # If the most frequent is black and covers less than 95%, return second most frequent bin
        return second_most_frequent_bin_idx, second_most_frequent_rgb
    else:
        # Otherwise, return the position of the black pixel bin
        return black_bin_idx, hsv_to_rgb_dominant(*get_hsv_from_idx(black_bin_idx))



# Function to process a screenshot and divide it into sections
def process_screenshot(scale_factor=0.5):
    # Grab the entire screen using dxcam
    screenshot = cam.grab()
    
    # If no screenshot is captured, return None
    if screenshot is None:
        return None, None
    
    # Resize and reduce darkness for better processing
    screenshot_resized = cv2.cvtColor(cv2.resize(screenshot, (0, 0), fx=scale_factor, fy=scale_factor), cv2.COLOR_RGB2HSV)
    
    return screenshot_resized, screenshot



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
def color_gen(led_span=1, h_leds=18, w_leds=32, h=1440, w=2560, use_constant_color=False, constant_color=None, method=histogram_selection):
   
    # Grab and process the full screen image
    resized_screenshot_bgr, full_screenshot_bgr = process_screenshot(1)
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
    l_image_list = [l_box[n * l_box.shape[0] // (h_leds//led_span):(n + 1) * l_box.shape[0] // (h_leds//led_span)] for n in range((h_leds//led_span-1), -1, -1)]
    u_image_list = [u_box[:, n * u_box.shape[1] // w_leds:(n + 1) * u_box.shape[1] // w_leds] for n in range(w_leds)]
    r_image_list = [r_box[n * r_box.shape[0] // (h_leds//led_span):(n + 1) * r_box.shape[0] // (h_leds//led_span)] for n in range((h_leds//led_span))]
    
    # print(prominent_color(l_image_list[0]))
    # cv2.waitKey()
    # Extract colors from the divided sections
    l_colors = color_extraction(l_image_list, lambda img: method(img))
    u_colors = color_extraction(u_image_list, lambda img: method(img))
    r_colors = color_extraction(r_image_list, lambda img: method(img))

    # Create a black image to display the LED colors
    black_image = np.zeros((full_screenshot_bgr.shape[0], full_screenshot_bgr.shape[1], 3), dtype=np.uint8)
    black_image[0:u_box_f.shape[0], 0:u_box_f.shape[1]] = u_box_f
    black_image[u_box_f.shape[0]:black_image.shape[0], 0:l_box_f.shape[1]] = l_box_f
    black_image[u_box_f.shape[0]:black_image.shape[0], black_image.shape[1] - r_box_f.shape[1]:black_image.shape[1]] = r_box_f
    
    black_image_l = np.zeros((l_box_f.shape[0], l_box_f.shape[1], 3), dtype=np.uint8)
    black_image_u = np.zeros((u_box_f.shape[0], u_box_f.shape[1], 3), dtype=np.uint8)
    black_image_r = np.zeros((r_box_f.shape[0], r_box_f.shape[1], 3), dtype=np.uint8)

    grad = create_gradient_image(l_colors[::-1], l_box_f.shape[0], l_box_f.shape[1])
    cv2.imshow("gradient", grad)
    # cv2.waitKey()
    # Apply extracted colors to the black image
    for i, color in enumerate(l_colors):
        region = black_image[u_box_f.shape[0] + l_box_f.shape[0] - (i + 1) * l_box_f.shape[0] // (h_leds//led_span):
                             u_box_f.shape[0] + l_box_f.shape[0] - i * l_box_f.shape[0] // (h_leds//led_span),
                             l_box_f.shape[1]:l_box_f.shape[1] * 2]
        
        black_image_l[black_image_l.shape[0] - (i + 1) * black_image_l.shape[0] // (h_leds//led_span):
                             black_image_l.shape[0] - i * black_image_l.shape[0] // (h_leds//led_span),
                             0:black_image_l.shape[1]] = color
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
    cv2.imshow("Black Image", black_image_l)
    cv2.waitKey()

    l_color_string = color_string(l_colors)
    u_color_string = color_string(u_colors)
    r_color_string = color_string(r_colors)

    return l_color_string + u_color_string + r_color_string


def create_gradient_image(colors, image_height, image_width):
    gradient_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
    num_colors = len(colors)
    section_height = image_height // num_colors
    
    for i in range(num_colors):
        start_color = colors[i]
        end_color = colors[i + 1] if i + 1 < num_colors else colors[i]
        
        for j in range(section_height):
            alpha = j / section_height
            color = (1 - alpha) * np.array(start_color) + alpha * np.array(end_color)
            gradient_image[i * section_height + j, :] = color

    return gradient_image

# Run the function
while True:
    print(color_gen())
