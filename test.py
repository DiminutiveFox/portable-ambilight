import tkinter as tk
from tkinter import messagebox, ttk
import threading
import cv2
import numpy as np
import serial
import time
from collections import Counter
import serial.tools.list_ports
import dxcam

# Global variable to control the running state of the serial communication thread
running = True

# Initialize the camera to capture the entire screen
cam = dxcam.create(output_idx=0, output_color="RGB")

# Function to extract colors based on a color extraction method
def hsv_to_rgb(color):
    h, s, v = color
    # Special case for the bin with V < 64
    if v == 0:
        return np.array([0, 0, 0])
    
    # Convert a single HSV value to RGB
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return [int(rgb[0]),int(rgb[1]),int(rgb[2])] # Normalize to [0, 1] for matplotlib

# Function to reduce darkness in the image (helps enhance visibility)
def reduce_darkness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    mask = (s < 20) & (v < 220) | (v < 50) | ((10 <= h) & (h <= 50) & (s < 50))
    image[mask] = [0, 0, 0]
    return image

# Function to get the dominant color from an image using k-means clustering
def load_predefined_centers(filepath):
    # Load the predefined centers from a file
    centers = []
    with open(filepath, 'r') as file:
        for line in file:
            center = list(map(int, line.strip().split(',')))
            centers.append(center)
    return np.array(centers, dtype=np.float32)

centers = load_predefined_centers("256_color_palette.txt")

def dominant_color(screenshot, centers=centers, k=3):
    # Load predefined centers
    
    
    # Reshape image pixels
    pixels = screenshot.reshape((-1, 3)).astype(np.float32)
    if pixels.shape[0] == 0:  # If the image is empty, return a default color
        return [0, 0, 0]
    
    # Ensure we have enough predefined centers
    if centers.shape[0] < k:
        raise ValueError("Not enough predefined centers for the given k")
    initial_centers = centers[:k]
    
    # Assign each pixel to the nearest initial center to create the initial labels
    distances = np.linalg.norm(pixels[:, np.newaxis] - initial_centers, axis=2)
    initial_labels = np.argmin(distances, axis=1).astype(np.int32)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, initial_labels, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS)
    
    centers = np.uint8(centers)
    return hsv_to_rgb(max(centers, key=lambda c: np.std(c)))




def create_hsv_histogram(hsv_image):
    # Read the image
    

    # Define bin edges for H, S, and V channels
    h_bins = np.linspace(0, 179, 17)  # 16 bins for Hue (0-178)
    s_bins = np.linspace(0, 256, 5)   # 4 bins for Saturation (0-255)
    v_bins = np.linspace(0, 256, 5)   # 4 bins for Value (Brightness) (0-255)
    
    # Compute the 3D histogram
    hist, edges = np.histogramdd(
        hsv_image.reshape(-1, 3),
        bins=(h_bins, s_bins, v_bins)
    )
    
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
def process_screenshot(scale_factor=0.1):
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
def color_gen(led_span=1, h_leds=18, w_leds=32, h=1440, w=2560, method=dominant_color):
   
    # Grab and process the full screen image
    resized_screenshot_bgr, full_screenshot_bgr = process_screenshot()
    start_time = time.time()
    
    # If no screenshot is captured, exit
    if resized_screenshot_bgr is None:
        print("Failed to capture screenshot")
        return None
    
    # Divide the image into 3 sections: upper, left, right
    h_divider = 8
    w_divider = 16
    if method == dominant_color:
        print("dominant")
    u_box = resized_screenshot_bgr[:resized_screenshot_bgr.shape[0] // h_divider, :]
    l_box = resized_screenshot_bgr[resized_screenshot_bgr.shape[0] // h_divider:, :resized_screenshot_bgr.shape[1] // w_divider]
    r_box = resized_screenshot_bgr[resized_screenshot_bgr.shape[0] // h_divider:, resized_screenshot_bgr.shape[1] - resized_screenshot_bgr.shape[1] // w_divider:]
    
    # u_box_f = full_screenshot_bgr[:h // h_divider, :]
    # l_box_f = full_screenshot_bgr[h // h_divider:, :w // w_divider]
    # r_box_f = full_screenshot_bgr[h // h_divider:, w - w // w_divider:]

    # Split the sections into image lists for LED color extraction
    l_image_list = [l_box[n * l_box.shape[0] // (h_leds//led_span):(n + 1) * l_box.shape[0] // (h_leds//led_span)] for n in range((h_leds//led_span))]
    u_image_list = [u_box[:, n * u_box.shape[1] // (w_leds//led_span):(n + 1) * u_box.shape[1] // (w_leds//led_span)] for n in range(w_leds//led_span)]
    r_image_list = [r_box[n * r_box.shape[0] // (h_leds//led_span):(n + 1) * r_box.shape[0] // (h_leds//led_span)] for n in range((h_leds//led_span))]
    
    # Extract colors from the divided sections
    l_colors = color_extraction(l_image_list, dominant_color)
    u_colors = color_extraction(u_image_list, dominant_color)
    r_colors = color_extraction(r_image_list, dominant_color)

    # # Create a black image to display the LED colors
    # black_image = np.zeros((full_screenshot_bgr.shape[0], full_screenshot_bgr.shape[1], 3), dtype=np.uint8)
    # black_image[0:u_box_f.shape[0], 0:u_box_f.shape[1]] = u_box_f
    # black_image[u_box_f.shape[0]:black_image.shape[0], 0:l_box_f.shape[1]] = l_box_f
    # black_image[u_box_f.shape[0]:black_image.shape[0], black_image.shape[1] - r_box_f.shape[1]:black_image.shape[1]] = r_box_f

    # # Apply extracted colors to the black image
    # for i, color in enumerate(l_colors):
    #     region = black_image[u_box_f.shape[0] + l_box_f.shape[0] - (i + 1) * l_box_f.shape[0] // (h_leds//led_span):
    #                          u_box_f.shape[0] + l_box_f.shape[0] - i * l_box_f.shape[0] // (h_leds//led_span),
    #                          l_box_f.shape[1]:l_box_f.shape[1] * 2]
    #     region[:] = color

    # for i, color in enumerate(r_colors):
    #     region = black_image[u_box_f.shape[0] + (i) * r_box_f.shape[0] // (h_leds//led_span):
    #                          u_box_f.shape[0] + (i+1) * r_box_f.shape[0] // (h_leds//led_span),
    #                          black_image.shape[1] - 2* r_box_f.shape[1]:black_image.shape[1] - r_box_f.shape[1]]
    #     region[:] = color

    # for i, color in enumerate(u_colors):
    #     region = black_image[u_box_f.shape[0]:2*u_box_f.shape[0],
    #                          i * u_box_f.shape[1] // w_leds:(i + 1) * u_box_f.shape[1] // w_leds]
    #     region[:] = color

    # # Output execution time
    # print(f"Execution Time: {time.time() - start_time:.4f} seconds")

    # # Show the black image with applied colors
    # cv2.imshow("Black Image", black_image)
    # cv2.waitKey(1)

    l_color_string = color_string(l_colors)
    u_color_string = color_string(u_colors)
    r_color_string = color_string(r_colors)
    print(l_color_string)
    return l_color_string + u_color_string + r_color_string


    # Configure the serial port
def serial_comm(port, baudrate, led_span=2, h_leds=18, w_leds=32, h=1440, w=2560, use_constant_color=False, constant_color=None, method='histogram'):
    config_bit1 = 0
    config_bit2 = 0

    ser = serial.Serial()
    ser.setDTR(False)
    ser.setRTS(False)
    ser.port = port
    ser.baudrate = baudrate
    ser.open()

    if method == 'dominant': 
        color_extract_method = dominant_color
    if method == 'histogram': 
        color_extract_method = histogram_selection



    try:
        while running:  # Check if the thread should keep running
            start_time = time.time()
            config_bit1 = led_span

            if send_constant_color_var.get() == 1:
                if constant_color is None:
                    raise ValueError("Constant color must be provided when 'Send Constant Color' is enabled.")
                print("Sending constant color...")
                constant_color = hsv_to_rgb([int(hue_slider.get()), int(saturation_slider.get()), int(brightness_slider.get())])
                config_bit2 = 1
                config_bit1 = 1
                color = f"{constant_color[0]:02x}{constant_color[1]:02x}{constant_color[2]:02x}" * (w_leds + 2*h_leds)
                message = f"{config_bit1}{config_bit2}{color}\n"
                ser.write(message.encode('utf-8'))
                
                response = ser.readline().decode('utf-8')
                print(f"Sent: {message.strip()}")
                print(f"Received: {response.strip()}")

                if update_hue_color_var.get() == 1:
                    update_hue_value()
                time.sleep(int(sec_entry.get()))
                
            else:
                color_string = color_gen(led_span=led_span, h_leds=h_leds, w_leds=w_leds, h=h, w=w, method=color_extract_method)
                if color_string:
                    message = f"{config_bit1}{config_bit2}{color_string}\n"
                    ser.write(message.encode('utf-8'))

                response = ser.readline().decode('utf-8')
                print(f"Sent: {message.strip()}")
                print(f"Received: {response.strip()}")
            
            time.sleep(0.02)
            end_time = time.time()
            print(f'Time elapsed: {end_time - start_time}')

    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        ser.close()
    finally:
        ser.close()  # Ensure the serial connection is closed when exiting


def find_port(device_name):
    """
    Finds port of specified device
    :param device_name:     Device's name (can be found in device manager)
    :return:                Port's name
    """

    ports = list(serial.tools.list_ports.comports())
    for port, desc, hwid in ports:
        if device_name.lower() in desc.lower():
            return port

def start_serial_comm(led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method):
    global running
    ESP32_name = 'CH340'

    # Specify the baudrate for UART communication - 115200 most stable
    ESP32_baudrate = 115200
    # ESP32_baudrate = 230400
    # Find port of specified module
    ESP32_port = find_port(ESP32_name)

    # Start communication
    serial_comm(ESP32_port, ESP32_baudrate, led_span, h_leds, w_leds, h, w) if ESP32_port else \
        print("Specified device not found!")
    if ESP32_port:
        running = True  # Reset the running flag to True
        serial_comm(ESP32_port, ESP32_baudrate, led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method)
    else:
        print("Specified device not found!")

def start_thread(led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method):
    global running
    threading.Thread(target=start_serial_comm, args=(led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method), daemon=True).start()

def start_button_clicked():
    try:
        led_span = int(led_span_entry.get())
        h_leds = int(h_leds_entry.get())
        w_leds = int(w_leds_entry.get())
        h = int(h_entry.get())
        w = int(w_entry.get())
        constant_color = [255,255,255]
        method = extraction_method_combobox.get()
        use_constant_color = send_constant_color_var.get() == 1

        start_thread(led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method)

    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

def stop_button_clicked():
    global running
    running = False  # Set the running flag to False to stop the thread


def update_color_display(hue, saturation, brightness):
    # Convert hue, saturation, brightness to RGB
    rgb = hsv_to_rgb([hue, saturation, brightness])  # You may need to adjust how you convert this
    r, g, b = rgb
    # Ensure the RGB values are within 0-255 and convert to hex
    color_hex = f"#{r:02x}{g:02x}{b:02x}".upper()
    
    # Update the background color of the color box
    color_box.config(background=color_hex)

def update_sliders_state():
    """Enable/Disable sliders and dropdown based on the 'Send Constant Color' checkbox state."""
    if send_constant_color_var.get() == 1:
        hue_slider.config(state=tk.NORMAL)
        saturation_slider.config(state=tk.NORMAL)
        brightness_slider.config(state=tk.NORMAL)
        extraction_method_combobox.config(state=tk.DISABLED)
    else:
        hue_slider.config(state=tk.DISABLED)
        saturation_slider.config(state=tk.DISABLED)
        brightness_slider.config(state=tk.DISABLED)
        extraction_method_combobox.config(state=tk.NORMAL)


def update_hue_value():
    current_hue = int(hue_slider.get())
    new_hue = current_hue + 1
    if new_hue == 181:
        hue_slider.config(value=0)
    else:
        hue_slider.config(value=new_hue)
    update_color_display(hue_slider.get(), saturation_slider.get(), brightness_slider.get())





root = tk.Tk()
root.title("Color Extraction")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create and place the input fields and labels
scale_label = ttk.Label(main_frame, text="Scale (0 or 1):")
scale_label.grid(row=0, column=0, sticky=tk.W)
scale_entry = ttk.Entry(main_frame)
scale_entry.grid(row=0, column=1)

led_span_label = ttk.Label(main_frame, text="LED Span (1, 2, or 3):")
led_span_label.grid(row=1, column=0, sticky=tk.W)
led_span_entry = ttk.Entry(main_frame)
led_span_entry.grid(row=1, column=1)

h_leds_label = ttk.Label(main_frame, text="Horizontal LEDs:")
h_leds_label.grid(row=2, column=0, sticky=tk.W)
h_leds_entry = ttk.Entry(main_frame)
h_leds_entry.grid(row=2, column=1)

w_leds_label = ttk.Label(main_frame, text="Vertical LEDs:")
w_leds_label.grid(row=3, column=0, sticky=tk.W)
w_leds_entry = ttk.Entry(main_frame)
w_leds_entry.grid(row=3, column=1)

h_label = ttk.Label(main_frame, text="Screen Height:")
h_label.grid(row=4, column=0, sticky=tk.W)
h_entry = ttk.Entry(main_frame)
h_entry.grid(row=4, column=1)

w_label = ttk.Label(main_frame, text="Screen Width:")
w_label.grid(row=5, column=0, sticky=tk.W)
w_entry = ttk.Entry(main_frame)
w_entry.grid(row=5, column=1)



# Hue Slider
hue_label = ttk.Label(main_frame, text="Hue (0-179):")
hue_label.grid(row=8, column=0, sticky=tk.W)
hue_slider = ttk.Scale(main_frame, from_=0, to=179, orient="horizontal", command=lambda val: update_color_display(round(float(val)), saturation_slider.get(), brightness_slider.get()))
hue_slider.grid(row=8, column=1, sticky=(tk.W, tk.E))

# Saturation Slider
saturation_label = ttk.Label(main_frame, text="Saturation (0-255):")
saturation_label.grid(row=9, column=0, sticky=tk.W)
saturation_slider = ttk.Scale(main_frame, from_=0, to=255, orient="horizontal", command=lambda val: update_color_display(hue_slider.get(), round(float(val)), brightness_slider.get()))
saturation_slider.grid(row=9, column=1, sticky=(tk.W, tk.E))

# Brightness Slider
brightness_label = ttk.Label(main_frame, text="Brightness (0-255):")
brightness_label.grid(row=10, column=0, sticky=tk.W)
brightness_slider = ttk.Scale(main_frame, from_=0, to=255, orient="horizontal", command=lambda val: update_color_display(hue_slider.get(), saturation_slider.get(), round(float(val))))
brightness_slider.grid(row=10, column=1, sticky=(tk.W, tk.E))

# Color Display Box
color_box_label = ttk.Label(main_frame, text="Color Display:")
color_box_label.grid(row=11, column=0, sticky=tk.W)
color_box = ttk.Label(main_frame, width=20, relief="solid", background="#000000", padding=(5, 5))
color_box.grid(row=11, column=1)

# Method selection
extraction_method_label = ttk.Label(main_frame, text="Extraction Method:")
extraction_method_label.grid(row=6, column=0, sticky=tk.W)
extraction_method_combobox = ttk.Combobox(main_frame, values=["dominant", "average", "histogram"])
extraction_method_combobox.grid(row=6, column=1)

# Checkbox for sending constant color
send_constant_color_var = tk.IntVar()
send_constant_color_check = ttk.Checkbutton(main_frame, text="Send Constant Color", variable=send_constant_color_var)
send_constant_color_check.grid(row=7, column=0, columnspan=1, sticky="w")

# Checkbox for updating hue and sending periodically
update_hue_color_var = tk.IntVar()
update_hue_color_check = ttk.Checkbutton(main_frame, text="Update hue value", variable=update_hue_color_var)
update_hue_color_check.grid(row=12, column=0, columnspan=1, sticky="w")

sec_label = ttk.Label(main_frame, text="Seconds:")
sec_label.grid(row=12, column=1, sticky=tk.W)
sec_entry = ttk.Entry(main_frame, width=6)
sec_entry.grid(row=12, column=1)


start_button = ttk.Button(main_frame, text="Start", command=start_button_clicked)
start_button.grid(row=13, column=0, columnspan=2, pady=5)

stop_button = ttk.Button(main_frame, text="Stop", command=stop_button_clicked)
stop_button.grid(row=14, column=0, columnspan=2, pady=5)

# Default values
DEFAULT_SCALE = 1
DEFAULT_LED_SPAN = 2
DEFAULT_H_LEDS = 18
DEFAULT_W_LEDS = 32
DEFAULT_H = 1440
DEFAULT_W = 2560
DEFAULT_Hue = 0
DEFAULT_Sat = 0
DEFAULT_Val = 0
DEFAULT_SEC = 1
DEFAULT_METHOD = "histogram"

# Set default values in the entry fields
scale_entry.insert(0, DEFAULT_SCALE)
led_span_entry.insert(0, DEFAULT_LED_SPAN)
h_leds_entry.insert(0, DEFAULT_H_LEDS)
w_leds_entry.insert(0, DEFAULT_W_LEDS)
h_entry.insert(0, DEFAULT_H)
w_entry.insert(0, DEFAULT_W)
sec_entry.insert(0, DEFAULT_SEC)
extraction_method_combobox.set(DEFAULT_METHOD)

root.mainloop()

