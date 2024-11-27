import tkinter as tk
from tkinter import messagebox, ttk
import threading
import cv2
import numpy as np
import serial
import time
from collections import Counter
import mss
import serial.tools.list_ports
import dxcam

# Global variable to control the running state of the serial communication thread
running = True

# Initialize the camera to capture the entire screen
cam = dxcam.create(output_idx=0, output_color="BGR")

# Function to extract colors based on a color extraction method
def color_extraction(image_list, color_func, led_number):
    return [color_func(image_list[n]) for n in range(led_number)]

# Function to reduce darkness in the image (helps enhance visibility)
def reduce_darkness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    mask = (s < 20) & (v < 220) | (v < 50) | ((10 <= h) & (h <= 50) & (s < 50))
    image[mask] = [0, 0, 0]
    return image

def average_color(image):
    """
    Returns average color of the image
    :param image:           NumPy array
    :return:                Average pixel value of a specified field
    """
    return np.average(image, axis=(1, 0))

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
def color_gen(led_span=1, h_leds=18, w_leds=32, h=1440, w=2560, method=dominant_color):
   
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
    l_colors = color_extraction(l_image_list, method, (h_leds//led_span))
    u_colors = color_extraction(u_image_list, method, w_leds)
    r_colors = color_extraction(r_image_list, method, (h_leds//led_span))

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


    # Configure the serial port
def serial_comm(port, baudrate, led_span=2, h_leds=18, w_leds=36, h=1440, w=2560, use_constant_color=False, constant_color=None, method='dominant'):
    
    """
    Exchanges data between device via serial port
    :param port:        ESP port
    :param baudrate:    ESP communication baudrate (115200 default - other might be unstable)
    :param scale:       parameter that specifies if color scale is reduced from 0-255 to 0-99
    :param              led_span: reduces the number of active WS2812B LEDS (1, 2, 3)
    :param              h_leds: number of leds on the side frame
    :param              w_leds: number of leds on the upper frame
    :param h:           number of rows of pixels for screen resolution
    :param w:           number of columns of pixels for screen resolution
    :return:            Nothing
    """
    config_bit1 = 0
    config_bit2 = 0

    ser = serial.Serial()
    ser.setDTR(False)
    ser.setRTS(False)
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    # ser.timeout = 0.001
    
    if method == 'average': 
        color_extract_mehod = average_color
    if method == 'dominant': 
        color_extract_mehod = dominant_color

    try:
        while running:  # Check if the thread should keep running
            start_time = time.time()
            config_bit1 = led_span

            if use_constant_color:
                channel1 = '0' + hex(constant_color[0]) if len(str(constant_color[0])) < 2 else channel1 = hex(constant_color[0])
                channel2 = '0' + hex(constant_color[1]) if len(str(constant_color[1])) < 2 else channel2 = hex(constant_color[1])
                channel3 = '0' + hex(constant_color[2]) if len(str(constant_color[2])) < 2 else channel3 = hex(constant_color[2])
                message = str(config_bit1) + str(config_bit2) + channel1 + channel2 + channel3 + '\n'
                ser.write(message.encode('utf-8'))
                break
            else:
                color_string = color_gen(led_span=led_span, h_leds=h_leds, w_leds=w_leds, h=h, w=w, method=color_extract_mehod)
                message = str(config_bit1) + str(config_bit2) + + color_string + '\n'
                ser.write(message.encode('utf-8'))
            print(message)
            time.sleep(0.01) if led_span == 1 else None
            end_time = time.time()
            print(f'Time elapsed: {end_time - start_time}')

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

def start_serial_comm(scale, led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method):
    global running
    ESP32_name = 'CH340'

    # Specify the baudrate for UART communication - 115200 most stable
    ESP32_baudrate = 115200
    # ESP32_baudrate = 230400
    # Find port of specified module
    ESP32_port = find_port(ESP32_name)

    # Start communication
    serial_comm(ESP32_port, ESP32_baudrate, scale, led_span, h_leds, w_leds, h, w) if ESP32_port else \
        print("Specified device not found!")
    if ESP32_port:
        running = True  # Reset the running flag to True
        serial_comm(ESP32_port, ESP32_baudrate, scale, led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method)
    else:
        print("Specified device not found!")

def start_thread(scale, led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method):
    global running
    threading.Thread(target=start_serial_comm, args=(scale, led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method), daemon=True).start()

def start_button_clicked():
    try:
        scale = int(scale_entry.get())
        led_span = int(led_span_entry.get())
        h_leds = int(h_leds_entry.get())
        w_leds = int(w_leds_entry.get())
        h = int(h_entry.get())
        w = int(w_entry.get())
        r = int(r_entry.get())
        g = int(g_entry.get())
        b = int(b_entry.get())
        constant_color = (r, g, b)
        method = extraction_method_combobox.get()
        use_constant_color = send_constant_color_var.get() == 1

        start_thread(scale, led_span, h_leds, w_leds, h, w, use_constant_color, constant_color, method)

    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

def stop_button_clicked():
    global running
    running = False  # Set the running flag to False to stop the thread

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

r_label = ttk.Label(main_frame, text="Blue Value (0-255):")
r_label.grid(row=6, column=0, sticky=tk.W)
r_entry = ttk.Entry(main_frame)
r_entry.grid(row=6, column=1)

g_label = ttk.Label(main_frame, text="Green Value (0-255):")
g_label.grid(row=7, column=0, sticky=tk.W)
g_entry = ttk.Entry(main_frame)
g_entry.grid(row=7, column=1)

b_label = ttk.Label(main_frame, text="Red Value (0-255):")
b_label.grid(row=8, column=0, sticky=tk.W)
b_entry = ttk.Entry(main_frame)
b_entry.grid(row=8, column=1)

# Method selection
extraction_method_label = ttk.Label(main_frame, text="Extraction Method:")
extraction_method_label.grid(row=9, column=0, sticky=tk.W)
extraction_method_combobox = ttk.Combobox(main_frame, values=["dominant", "average"])
extraction_method_combobox.grid(row=9, column=1)

# Checkbox for sending constant color
send_constant_color_var = tk.IntVar()
send_constant_color_check = ttk.Checkbutton(main_frame, text="Send Constant Color", variable=send_constant_color_var)
send_constant_color_check.grid(row=10, column=0, columnspan=2)

# Default values
DEFAULT_SCALE = 1
DEFAULT_LED_SPAN = 2
DEFAULT_H_LEDS = 18
DEFAULT_W_LEDS = 32
DEFAULT_H = 1440
DEFAULT_W = 2560
DEFAULT_R = 0
DEFAULT_G = 0
DEFAULT_B = 0
DEFAULT_METHOD = "dominant"

# Set default values in the entry fields
scale_entry.insert(0, DEFAULT_SCALE)
led_span_entry.insert(0, DEFAULT_LED_SPAN)
h_leds_entry.insert(0, DEFAULT_H_LEDS)
w_leds_entry.insert(0, DEFAULT_W_LEDS)
h_entry.insert(0, DEFAULT_H)
w_entry.insert(0, DEFAULT_W)
r_entry.insert(0, DEFAULT_R)
g_entry.insert(0, DEFAULT_G)
b_entry.insert(0, DEFAULT_B)
extraction_method_combobox.set(DEFAULT_METHOD)

start_button = ttk.Button(main_frame, text="Start", command=start_button_clicked)
start_button.grid(row=11, column=0, columnspan=2, pady=5)

stop_button = ttk.Button(main_frame, text="Stop", command=stop_button_clicked)
stop_button.grid(row=12, column=0, columnspan=2, pady=5)

root.mainloop()

