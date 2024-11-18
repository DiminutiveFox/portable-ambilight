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

# Global variable to control the running state of the serial communication thread
running = True

def is_vivid_color(pixel, threshold=30):
    r, g, b = pixel
    if max(abs(r - g), abs(g - b), abs(b - r)) > threshold:
        return True
    return False

def dominant_color(screenshot, k=5):
    
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

def average_color(image):
    """
    Returns average color of the image
    :param image:           NumPy array
    :return:                Average pixel value of a specified field
    """
    return np.average(image, axis=(1, 0))

def color_extraction(image_list, color_func, scale, led_number, use_constant_color=False, constant_color=None):
    if use_constant_color and constant_color:
        color_list = [constant_color for _ in range(led_number)]
    else:
        color_list = [color_func(image_list[n]) for n in range(led_number)]
    if scale:
        color_list = [[str(scale_channel(channels[2])), str(scale_channel(channels[1])),
                       str(scale_channel(channels[0]))] for channels in color_list]
    else:
        color_list = [[str(int(channels[2])), str(int(channels[1])), str(int(channels[0]))] for channels in color_list]

    return color_string(color_list, scale)

def scale_channel(channel_value):
    """
    Scales down channel value from range 0-255 to 0-99
    :param      channel_value: color channel
    :return:    Scaled channel value
    """
    channel_max_value = 255
    channel_scaled_max_value = 99
    ret_val = channel_value * channel_scaled_max_value / channel_max_value
    if ret_val > 99:
        ret_val = 99
    return int(ret_val)

def color_string(color_list, scale):
    """
    Returns color string completed with redundant zeros.
    As a result output string for esp32 has the same length every time.
    :param color_list:  list of colors extracted in previous steps
    :param scale:       parameter that specifies if color scale is reduced from 0-255 to 0-99
    :return:            Color string
    """

    channel_length = 2 if scale else 3
    for color in color_list:
        while len(color[0]) < channel_length:
            color[0] = '0' + color[0]
        while len(color[1]) < channel_length:
            color[1] = '0' + color[1]
        while len(color[2]) < channel_length:
            color[2] = '0' + color[2]
    return ''.join([''.join(sublist) for sublist in color_list])


def color_gen(scale=True, led_span=1, h_leds=18, w_leds=36, h=1440, w=2560):
    """
    Returns color spectrum of a display's frame
    :param scale:       parameter that specifies if color scale is reduced from 0-255 to 0-99
    :param led_span:    reduces the number of active WS2812B LEDS (1, 2, 3)
    :param h_leds:      number of leds on side frame
    :param w_leds:      number of leds on upper frame
    :param h:           number of rows of pixels for screen resolution
    :param w:           number of columns of pixels for screen resolution
    :return:            Color string
    """

    # For performance measurement purpose
def color_gen(scale=True, led_span=1, h_leds=18, w_leds=36, h=1440, w=2560, use_constant_color=False, constant_color=None, method=average_color):
    s_time = time.time()
    led_span = 1 if led_span not in [1, 2, 3] else led_span
    h_leds = int(h_leds / led_span)
    w_leds = int(w_leds / led_span)
    h_divider = 8
    w_divider = 16

    u_box = {"top": 0, "left": 0, "width": w, "height": int(h / h_divider)}
    l_box = {"top": int(h / h_divider), "left": 0, "width": int(w / w_divider), "height": int(h - h / h_divider)}
    r_box = {"top": int(h / h_divider), "left": int(w - w / w_divider),
             "width": int(w / w_divider), "height": int(h - h / h_divider)}

    with mss.mss() as camera:
        u_screenshot = np.array(camera.grab(u_box))
        l_screenshot = np.array(camera.grab(l_box))
        r_screenshot = np.array(camera.grab(r_box))
        u_screenshot = cv2.resize(u_screenshot, dsize=(int(w/4), int(h/h_divider/4)), interpolation=cv2.INTER_NEAREST)
        l_screenshot = cv2.resize(l_screenshot, dsize=(int(h/4), int(w/w_divider/4)), interpolation=cv2.INTER_NEAREST)
        r_screenshot = cv2.resize(r_screenshot, dsize=(int(h/4), int(w/w_divider/4)), interpolation=cv2.INTER_NEAREST)

    print(f"Time of taking screenshot: {time.time() - s_time}")

    h_r, w_r, c_r = r_screenshot.shape
    h_l, w_l, c_l = l_screenshot.shape
    h_u, w_u, c_u = u_screenshot.shape

    l_image_list = [l_screenshot[int(n*h_l/h_leds):int(n*h_l/h_leds+h_l/h_leds):] for n in range(h_leds - 1, -1, -1)]
    l_colors = color_extraction(l_image_list, method, scale, h_leds, use_constant_color, constant_color)

    u_image_list = [u_screenshot[:, int(n * w_u / w_leds):int(n * w_u / w_leds + w_u / w_leds)] for n in range(w_leds)]
    u_colors = color_extraction(u_image_list, method, scale, w_leds, use_constant_color, constant_color)

    r_image_list = [r_screenshot[int(n * h_r / h_leds):int(n * h_r / h_leds + h_r / h_leds), :] for n in range(h_leds)]
    r_colors = color_extraction(r_image_list, method, scale, h_leds, use_constant_color, constant_color)

    print(f"Time of color extraction: {time.time() - s_time}")
    return l_colors + u_colors + r_colors

    # Configure the serial port
def serial_comm(port, baudrate, scale=1, led_span=2, h_leds=18, w_leds=36, h=1440, w=2560, use_constant_color=False, constant_color=None, method='average'):
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
    ser = serial.Serial()
    ser.setDTR(False)
    ser.setRTS(False)
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    # ser.timeout = 0.001
    
    color_extract_mehod = average_color if method == 'average' else dominant_color

    try:
        while running:  # Check if the thread should keep running
            start_time = time.time()

            # Send a message (2 first characters are meant for led strip configuration)
            # message = str(scale) + str(led_span) + color_gen(scale=bool(scale), led_span=led_span,
            #                                                  h_leds=h_leds, w_leds=w_leds, h=h, w=w) + '\n'
            color_str = f"{constant_color[0]},{constant_color[1]},{constant_color[2]}" if use_constant_color and constant_color else "0,0,0"
            color_string = color_gen(scale=bool(scale), led_span=led_span,
                                                             h_leds=h_leds, w_leds=w_leds, h=h, w=w, use_constant_color=use_constant_color, constant_color=constant_color, method=color_extract_mehod)
            message = str(scale) + str(led_span) + color_string + '\n'
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
extraction_method_combobox = ttk.Combobox(main_frame, values=["average", "dominant"])
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
DEFAULT_METHOD = "average"

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

