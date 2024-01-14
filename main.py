import cv2
import serial.tools.list_ports
from PIL import ImageGrab
import numpy as np
import serial
import time
from sklearn.cluster import KMeans
from collections import Counter
import mss

def dominant_color(screenshot, k=1):

    # Reshape the image to a list of pixels
    pixels = screenshot.reshape((-1, 3))

    # Convert the data type to float32
    pixels = np.float32(pixels)

    # Define criteria (epsilon and max_iter) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # If there's only one cluster, return its color
    if k == 1:
        dominant_color = centers[0]
    else:
        # Count occurrences of each label
        label_counts = Counter(labels.flatten())

        # Find the label with the maximum count
        dominant_label = max(label_counts, key=label_counts.get)

        # Find the corresponding color for the dominant label
        dominant_color = centers[dominant_label]

    return list(dominant_color)


def take_picture(box=(0, 0, 2560, 1440)):
    """ Takes screenshot """
    # bbox = (100, 100, 500, 500)
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab(box)), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Img', screenshot)
    # cv2.waitKey()
    return screenshot


def average_color(image):
    """
    Returns average color of the image
    :param image: NumPy array
    :return:
    """
    return np.average(image, axis=(1, 0))


def color_extraction(image_list, color_func, scale, led_number):
    """
    Returns string of extracted RGB colors from given list of screenshots
    :param color_func: function that finds preferable color on the screenshot e.g. average color
    :param image_list: list of screenshots
    :param scale: parameter that specifies if color scale is reduced from 0-255 to 0-99
    :param led_number: number of leds for given list of screenshots
    :return:
    """
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
    :param channel_value: color channel
    :return:
    """
    channel_max_value = 255
    channel_scaled_max_value = 99
    ret_val = channel_value*channel_scaled_max_value/channel_max_value
    if ret_val > 99:
        ret_val = 99

    return int(ret_val)


def color_string(color_list, scale):
    """
    Returns color string completed with redundant zeros.
    As a result output string for esp32 has the same length every time.
    :param color_list: list of colors extracted in previous steps
    :param scale: parameter that specifies if color scale is reduced from 0-255 to 0-99
    :return:
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
    :param scale:
    :param led_span:
    :param h_leds:
    :param w_leds:
    :param h:
    :param w:
    :return:
    """

    # For performance measurement purpose
    s_time = time.time()

    # Defining the number of active leds
    led_span = 1 if led_span not in [1, 2, 3] else led_span
    h_leds = int(h_leds / led_span)
    w_leds = int(w_leds / led_span)
    # Parameters needed for defining the screenshots' areas
    h_divider = 8
    w_divider = 16

    # Defining bounding boxes for screenshots
    u_box = {"top": 0, "left": 0, "width": w, "height": int(h / h_divider)}
    l_box = {"top": int(h / h_divider), "left": 0, "width": int(w / w_divider), "height": int(h - h / h_divider)}
    r_box = {"top": int(h / h_divider), "left": int(w - w / w_divider),
             "width": int(w / w_divider), "height": int(h - h / h_divider)}

    # Taking screenshot of the frame - frame is composed of 3 screenshots (2 side and 1 upper fields)
    with mss.mss() as camera:
        u_screenshot = np.array(camera.grab(u_box))
        l_screenshot = np.array(camera.grab(l_box))
        r_screenshot = np.array(camera.grab(r_box))
        u_screenshot = cv2.resize(u_screenshot, dsize=(int(w/4), int(h/h_divider/4)), interpolation=cv2.INTER_CUBIC)
        l_screenshot = cv2.resize(l_screenshot, dsize=(int(h/4), int(w/w_divider/4)), interpolation=cv2.INTER_CUBIC)
        r_screenshot = cv2.resize(r_screenshot, dsize=(int(h/4), int(w/w_divider/4)), interpolation=cv2.INTER_CUBIC)

    print(f"Time of taking screenshot: {time.time() - s_time}")

    # Screenshots' dimensions needed for their division
    h_r, w_r, c_r = r_screenshot.shape
    h_l, w_l, c_l = l_screenshot.shape
    h_u, w_u, c_u = u_screenshot.shape

    # Dividing screenshots into smaller ones and extracting average color of them
    l_image_list = [l_screenshot[int(n*h_l/h_leds):int(n*h_l/h_leds+h_l/h_leds):] for n in range(h_leds - 1, -1, -1)]
    l_colors = color_extraction(l_image_list, average_color, scale, h_leds)
    print(len(l_colors))
    u_image_list = [u_screenshot[:, int(n * w_u / w_leds):int(n * w_u / w_leds + w_u / w_leds)] for n in range(w_leds)]
    u_colors = color_extraction(u_image_list, average_color, scale, w_leds)

    r_image_list = [r_screenshot[int(n * h_r / h_leds):int(n * h_r / h_leds + h_r / h_leds), :] for n in range(h_leds)]
    r_colors = color_extraction(r_image_list, average_color, scale, h_leds)




    # Returning the extracted colors
    print(f"Time of color extraction: {time.time() - s_time}")
    return l_colors + u_colors + r_colors


def get_dominant_color(screenshot):

    unique, counts = np.unique(screenshot.reshape(-1, 3), axis=0, return_counts=True)
    return list(unique[np.argmax(counts)])


def serial_comm(port, baudrate):
    """ Exchanges data between device via serial port """

    # Configure the serial port
    ser = serial.Serial()

    # In my case DTR and RTS pins have to be set to false
    ser.setDTR(False)
    ser.setRTS(False)
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    scale = 1
    led_span = 1

    try:
        while True:
            # Get time for measurement
            start_time = time.time()

            # Send a message (2 first characters are meant for led strip configuration)
            message = str(scale) + str(led_span) + color_gen(scale=bool(scale), led_span=led_span) + '\n'
            ser.write(message.encode('utf-8'))
            print(message)

            # Wait for response (for debugging mostly - it slows down the whole process)
            # response = ser.readline().strip()
            # print("Response from ESP32:", response)
            time.sleep(0.01)
            # Print time
            end_time = time.time()
            print(f'Time elapsed: {end_time - start_time}')

    except KeyboardInterrupt:
        ser.close()


def find_port(device_name):
    """ Finds port of specified device """

    ports = list(serial.tools.list_ports.comports())

    for port, desc, hwid in ports:
        if device_name.lower() in desc.lower():
            return port


if __name__ == "__main__":

    # Specify the name of the esp module that appears in device manager and baudrate
    ESP32_name = 'CH340'
    ESP32_baudrate = 115200
    # ESP32_baudrate = 230400
    # ESP32_baudrate = 460800
    # ESP32_baudrate = 921600

    # Find port of specified module
    ESP32_port = find_port(ESP32_name)

    # Start communication
    serial_comm(ESP32_port, ESP32_baudrate) if ESP32_port else print("Specified device not found!")
