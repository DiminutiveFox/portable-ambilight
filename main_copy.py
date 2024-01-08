import cv2
import serial.tools.list_ports
from PIL import ImageGrab
import numpy as np
from screeninfo import get_monitors
import serial
import time
import json
import random
import colorthief
from sklearn.cluster import KMeans
from collections import Counter
import dxcam
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

def color_gen2(h_LEDs=18, w_LEDs=36):
    """ Returns color spectrum of a display's frame """

    left_pixels = 0  # Offset for LEDs on the left frame
    upper_pixels = 18  # Offset for LEDs on the upper frame
    right_pixels = 54  # Offset for LEDs on the right frame

    # Getting the screen resolution
    w, h = get_monitors()[0].width, get_monitors()[0].height

    # Defining bounding boxes
    l_box = [int(w / w_LEDs), int(h / h_LEDs), int(1.5 * w / w_LEDs), int(h)]
    u_box = [int(2 * w / w_LEDs), int(h / h_LEDs), int(w - 2 * w / w_LEDs), int(1.5 * h / h_LEDs)]
    r_box = [int(w - 2 * w / w_LEDs), int(h / h_LEDs), int(w - 1.5 * w / w_LEDs), int(h)]

    # Taking screenshot of the frame - frame is composed of 3 screenshots (2 side and 1 upper fields)
    r_screenshot = take_picture(r_box)
    u_screenshot = take_picture(u_box)
    l_screenshot = take_picture(l_box)

    h_r, w_r, c_r = r_screenshot.shape
    h_l, w_l, c_l = l_screenshot.shape
    h_u, w_u, c_u = u_screenshot.shape

    # Dividing screenshots to 16 smaller ones and extracting average color of them
    r_image_list = [r_screenshot[int(n * h_r / h_LEDs):int(n * h_r / h_LEDs + h_r / h_LEDs), :] for n in range(h_LEDs)]
    r_average_img_color = [np.average(r_image_list[n], axis=(1, 0)) for n in range(h_LEDs)]
    r_average_img_color = [[int(channels[2]), int(channels[1]), int(channels[0])] for channels in r_average_img_color]
    r_colors = [[i+right_pixels, color] for i, color in enumerate(r_average_img_color)]

    u_image_list = [u_screenshot[:, int(n * w_u / w_LEDs):int(n * w_u / w_LEDs + w_u / w_LEDs)] for n in range(w_LEDs)]
    u_average_img_color = [np.average(u_image_list[n], axis=(1, 0)) for n in range(w_LEDs)]
    u_average_img_color = [[int(channels[2]), int(channels[1]), int(channels[0])] for channels in u_average_img_color]
    u_colors = [[i+upper_pixels, color] for i, color in enumerate(u_average_img_color)]

    l_image_list = [l_screenshot[int(n * h_l / h_LEDs):int(n * h_l / h_LEDs + h_l / h_LEDs), :] for n in range(h_LEDs)]
    l_average_img_color = [np.average(l_image_list[n], axis=(1, 0)) for n in range(h_LEDs)]
    l_average_img_color = [[int(channels[2]), int(channels[1]), int(channels[0])] for channels in l_average_img_color]
    l_colors = [[i+left_pixels, color] for i, color in enumerate(l_average_img_color)]

    # Dividing screenshots to smaller ones and extracting dominant color of them
    # r_image_list = [r_screenshot[int(n * h_r / h_LEDs):int(n * h_r / h_LEDs + h_r / h_LEDs), :] for n in range(h_LEDs)]
    # r_dominant_img_color = [dominant_color(screenshot) for screenshot in r_image_list]
    # r_colors = [[i + right_pixels, color] for i, color in enumerate(r_dominant_img_color)]
    #
    # u_image_list = [u_screenshot[:, int(n * w_u / w_LEDs):int(n * w_u / w_LEDs + w_u / w_LEDs)] for n in range(w_LEDs)]
    # u_dominant_img_color = [dominant_color(screenshot) for screenshot in u_image_list]
    # u_colors = [[i + upper_pixels, color] for i, color in enumerate(u_dominant_img_color)]
    #
    # l_image_list = [l_screenshot[int(n * h_l / h_LEDs):int(n * h_l / h_LEDs + h_l / h_LEDs), :] for n in range(h_LEDs)]
    # l_dominant_img_color = [dominant_color(screenshot) for screenshot in l_image_list]
    # l_colors = [[i + left_pixels, color] for i, color in enumerate(l_dominant_img_color)]

    # end_time = time.time()
    # print(f'Time elapsed {end_time-start_time}')

    # Returning the extracted colors
    return l_colors + u_colors + r_colors
    # return l_average_img_color + u_average_img_color + r_average_img_color

def color_gen(h_LEDs=18, w_LEDs=36, h=1440, w=2560):
    """ Returns color spectrum of a display's frame """
    s_time = time.time()
    left_pixels = 0  # Offset for LEDs on the left frame
    upper_pixels = 18  # Offset for LEDs on the upper frame
    right_pixels = 54  # Offset for LEDs on the right frame

    h_divider = 8
    w_divider = 16

    # Defining bounding boxes
    u_box = {"top": 0, "left": 0, "width": w, "height": int(h / h_divider)}
    l_box = {"top": int(h / h_divider), "left": 0, "width": int(w / w_divider), "height": int(h - h / h_divider)}
    r_box = {"top": int(h / h_divider), "left": int(w - w / w_divider), "width": int(w / w_divider), "height": int(h - h / h_divider)}

    # Taking screenshot of the frame - frame is composed of 3 screenshots (2 side and 1 upper fields)
    with mss.mss() as camera:
        u_screenshot = np.array(camera.grab(u_box))
        l_screenshot = np.array(camera.grab(l_box))
        r_screenshot = np.array(camera.grab(r_box))
        u_screenshot = cv2.resize(u_screenshot, dsize=(int(w/4), int(h/h_divider/4)), interpolation=cv2.INTER_CUBIC)
        l_screenshot = cv2.resize(l_screenshot, dsize=(int(h/4), int(w/w_divider/4)), interpolation=cv2.INTER_CUBIC)
        r_screenshot = cv2.resize(r_screenshot, dsize=(int(h/4), int(w/w_divider/4)), interpolation=cv2.INTER_CUBIC)

    print(time.time() - s_time)
    # cv2.imshow("im", cv2.resize(u_screenshot, dsize=(int(w/2), int(h/h_divider/2)), interpolation=cv2.INTER_CUBIC))
    # cv2.waitKey()

    h_r, w_r, c_r = r_screenshot.shape
    h_l, w_l, c_l = l_screenshot.shape
    h_u, w_u, c_u = u_screenshot.shape

    # Dividing screenshots to 16 smaller ones and extracting average color of them
    l_image_list = [l_screenshot[int(n * h_l / h_LEDs):int(n * h_l / h_LEDs + h_l / h_LEDs), :] for n in range(h_LEDs)]
    l_average_img_color = [np.average(l_image_list[n], axis=(1, 0)) for n in range(h_LEDs)]
    l_average_img_color = [[int(channels[2]), int(channels[1]), int(channels[0])] for channels in l_average_img_color]
    l_colors = [[upper_pixels-1-i, color] for i, color in enumerate(l_average_img_color)]

    u_image_list = [u_screenshot[:, int(n * w_u / w_LEDs):int(n * w_u / w_LEDs + w_u / w_LEDs)] for n in range(w_LEDs)]
    u_average_img_color = [np.average(u_image_list[n], axis=(1, 0)) for n in range(w_LEDs)]
    u_average_img_color = [[int(channels[2]), int(channels[1]), int(channels[0])] for channels in u_average_img_color]
    u_colors = [[i+upper_pixels, color] for i, color in enumerate(u_average_img_color)]

    r_image_list = [r_screenshot[int(n * h_r / h_LEDs):int(n * h_r / h_LEDs + h_r / h_LEDs), :] for n in range(h_LEDs)]
    r_average_img_color = [np.average(r_image_list[n], axis=(1, 0)) for n in range(h_LEDs)]
    r_average_img_color = [[int(channels[2]), int(channels[1]), int(channels[0])] for channels in r_average_img_color]
    r_colors = [[i+right_pixels, color] for i, color in enumerate(r_average_img_color)]

    # Dividing screenshots to smaller ones and extracting dominant color of them
    # r_image_list = [r_screenshot[int(n * h_r / h_LEDs):int(n * h_r / h_LEDs + h_r / h_LEDs), :] for n in range(h_LEDs)]
    # r_dominant_img_color = [get_dominant_color(screenshot) for screenshot in r_image_list]
    # r_colors = [[i + right_pixels, color] for i, color in enumerate(r_dominant_img_color)]
    #
    # u_image_list = [u_screenshot[:, int(n * w_u / w_LEDs):int(n * w_u / w_LEDs + w_u / w_LEDs)] for n in range(w_LEDs)]
    # u_dominant_img_color = [get_dominant_color(screenshot) for screenshot in u_image_list]
    # u_colors = [[i + upper_pixels, color] for i, color in enumerate(u_dominant_img_color)]
    #
    # l_image_list = [l_screenshot[int(n * h_l / h_LEDs):int(n * h_l / h_LEDs + h_l / h_LEDs), :] for n in range(h_LEDs)]
    # l_dominant_img_color = [get_dominant_color(screenshot) for screenshot in l_image_list]
    # l_colors = [[i + left_pixels, color] for i, color in enumerate(l_dominant_img_color)]

    # Returning the extracted colors
    print(time.time()-s_time)
    # print(l_colors + u_colors + r_colors)
    return l_colors + u_colors + r_colors
    # return l_average_img_color + u_average_img_color + r_average_img_color


def get_dominant_color(screenshot):

    unique, counts = np.unique(screenshot.reshape(-1, 3), axis=0, return_counts=True)
    return list(unique[np.argmax(counts)])


def serial_comm(port, baudrate):
    """ Exchanges data between device via serial port """

    # Configure the serial port
    ser = serial.Serial()
    # In my case DTR and RTS pins have to be set to false to be able to open port
    # It may vary depends on the ESP model that you have
    ser.setDTR(False)
    ser.setRTS(False)
    ser.port = port
    ser.timeout = 0.1
    ser.baudrate = baudrate
    ser.open()

    try:
        while True:
            start_time = time.time()
            color_list = color_gen()
            list_size = len(color_list)
            color_list1 = str(color_list[0:18]) + '\n'
            # message = json.dumps(color_gen()).strip(' ') + '\n'
            # message = str(color_gen()).replace(' ', '') + '\n'

            # Send the command to the ESP32
            ser.write(color_list1.encode('utf-8'))

            # Wait for a response
            # response = ser.readline().strip()
            # print("Response from ESP32:", response)

            end_time = time.time()
            print(f'Time elapsed: {end_time - start_time}')
            # time.sleep(5)w

    except KeyboardInterrupt:
        ser.close()


def find_port(device_name):
    """ Finds port of specified device """

    ports = list(serial.tools.list_ports.comports())
    for port, desc, hwid in ports:
        if device_name.lower() in desc.lower():
            return port

    return None


if __name__ == "__main__":

    ESP32_name = 'CH340'
    # ESP32_baudrate = 115400
    # ESP32_baudrate = 230400
    ESP32_baudrate = 460800
    # ESP32_baudrate = 921600
    ESP32_port = find_port(ESP32_name)
    serial_comm(ESP32_port, ESP32_baudrate)
