import cv2
import serial
import serial.tools.list_ports
from PIL import ImageGrab
import numpy as np
from screeninfo import get_monitors


def take_picture(box):
    """ Takes screenshot """
    bbox = (100, 100, 500, 500)
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox)), cv2.COLOR_BGR2RGB)
    cv2.imshow('Img', screenshot)
    cv2.waitKey()
    return screenshot

def color_gen():
    """ Returns color spectrum of a display's frame """

    # Getting the screen resolution
    w, h = get_monitors()[0].width, get_monitors()[0].height

    # Defining bounding boxes
    l_box = [int(w / 16), int(h / 16), int(2 * w / 16), int(h)]
    u_box = [int(2 * w / 16), int(h / 16), int(w - 2 * w / 16), int(2 * h / 16)]
    r_box = [int(w - 2 * w / 16), int(h / 16), int(w - w / 16), int(h)]

    # Taking screenshot of the frame - frame is composed of 3 screenshots (2 side and 1 upper fields)
    r_screenshot, l_screenshot, u_screenshot = take_picture(r_box), take_picture(l_box), take_picture(u_box)
    h_r, w_r, c_r = r_screenshot.shape
    h_l, w_l, c_l = l_screenshot.shape
    h_u, w_u, c_u = u_screenshot.shape

    # Dividing screenshots to 16 smaller ones and extracting average color of them
    r_image_list = [r_screenshot[int(n * h_r / 16):int(n * h_r / 16 + h_r / 16), :] for n in range(16)]
    r_average_img_color = [np.average(r_image_list[n], axis=(1, 0)) for n in range(16)]
    r_average_img_color = [[int(channel[2]), int(channel[1]), int(channel[0])] for channel in r_average_img_color]

    u_image_list = [u_screenshot[:, int(n * w_u / 16):int(n * w_u / 16 + w_u / 16)] for n in range(16)]
    u_average_img_color = [np.average(u_image_list[n], axis=(1, 0)) for n in range(16)]
    u_average_img_color = [[int(channel[2]), int(channel[1]), int(channel[0])] for channel in u_average_img_color]

    l_image_list = [l_screenshot[int(n * h_l / 16):int(n * h_l / 16 + h_l / 16), :] for n in range(16)]
    l_average_img_color = [np.average(l_image_list[n], axis=(1, 0)) for n in range(16)]
    l_average_img_color = [[int(channel[2]), int(channel[1]), int(channel[0])] for channel in l_average_img_color]

    # Returning the extracted colors
    return l_average_img_color, u_average_img_color, r_average_img_color


def serial_comm(port='COM1', baudrate=9600):
    """ Exchanges data between device via serial port """

    # Open the serial port
    with serial.Serial(port, baudrate) as ser:

        try:
            while True:
                # Taking a color spectrum of the display's frame
                color_data = color_gen()

                # Writing a line of data to the serial port
                ser.write(color_data).encode('utf-8')

        except serial.SerialException as Err:
            print(f'An error occurred: {Err}')


def find_port(device_name):
    """ Finds port of specified device """

    ports = list(serial.tools.list_ports.comports())
    for port, desc, hwid in ports:
        if device_name.lower() in desc.lower():
            return port

    return None


if __name__ == "__main__":

    ESP32_name = 'COM1'
    ESP32_baudrate = 9600
    ESP32_port = find_port(ESP32_name)

