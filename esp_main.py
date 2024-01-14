import sys
import uselect
from machine import Pin
import machine
import neopixel
import json
import time


# Configure the Neopixel
NUM_PIXELS = 72
# Total number of LEDs
NEO_PIN = 5  # GPIO for neopixel data

np = neopixel.NeoPixel(Pin(NEO_PIN), NUM_PIXELS)
serialPoll = uselect.poll()
serialPoll.register(sys.stdin, uselect.POLLIN)
LED = Pin(6, Pin.OUT)


def convert_to_list(color_string):
    """Converts received string of colors to list"""
    return json.loads(color_string)


def handle_command(command):
    """Returns command to the stream"""
    if command:
        sys.stdout.buffer.write(command)


def create_ambience2(color_string, scale=True, led_span=1):
    """
    Converts message to list of colors and writes it to NeoPixel instance \n \n
    Arguments: \n
    scale:            determines if color scale is reduced from 0-255 to 0-99
    channel_length:   length of the channel string (3 by default, 2 if scale is used)\n
    led_span:         reduces the number of active WS2812B LEDS
    """

    color_length = 6 if scale else 9
    channel_length = 2 if scale else 3

    color_strings = [color_string[i:i + color_length] for i in range(0, len(color_string), color_length)]

    if scale:
        color_list = [[scale_channel(int(color[i:i + channel_length])) for i in range(0, len(color), channel_length)]
                      for color in color_strings]
    else:
        color_list = [[int(color[i:i + channel_length]) for i in range(0, len(color), channel_length)]
                      for color in color_strings]

    # print(len(color_list))
    for i, color in enumerate(color_list):
        np[led_span * i] = color if len(color) == 3 else [0, 0, 0]
    np.write()


def scale_channel(channel_value):
    channel_max_value = 99
    channel_scaled_max_value = 255

    return int(channel_value*channel_scaled_max_value/channel_max_value)

def reset_np():

    for i in range(NUM_PIXELS):
        np[i] = [0, 0, 0]
    np.write()

def create_ambience(message):
    """Activates WS2812B strip according to the message"""
    for color in message:
        np[color[0]] = color[1]
    np.write()



def read_serial():
    """Reads a message"""
    # return sys.stdin.buffer.readline()
    return sys.stdin.buffer.readline(652) if serialPoll.poll(10) else None
    # return sys.stdin.read(648) if serialPoll.poll(0) else None
    # if serialPoll.poll(0):
    #     data = ""
    #     while sys.read


def serial_comm(baudrate=115400):

    #Changing baudrate
    if baudrate != 115200:
        machine.UART(0, baudrate=baudrate)

    config_mem = []

    # Main communication loop
    while True:
        try:
            data = read_serial()
            if data is not None:
                data = data.decode('utf-8').strip('\n')
                print(data)
                config = list(data[:2])
                color_string = data[2:]
                if config_mem != config:
                    # reset_np()
                    print('reset')
                else:
                    create_ambience2(color_string, bool(config[0]), int(config[1]))
                # print(data)
                config_mem = config
        except Exception as err:
            print(err)
            pass


def main():

    print("Entering main...")
    time.sleep(0.5)
    # esp32_baudrate = 230400
    # esp32_baudrate = 460800
    serial_comm()
