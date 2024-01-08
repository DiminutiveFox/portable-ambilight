import sys
import uselect
from machine import Pin
import machine
import neopixel
import time
import random
import json
import os

# Configure the Neopixel
NUM_PIXELS = 72
# Total number of LEDs
PIN = 5  # GPIO for neopixel data

np = neopixel.NeoPixel(Pin(PIN), NUM_PIXELS)
serialPoll = uselect.poll()
serialPoll.register(sys.stdin, uselect.POLLIN)
LED = Pin(4, Pin.OUT)

def convert_to_list(color_string):
    """Converts received string of colors to list"""
    return json.loads(color_string)

def handle_command(command):

    if command == None: # filter out empty messages
        return 0
    sys.stdout.buffer.write(command)

def led_strip():

    try:
        while True:
            color_wipe((255, 0, 0))  # Red wipe
            time.sleep(1)

            color_wipe((0, 255, 0))  # Green wipe
            time.sleep(1)

            color_wipe((0, 0, 255))  # Blue wipe
            time.sleep(1)

            rainbow_cycle()  # Rainbow cycle
            time.sleep(1)

    except KeyboardInterrupt:
        # Turn off LEDs on exit
        np.fill((0, 0, 0))
        np.write()


def color_wipe(color, wait_ms=50):
    for i in range(NUM_PIXELS):
        np[i] = color
        np.write()
        time.sleep_ms(wait_ms)

def rainbow_cycle(wait_ms=20, iterations=5):
    num_colors = 256
    for j in range(iterations):
        for i in range(num_colors):
            r, g, b = wheel((i * 256 // num_colors) % 256)
            for pixel in range(NUM_PIXELS):
                np[pixel] = (r, g, b)
            np.write()
            time.sleep_ms(wait_ms)

def wheel(pos):
    # Generate rainbow colors across 0-255 positions
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    else:
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)


def rand_color():

    rand_LED = random.randint(0, 71)
    col1 = random.randint(0, 255)
    col2 = random.randint(0, 255)
    col3 = random.randint(0, 255)
    ambience(rand_LED, (col1, col2, col3))

def ambience(led_number, color):
    np[led_number] = color
    np.write()

def create_ambience(message):

    for color in message:
        np[color[0]] = color[1]
    np.write()


def readSerial():

    return(sys.stdin.readline() if serialPoll.poll(0) else None)

def custom_eval(input_str):
    stack = []
    current_list = None

    for char in input_str:
        if char == "[":
            new_list = []
            if current_list is not None:
                stack.append(current_list)
            current_list = new_list
        elif char == "]":
            if stack:
                previous_list = stack.pop()
                previous_list.append(current_list)
                current_list = previous_list
        elif char.isdigit():
            num_str = char
            while input_str[input_str.index(char) + 1].isdigit():
                char = input_str[input_str.index(char) + 1]
                num_str += char
            current_list.append(int(num_str))

    return current_list

def readSerial_continously():
    # ESP32_baudrate = 115400
    # ESP32_baudrate = 230400
    ESP32_baudrate = 460800
    # ESP32_baudrate = 921600

    machine.UART(0, baudrate=ESP32_baudrate)

    while True:
        # continuously read commands over serial and handle them
        try:
            message = readSerial()
            # time.sleep_ms(5)
            if message is not None:
                message = message.strip('\n')
                color = convert_to_list(message)
                create_ambience(color)
        except Exception as err:
            print(err)
            pass

def main():
    readSerial_continously()

