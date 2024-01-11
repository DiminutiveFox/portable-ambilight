import usys
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
serialPoll.register(usys.stdin, uselect.POLLIN)
LED = Pin(6, Pin.OUT)
IRQ_PIN = Pin(9, Pin.IN, Pin.PULL_UP)


def convert_to_list(color_string):
    """Converts received string of colors to list"""
    return json.loads(color_string)


def handle_command(command):
    """Returns command to the stream"""
    if command:
        usys.stdout.buffer.write(command)


def create_ambience(message):
    """Activates WS2812B strip according to the message"""
    for color in message:
        np[color[0]] = color[1]
    np.write()


def read_serial():
    """Reads the message"""
    # return usys.stdin.buffer.readline()
    return usys.stdin.buffer.read(648) if serialPoll.poll(0) else None

def serial_comm():
    # esp32_baudrate = 115400
    # esp32_baudrate = 230400
    esp32_baudrate = 460800

    # Changing baudrate of the serial to be able to read data faster
    machine.UART(0, baudrate=esp32_baudrate)

    while True:
        # continuously read commands over serial and handle them
        try:
            data = read_serial()
            if data is not None:
                data = data.decode('utf-8')
                # handle_command(data)
                print(data)
                # color = convert_to_list(data)
                # create_ambience(color)
                usys.stdin.buffer.flush()
                usys.stdout.flush()
        except Exception as err:
            # print(err)
            pass


def main():

    print("Entering main...")
    time.sleep(0.5)
    serial_comm()
