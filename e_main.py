import sys
import uselect
from machine import Pin
import machine
import json
import uos

LED = Pin(4, Pin.OUT)


def handle_command(command):

    if command == None: # filter out empty messages
        return 0
    sys.stdout.buffer.write(command)


def convert_to_list(color_string):
    """Converts received string of colors to list"""
    return json.loads(color_string)

def read_serial(uart):

    return(uart.readline() if uart.any() else None)


def serial_communication():
    # uos.dupterm(None, 0)
    uart = machine.UART(0, baudrate=115200, rx=12, tx=13)
    uos.dupterm(uart)
    while True:
        uart.write('PETLA TRUE'.encode('utf-8'))
        uart.write('UART ANY TRUE'.encode('utf-8'))
        message = uart.readline().strip('\n') if uart.readline() is not None else None
        if message is not None:
            LED.value(1)
            uart.write(message.encode('utf-8'))
        # if uart.any():
        #     uart.write('UART ANY TRUE'.encode('utf-8'))
        #     message = uart.readline().decode('utf-8').strip('\n')
        #     if message is not None:
        #         LED.value(1)
        #         uart.write(message.encode('utf-8'))


def main():
    serial_communication()


