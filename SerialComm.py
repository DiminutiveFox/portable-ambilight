# PC Python UART Example

import serial
import time
import threading

# Define the COM port (adjust accordingly)
com_port = '/dev/ttyUSB0'  # Linux
# com_port = 'COMx'  # Windows (replace x with the actual COM port number)

ser = serial.Serial(com_port, 115200, timeout=1)

def receive_data():
    while True:
        data = ser.readline().decode("utf-8").strip()
        print("Received:", data)

# Start a new thread for receiving data concurrently
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()

# Your main code goes here
try:
    while True:
        ser.write('{"command": "ping"}\n'.encode('utf-8'))
        time.sleep(1)
except KeyboardInterrupt:
    ser.close()
    receive_thread.join()