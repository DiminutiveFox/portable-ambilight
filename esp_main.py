import sys
import uselect
from machine import Pin
import machine
import neopixel
import json
import time
import urandom

# Serial port configuration - only way to communicate thought USB
serialPoll = uselect.poll()
serialPoll.register(sys.stdin, uselect.POLLIN)


# Total number of LEDs
NUM_PIXELS = 68

# Configure NeoPixel and Mode Pin
NEO_PIN = 5  # GPIO for NeoPixel data
MODE_PIN_NUM = 3  # GPIO for mode selection

np = neopixel.NeoPixel(Pin(NEO_PIN), NUM_PIXELS)
MODE_PIN = Pin(MODE_PIN_NUM, Pin.IN, Pin.PULL_DOWN)

################################################################################
# Global variables
running_mode = 0  # Initial mode (0 to 4)
config_mode = False
mode_settings = 0  # Initial mode settings

running = False  # Indicates if the main loop is running
last_interrupt_time = 0  # For debouncing interrupts
NUM_MODES = 5  # Total number of modes



################################################################################
# Mode 1 settings
mode1_brightness = 255  # Brightness value (0–255)
mode1_saturation = 255  # Saturation value (0–255)
mode1_update_delay = 10000  # Delay in milliseconds (10 seconds)

################################################################################
# Mode 2 settings
mode2_brightness = 255  # Brightness value (0–255)
mode2_saturation = 255  # Saturation value (0–255)
mode2_hue_min = 0  # Minimum hue value (0–180)
mode2_hue_max = 180  # Minimum hue value (0–180)


class BreakLoop(Exception):
    """
    Custom exception to break out of the main loop.
    """
    pass    


def handle_interrupt(pin):
    """
    IRQ handler for the mode pin. This will increment the mode and wrap back to 0 after reaching the last mode.
    """
    global running, running_mode, config_mode, last_interrupt_time 
    global mode2_hue_min, mode2_hue_max, mode2_brightness
    global mode1_brightness, mode1_saturation

    current_time = time.ticks_ms()
    
    if pin.value() == 1:
        print(f"Button pressed!")
        last_interrupt_time = current_time
        return
    
    if pin.value() == 0:
        print(f"Button released!")
        if time.ticks_diff(current_time, last_interrupt_time) < 1000 and not config_mode:
            running_mode = (running_mode + 1) % NUM_MODES  # Increment mode and wrap back to 0 after NUM_MODES
            print(f"Mode changed to: {running_mode}")
            

        if time.ticks_diff(current_time, last_interrupt_time) < 1000 and config_mode:
            if running_mode == 1:
                mode1_brightness = (mode1_brightness + 10) % 255
                print(f"Mode 1 brightness changed to: {mode1_brightness}")
            elif running_mode == 2:
                if time.ticks_diff(current_time, last_interrupt_time) < 500:
                    if mode2_hue_max == 180 and mode2_hue_min == 0:
                        mode2_hue_max = 60
                        mode2_hue_min = 0
                        print(f"Mode 2 hue changed to [{mode2_hue_min} - {mode2_hue_max}]")
                    elif mode2_hue_max == 180 and mode2_hue_min == 120:
                        mode2_hue_max = 180
                        mode2_hue_min = 0
                        print(f"Mode 2 hue changed to [{mode2_hue_min} - {mode2_hue_max}]")
                        
                    else:
                        mode2_hue_min = (mode2_hue_min + 60) % 181
                        mode2_hue_max = (mode2_hue_max + 60) % 181  
                        print(f"Mode 2 hue changed to [{mode2_hue_min} - {mode2_hue_max}]")
                if time.ticks_diff(current_time, last_interrupt_time) >= 500:
                    mode2_brightness = (mode2_brightness + 25) % 255
                    print(f"Mode 2 brightness changed to: {mode2_brightness}")
            elif running_mode == 3: 
                pass
            elif running_mode == 4:
                pass
            else:
                print("Config mode not available for the current mode.")
                return
            print(f"Parameters for mode {running_mode} updated.")

        
        if  5000 >= time.ticks_diff(current_time, last_interrupt_time) >= 1000:
            config_mode = not config_mode
            if config_mode:
                print(f"Entering config mode")
            else:
                print(f"Exiting config mode")
            return
        
        if  time.ticks_diff(current_time, last_interrupt_time) >= 10000:
            running = False
            raise BreakLoop

    else:
        print("Config mode not available for the current mode.")    


def mode_0():
    print("Executing Mode 0: Serial Communication Mode")
    serial_comm()

def mode_1():
    global mode1_update_delay
    print("Executing Mode 1: HSV palette transition mode")
    HSV = [0, mode1_saturation, mode1_brightness]
    h = 0
    mode1_update_delay = 10000  # Delay in milliseconds (10 seconds)
    last_update_time = time.ticks_ms()  # Record the current time

    for i in range(NUM_PIXELS):
        np[i] = hsv_to_rgb(HSV)    
    np.write()
    h += 1

    while running_mode == 1:
        current_time = time.ticks_ms()
        
        # Update LED colors only if the delay has passed
        if time.ticks_diff(current_time, last_update_time) >= mode1_update_delay:
            for i in range(NUM_PIXELS):
                np[i] = hsv_to_rgb(HSV)
            
            np.write()
            print(f"Updated color to hue: {h}")
            
            # Update hue
            if h < 180:
                h += 1
            else:
                h = 0
            
            HSV = [h, mode1_saturation, mode1_brightness]
            last_update_time = current_time  # Reset the timer

        # Allow time for other tasks (interrupts) to be processed
        time.sleep_ms(1)  # Sleep for 1 ms to prevent CPU hogging

def mode_4():
    print("Executing Mode 2: LED Pattern Mode 2")
    for i in range(NUM_PIXELS):
        np[i] = [0, 255, 0]  # Green
    np.write()

def mode_3():
    print("Executing Mode 3: LED Pattern Mode 3")
    for i in range(NUM_PIXELS):
        np[i] = [0, 0, 255]  # Blue
    np.write()
    


def mode_2():
    reset_np()
    print("Executing Mode 4: Smooth Random Color Transition")


    transition_steps = 50  # Number of gradient steps for each transition
    hot_pixels = [1, 9, 18, 26, 34, 42, 50, 59, 67]

    start_colors = [hsv_to_rgb(random_hsv_color(mode2_hue_min, mode2_hue_max)) for _ in hot_pixels]
    end_colors = [hsv_to_rgb(random_hsv_color(mode2_hue_min, mode2_hue_max)) for _ in hot_pixels]

    while running_mode == 2:
        for step in range(transition_steps):
            for i, pixel in enumerate(hot_pixels):
                color = create_gradient_rgb(transition_steps, start_colors[i], end_colors[i])[step]
                np[pixel] = color
            np.write()
            time.sleep(0.05)

        # Prepare new colors for the next transition
        start_colors = end_colors
        end_colors = [hsv_to_rgb(random_hsv_color(mode2_hue_min, mode2_hue_max)) for _ in hot_pixels]
        print(end_colors)




def create_gradient_rgb(steps, start_color, end_color):
    """
    Creates a smooth gradient between two colors across a range of steps.
    The first color in the list will be the start_color, followed by the gradient,
    but the end color will not be included in the list.
    
    :param steps: Number of steps to create in the gradient.
    :param start_color: Starting color (R, G, B) as a tuple.
    :param end_color: Ending color (R, G, B) as a tuple.
    
    :return: A list of colors for the gradient (as (R, G, B) tuples).
    """
    # print(f'Start color: {start_color}')
    # print(f'End color: {end_color}')

    r1, g1, b1 = start_color
    r2, g2, b2 = end_color
    
    gradient_colors = []  # Start with the start color

    for i in range(1, steps+1):  # Start from 1 to exclude the end color
        # Calculate the interpolation factor (0.0 to 1.0)
        factor = i / (steps)
        
        # Interpolate each color channel
        r = int(r1 + (r2 - r1) * factor)
        g = int(g1 + (g2 - g1) * factor)
        b = int(b1 + (b2 - b1) * factor)
        
        # Append the calculated color to the gradient list
        gradient_colors.append([r, g, b])
    
    return gradient_colors


def random_hsv_color(hue_min=0, hue_max=180):
    """
    Generates a random HSV color for ESP32 within a specified hue range.
    
    :param hue_min: Minimum hue value (0–180).
    :param hue_max: Maximum hue value (0–180).
    :return: Tuple (h, s, v) where:
             h is Hue (hue_min–hue_max),
             s is Saturation (100–255),
             v is Value/Brightness (fixed at 255).
    """
    if hue_min < 0 or hue_max > 180 or hue_min >= hue_max:
        raise ValueError("hue_min must be >= 0, hue_max <= 180, and hue_min < hue_max")

    h = urandom.getrandbits(8) % (hue_max - hue_min + 1) + hue_min  # Random hue within the range
    s = urandom.getrandbits(8) + 180  # Random saturation (100–255)
    s = min(s, 255)  # Clamp saturation to 255
    v = 255  # Fixed brightness value

    return h, s, v


def hsv_to_rgb(hsv):
    """
    Converts HSV to RGB.
    
    :param h: Hue (0–180)
    :param s: Saturation (0–255)
    :param v: Value/Brightness (0–255)
    :return: Tuple (r, g, b) with values in the range (0–255)
    """
    h, s, v = hsv

    h = h % 180  # Ensure hue stays within 0-180
    s = max(0, min(255, s))
    v = max(0, min(255, v))

    s /= 255
    v /= 255

    c = v * s
    x = c * (1 - abs((h / 30) % 2 - 1))
    m = v - c

    r = g = b = 0
    if 0 <= h < 30:
        r, g, b = c, x, 0
    elif 30 <= h < 60:
        r, g, b = x, c, 0
    elif 60 <= h < 90:
        r, g, b = 0, c, x
    elif 90 <= h < 120:
        r, g, b = 0, x, c
    elif 120 <= h < 150:
        r, g, b = x, 0, c
    elif 150 <= h < 180:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return [r, g, b]


def convert_to_list(color_string):
    """Converts received string of colors to list"""
    return json.loads(color_string)

def hex_string_to_int_list(hex_string):
    # Ensure the string length is a multiple of 2 (it should be, as hex pairs represent RGB)
    if len(hex_string) % 2 != 0:
        raise ValueError("The hexadecimal string should have an even number of characters.")
    
    # Convert the hex string to a list of integers
    return [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)]

def handle_command(command):
    """Returns command to the stream"""
    if command:
        sys.stdout.buffer.write(command)

def create_ambience(color_string, led_span=1):
    """
    Converts message to list of colors and writes it to NeoPixel instance
    """
    color_length = 6  # Length of each color in the string (e.g., "RRGGBB")
    color_strings = [color_string[i:i + color_length] for i in range(0, len(color_string), color_length)]

    # Convert hex color strings to RGB lists
    try:
        color_list = [hex_string_to_int_list(color_string) for color_string in color_strings]
    except ValueError as e:
        print(f"Error converting hex string to int list: {e}")
        return

    # Assign colors to the NeoPixel LEDs
    for i, color in enumerate(color_list):
        if len(color) == 3:
            np[led_span * i] = color
        else:
            print(f"Invalid color length at index {i}: {color}")
            np[led_span * i] = [0, 0, 0]
    
    np.write()

def reset_np():
    """
    Resets all pixel values back to [0, 0, 0]
    """
    for i in range(NUM_PIXELS):
        np[i] = [0, 0, 0]
    np.write()

def read_serial():
    """
    Reads a message - 5216 is the longest
    :return: message (in bytes)
    """
    return sys.stdin.buffer.readline() if serialPoll.poll(10) else None

def constant_color(color_string):

    for i in range(NUM_PIXELS):
        np[i] = [int('0x'+color_string[0:2]), int('0x'+color_string[2:4]), int('0x'+color_string[4:6])]

    np.write()

def serial_comm(baudrate=115200):
    """
    Main serial communication loop
    """
    global running
    global running_mode
    
    reset_np()

    # Change baudrate if specified
    if baudrate != 115200:
        machine.UART(0, baudrate=baudrate)

    config_mem = [0, 0]

    # Main communication loop
    while running:
        try:
            data = read_serial()
            if data is not None:
                data = data.decode('utf-8').strip()
                print(f"Received data: {data}")

                # Ensure the data is long enough to contain config bits and color data
                if len(data) < 2:
                    print(f"Invalid data length: {len(data)}")
                    continue

                config = list(data[:2])
                color_string = data[2:]

                print(f"Config: {config}, Color String: {color_string}")

                if config_mem[1] == 1:
                    constant_color(color_string)
                else:
                    if config_mem != config:
                        reset_np()
                    else:
                        create_ambience(color_string, int(config[0]))

                # sys.stdout.write("ok\n")
                config_mem = config

        except Exception as err:
            print(f"Error: {err}")
        if running_mode == 1:
            break

def blink_red():
    reset_np()
    time.sleep(0.5)
    for i in range(NUM_PIXELS):
        np[i] = [255, 0, 0]  # RED
    np.write()
    time.sleep(0.5)
    reset_np()
    time.sleep(0.5)
    for i in range(NUM_PIXELS):
        np[i] = [255, 0, 0]  # RED
    np.write()
    time.sleep(0.5)
    reset_np()
    time.sleep(0.5)
    for i in range(NUM_PIXELS):
        np[i] = [255, 0, 0]  # RED
    np.write()
    time.sleep(0.5)
    reset_np()

def blink_blue():
    reset_np()
    time.sleep(0.2)
    for i in range(NUM_PIXELS):
        np[i] = [0, 0, 255]  # BLUE
    np.write()
    time.sleep(0.2)
    reset_np()
    time.sleep(0.2)
    for i in range(NUM_PIXELS):
        np[i] = [0, 0, 255]  # BLUE
    np.write()
    time.sleep(0.2)
    reset_np()
    time.sleep(0.2)
    for i in range(NUM_PIXELS):
        np[i] = [0, 0, 255]  # BLUE
    np.write()
    time.sleep(0.2)
    reset_np()

def main():
    """
    Main function to manage modes and handle tasks based on the current mode.
    """
    global running
    running = True

    # Configure the interrupt for mode pin (falling edge trigger)
    MODE_PIN.irq(trigger=Pin.IRQ_FALLING | Pin.IRQ_RISING, handler=handle_interrupt)

    print("Entering main loop...")
    try:
        while running:
            if running_mode == 0:
                mode_0()
            elif running_mode == 1:
                mode_1()
            elif running_mode == 2:
                mode_2()
            elif running_mode == 3:
                mode_3()
            elif running_mode == 4:
                mode_4()
            
            time.sleep(0.1)  # Prevent CPU overloading
    except BreakLoop:
        print("Exiting main loop...")
    finally:
        print("Exiting main...")
        blink_red()
    
# Start the main function when this script is run
if __name__ == "__main__":
    main()
