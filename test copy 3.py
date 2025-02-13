import json
import time
import random

mode = 0  # Initial mode (0 to 4)
mode_settings = 0  # Initial mode settings

running = False  # Indicates if the main loop is running
last_interrupt_time = 0  # For debouncing interrupts
NUM_MODES = 5  # Total number of modes
NUM_PIXELS = 68

def mode_4():
    """
    Mode 4: Smooth random color transitions similar to Roccat Swarm AIMO.
    """
    print("Executing Mode 4: Smooth Random Color Transition")
    
    transition_steps = 50  # Number of gradient steps for each transition
    hot_pixels = [1, 9, 18, 26, 34, 42, 50, 59, 68]  # Pixel positions to apply random colors
    start_hot_pixels_colors = [hsv_to_rgb(random_hsv_color()) for _ in hot_pixels]
    end_hot_pixels_colors = [hsv_to_rgb(random_hsv_color()) for _ in hot_pixels]
    
    # Create smooth gradients between each pair of hot pixels
    while mode == 4:
        pixels_start_colors = []
        pixels_start_colors.append([start_hot_pixels_colors[0]])

        for i in range(len(hot_pixels)-1):
            gradient = create_gradient_rgb(hot_pixels[i+1]-hot_pixels[i], start_hot_pixels_colors[i], start_hot_pixels_colors[i+1])
            pixels_start_colors.append(gradient)
            
        pixels_start_colors = sum(pixels_start_colors, [])  # Flatten the list

        pixels_end_colors = []
        pixels_end_colors.append([end_hot_pixels_colors[0]])
        for i in range(len(hot_pixels)-1):
            gradient = create_gradient_rgb(hot_pixels[i+1]-hot_pixels[i], end_hot_pixels_colors[i], end_hot_pixels_colors[i+1])
            pixels_end_colors.append(gradient)
        
        pixel_pallete = []
        pixels_end_colors = sum(pixels_end_colors, [])  # Flatten the list
        
        for start_pixel, end_pixel in zip(pixels_start_colors, pixels_end_colors):
            gradient = create_gradient_rgb(transition_steps, start_pixel, end_pixel)
            pixel_pallete.append(gradient)

        transposed_pallete = [list(step_colors) for step_colors in zip(*pixel_pallete)]

        for gradient in transposed_pallete:
            for pixel in NUM_PIXELS:
                np[pixel] = gradient[pixel]
            time.sleep(1)
        
        start_hot_pixels_colors = end_hot_pixels_colors
        end_hot_pixels_colors = [hsv_to_rgb(random_hsv_color()) for _ in hot_pixels]
    
    # Output for verification
    print("Start Colors Gradients: ", transposed_pallete)
    print("Length: ", len(transposed_pallete))


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
    print(f'Start color: {start_color}')
    print(f'End color: {end_color}')

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
    
    # print("Gradient Colors: ", gradient_colors)
    # time.sleep(10)
    
    return gradient_colors

def random_hsv_color():
    """
    Generates a random HSV color for ESP32.
    :return: Tuple (h, s, v) where:
             h is Hue (0–180),
             s is Saturation (0–255),
             v is Value/Brightness (0–255)
    """
    h = random.getrandbits(8) % 181  # Random hue between 0 and 180
    s = random.getrandbits(8) + 100  # Random saturation (100–255)
    v = 100  # Fixed brightness value
    
    # Clamp values to a max of 255
    s = min(s, 255)

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


# Execute the function to test
mode_4()
