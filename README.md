# Portable Ambilight
Prototype of custom portable ambilight solution made of 3d printed parts, ESP32-C3 development board and WS2812B LED strip.
It is meant for laptops' screens and has a few interesting traits:
- it can be easily removed from the back of the screen
- comunicates with PC over USB
- is customizable
- easy to use


![Ambilight](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/57772cf3-2aa2-4531-8323-f84526c7870e)


# Project description
Project's code is entirely written in python. It has 3 files - 2 of them are meant for ESP32-C3 (boot.py and esp_main.py). To be able to run python code on this module you first need to install micropython on it. There are several tutorials all over the internet that show the process and also how to flash the board. Need to know that different ESP development boards can be used, but installation of micropython might vary depending on the module. Micropython might not be the right choice for this kind of project but I wanted to show that this task is doable using it. 

# Way of communication
The goal is to communicate and send all the needed data through USB. It rises a few challanges since serial communication is not suitable choice for video stream - it's too slow. Message sent to ESP needs to be as short as possible to keep LEDs frame's framerate around 20-30fps. It provides the best experience. File main.py needs to be run in the background on the PC - it takes screnshot of the areas near the screen's sides and extracts RGB values for LEDs to display. The value depends on the function applied for the extracted area - at this moment only 'average' function is available - more will be added in the future. Message's length depends mostly on how many LEDs are in the setup - in my case there are 72 LEDs - 36 on the upper frame, 18 LEDs for each side. Value for every RGB LED is scaled down (scale parameter set to 1) from 0-255 to 0-99 range - it shortens a message a bit but the result is indistinguishable. Additionally user can specify how many LEDs light up e.g. every second or third LED (led_span parameter). Below is a example of message combined of scaled LEDs' values.

![image](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/2f02d839-b3f1-4b39-8806-ea0d87127c2e)


# boot.py
This is the file which is executed when ESP boots. It has standard structure and executes main.py which contains all important functions. 

# esp_main.py
This is the main file for ESP. The most important aspect is that we want to communicate with PC over USB-TTL bridge already mounted on board. It cannot be done easily since this bridge is connected to the UART0 which is reserved for REPL. Instead of UART communication sys.stdin is used for message reading on the ESP side. Default baudrate for REPL is 115200. It can be increased but communication becames unpredictable in higher rates - sometimes message is not read entirely or new message is inserted in the middle of the previous one.  

