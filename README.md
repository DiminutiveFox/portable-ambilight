# Portable Ambilight
Prototype of custom portable ambilight solution made of 3d printed parts, ESP32-C3 development board and WS2812B LED strip.
It is meant for laptops' screens and has a few interesting traits:
- it can be easily removed from the back of the screen
- comunicates with system over USB
- is customizable
- easy to use
- WS2812B does not need to be externally powered up

![Ambilight](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/57772cf3-2aa2-4531-8323-f84526c7870e)

# Project description
Project's code is entirely written in python. It has 3 files - 2 of them are meant for ESP32-C3 (boot.py and esp_main.py). To be able to run python code on this module you first need to install micropython on it. There are several tutorials all over the internet that show the process and also how to flash the board. Need to know that different ESP development boards can be used, but installation of micropython might vary depending on the module. Micropython might not be the right choice for this kind of project but I wanted to show that this task is doable using it. 

# Way of communication
The goal is to communicate and send all the needed data through USB. It rises a few challanges since serial communication is not suitable choice for video stream - it's too slow. Message sent to ESP needs to be as short as possible to keep LEDs frame's framerate around 20-30fps. It provides the best experience. File main.py needs to be run in the background - it takes screnshot of the areas near the screen's sides and extracts RGB values for LEDs to display. The value depends on the function applied for the extracted area - at this moment only 'average' function is available - more will be added in the future. Message's length depends mostly on how many LEDs are in the setup - in my case there are 72 LEDs - 36 on the upper frame, 18 LEDs for each side. Value for every RGB LED is scaled down (scale parameter set to 1) from 0-255 to 0-99 range - it shortens a message a bit and the result is indistinguishable comparing to not scaled down message. Additionally user can specify how many LEDs light up e.g. every second or third LED (led_span parameter). Below is a example of message combined of scaled LEDs' values.

![image](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/2f02d839-b3f1-4b39-8806-ea0d87127c2e)

Framerate also depends on the screen resolution - the higher the resolution the more it takes to create a screenshot and compute LEDs' values. Also compute power of your PC is relevant - code is much slower on older machines.  

# main.py
The heart of the project - it's functionality was mostly described above. User needs to specify parameters of the setup e.g. number of LEDs and screen resolution (all parameters are found in __main__). On PC side we use UART for communication. Using baudrate above 115200 is not recommended.

# boot.py
This is the file which is executed when ESP boots. It has standard structure and executes main.py which contains all important functions. Read the comments to configure the esp startup. Starting main loop from REPL is though recommended. Remember when esp_main.main() is started directly in boot.py file the REPL is lost and all files need to be erased (for example using esptool) to be able to flash ESP again. 

# esp_main.py
This is the main file for ESP. The most important aspect is that we want to communicate with system over USB-TTL bridge already mounted on board. It cannot be done easily since this bridge is connected to the UART0 which is reserved for REPL. Instead of UART communication sys.stdin is used for message reading on the ESP side. Default baudrate for REPL is 115200. It can be increased but communication becames unpredictable in higher rates - sometimes message is not read entirely or new message is inserted in the middle of the previous one. ESP behavior highly depends on the module. Check manufacturer's docs before buying one (and avoid my rookie mistake). Specify number of LEDs and GPIO for NeoPixel object. For debugging purpose you can uncomment part of code which sends the message back to the PC (also you need to do this in main.py file) 

![image](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/662ab1bc-f1a4-4cb6-8715-5ed4acf35490)

# 3d parts 
LED strip (if you are using standard monitor) can be attached directly to the back of your screen. But if you are using laptop (like me) you might want to use your ambilight only when you are home and remove it when travelling. So that is why I made a 3d printed frame that can be attached to the back using suction cups. All parts (apart of suction cups) are provided in the project. Mind that ESP case might not be suitable for your dev board. This design might not be the best (in my humble opinion it is even ugly and there is a lot of space for improvement) but it provides all functionality that is needed. Frame is combined out of 4 parts - 2 side frames, upper frame and base frame. I recommend to glue everything together. Insert suction cups into the holes in side frames. Put ESP in case and glue it into the base. LED strip is glued to the sides of the frame (picture below).

![image](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/e82487e0-286e-4c1d-bbc6-3ae92bf21835)

# Step by step guide

- download and print 3d parts - you can of course make your own design
- glue everything together
- put suction cups into the holes
- cut and solder LED strip accordingly to the picture below
- install micropython on your dev board
- download code for ESP, specify parameters of your setup and flash the board
- specify parameters of your setup in main.py file and run it
- put it on the back on your laptop
- enjoy!

 ![image](https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/566e5c64-f5a7-4bc0-9649-e9564c1cf86e)



