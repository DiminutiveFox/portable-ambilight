# Portable Ambilight
Prototype of custom portable ambilight solution made of 3d printed parts, ESP32-C3 development board and WS2812B LED strip.
It is meant for laptops' screens and has a few interesting traits:
- it can be easily removed from the back of the screen
- comunicates with PC over USB
- is customizable
- easy to use

https://github.com/DiminutiveFox/portable-ambilight/assets/135659343/e31a8899-3bb5-47f5-a258-803ae1b41607


# Project description
Project's code is entirely written in python. It has 3 files - 2 of them are meant for ESP32-C3 (boot.py and esp_main.py). To be able to run python code on this module you first need to install micropython on it. There are several tutorials all over the internet that show the process and also how to flash the board. Need to know that different ESP development boards can be used, but installation of micropython might vary for every module. Micropython might not be the right choice for this kind of project but I wanted to show that this task is doable using it. 

# boot.py
This is the file which is executed when ESP boots. It has standard structure and executes main.py which contains all important functions. 

# esp_main.py
This is the main file for ESP. The most important aspect is that we want to communicate with PC over USB-TTL bridge already mounted on board. It cannot be done easily since this bridge is connected to the UART0 which is reserved for REPL. Instead of UART communication sys.stdin is used for message reading on the ESP side. Default baudrate for REPL is 115200. It can be increased but communication becames unpredictable in higher rates - sometimes message is not read entirely or new message is inserted in the middle of the previous one.  

