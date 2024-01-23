import esp_main
import esp
from machine import Pin
import gc
import _thread

esp.osdebug(None)

gc.collect()
print("Hello!")

# --------------------------------------------------------------------------------

# Uncomment line below to start main loop automatically when esp pluged in to the PC

# esp_main.main()

# --------------------------------------------------------------------------------

# Uncomment code below if you want to start main loop when there is high state on specified GPIO

# Define the GPIO pin
# IRQ_PIN = Pin(9, Pin.IN, Pin.PULL_UP)

# def irq_handler_thread():
#     while True:
#         if IRQ_PIN.value() == 0:
#             esp_main.main()

# _thread.start_new_thread(irq_handler_thread, ())

# --------------------------------------------------------------------------------

print("Boot.py executed.")