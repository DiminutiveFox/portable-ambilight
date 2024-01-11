import esp_main
import esp
from machine import Pin
import gc
import _thread

# Define the GPIO pin
# IRQ_PIN = Pin(9, Pin.IN, Pin.PULL_UP)

esp.osdebug(None)

gc.collect()
print("Hello!")

# def irq_handler_thread():
#     while True:
#         if IRQ_PIN.value() == 0:
#             esp_main.main()
#
# # MAIN
# _thread.start_new_thread(irq_handler_thread, ())
# print("Boot.py executed.")
