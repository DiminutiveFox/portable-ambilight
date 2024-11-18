import esp_main
import esp
from machine import Pin
import gc
import _thread

esp.osdebug(None)

gc.collect()
print("Hello!")

# def irq_handler_thread():
#     while True:
#         if IRQ_PIN.value() == 0:
#             esp_main.main()

# _thread.start_new_thread(irq_handler_thread, ())

# --------------------------------------------------------------------------------

print("Boot.py executed.")

esp_main.main()