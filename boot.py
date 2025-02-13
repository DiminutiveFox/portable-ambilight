import esp_main
import esp
from machine import Pin
import gc
import _thread
import time

esp.osdebug(None)

gc.collect()
print("Hello!")

#
esp_main.main()
# --------------------------------------------------------------------------------

print("Boot.py executed.")