import esp_main
import esp
from machine import Pin
import gc
import _thread
import time

esp.osdebug(None)

gc.collect()
print("Hello!")

# Pin connected to the button
BUTTON_PIN = 4  # Change to your specific GPIO pin
IRQ_PIN = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_DOWN)  # Assuming the button pulls the pin LOW when pressed

# State management
main_running = False
debounce_time = 200  # Debounce time in milliseconds


def toggle_main():
    """
    Toggle the execution of `esp_main.main()`.
    """
    # global main_running
    # if main_running:
    #     print("Stopping main()...")
    #     # Logic to stop main(), if necessary
    #     main_running = False
    # else:

    print("Starting main()...")
    esp_main.main()  # Call the main function
    main_running = True


def irq_handler_thread():
    """
    Thread that monitors the button state and toggles `esp_main.main()`.
    """
    last_press_time = 0  # To manage debouncing
    while True:
        if IRQ_PIN.value() == 1:  # Button pressed (assuming active LOW)
            current_time = time.ticks_ms()
            if time.ticks_diff(current_time, last_press_time) > debounce_time:
                toggle_main()
                # last_press_time = current_time
            # time.sleep(0.05)  # Small delay to reduce CPU usage


# Start the monitoring thread
_thread.start_new_thread(irq_handler_thread, ())

# --------------------------------------------------------------------------------

print("Boot.py executed.")