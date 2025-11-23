from machine import FPIOA, Pin
import time

# CONFIG
PIR_IO = 7   # motion sensor output wired to IO_7

# FPIOA MAPPING
fpioa = FPIOA()
fpioa.set_function(PIR_IO, FPIOA.GPIO7)  # map IO_7 to GPIO7

# PIN SETUP
pir_pin = Pin(PIR_IO, Pin.IN, Pin.PULL_DOWN)  # digital input with pulldown

print("PIR warmup... give it 30–60 seconds.")

# LOOP
while True:
    if pir_pin.value():   # 1 = motion detected
        print("MOTION DETECTED!")
        # small delay so it doesn’t spam
        time.sleep(0.2)
    else:
        # no motion
        pass

    time.sleep(0.05)
