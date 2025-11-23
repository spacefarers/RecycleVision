'''
DIR  -> IO_3
STEP -> IO_5
EN   -> IO_7
'''

from machine import FPIOA, Pin
import time

# Pin Numbers
IO_DIR  = 3
IO_STEP = 5
IO_EN   = 7

# FPIOA Mapping
fpioa = FPIOA()
fpioa.set_function(IO_DIR,  FPIOA.GPIO3)
fpioa.set_function(IO_STEP, FPIOA.GPIO5)
fpioa.set_function(IO_EN,   FPIOA.GPIO7)

# GPIO Pins
dir_pin  = Pin(IO_DIR,  Pin.OUT)
step_pin = Pin(IO_STEP, Pin.OUT)
en_pin   = Pin(IO_EN,   Pin.OUT)

# Enable driver (active LOW)
en_pin.value(0)

# One Step
def step_once(delay_us):
    step_pin.value(1)
    time.sleep_us(delay_us)
    step_pin.value(0)
    time.sleep_us(delay_us)

# Forward Steps
def step_motor_forward(steps, delay_us):
    dir_pin.value(1)
    for _ in range(steps):
        step_once(delay_us)

# Main Loop
print("Stepper ready, forward only.")

while True:
    step_motor_forward(800, delay_us=1)
    time.sleep(.5)
