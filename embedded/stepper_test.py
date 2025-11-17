from machine import FPIOA, GPIO
import time

# ---------------- Pin assignments ----------------
PIN_STEP = 20
PIN_DIR  = 21
PIN_EN   = 22

# ---------------- FPIOA routing ----------------
fpioa = FPIOA()
fpioa.set_function(PIN_STEP, fpioa.GPIO0)
fpioa.set_function(PIN_DIR,  fpioa.GPIO1)
fpioa.set_function(PIN_EN,   fpioa.GPIO2)

# ---------------- Setup GPIO ----------------
step_pin = GPIO(GPIO.GPIO0, GPIO.OUT)
dir_pin  = GPIO(GPIO.GPIO1, GPIO.OUT)
en_pin   = GPIO(GPIO.GPIO2, GPIO.OUT)

# Enable: LOW = on for most drivers
en_pin.value(0)

# ---------------- Basic Helpers ----------------

def set_direction(clockwise=True):
    dir_pin.value(0 if clockwise else 1)

def enable_motor():
    en_pin.value(0)

def disable_motor():
    en_pin.value(1)

def step_once(delay_us=500):
    """
    Issue one step pulse.
    delay_us = pulse width (and speed control)
    """
    step_pin.value(1)
    time.sleep_us(delay_us)
    step_pin.value(0)
    time.sleep_us(delay_us)

def step_n(count, delay_us=500):
    """
    Move a fixed number of steps using manual pulses.
    """
    for _ in range(count):
        step_once(delay_us)

# ---------------- Example usage ----------------

enable_motor()

# Move 200 steps clockwise, medium speed
set_direction(clockwise=True)
step_n(200, delay_us=500)

time.sleep(1)

# Move 400 steps CCW, slower
set_direction(clockwise=False)
step_n(400, delay_us=1000)

disable_motor()
