from machine import FPIOA, PWM
import time

# 1) Route PWM0 to IO60 explicitly (IOMUX)
fpioa = FPIOA()
fpioa.set_function(60, fpioa.PWM0)   # IO60 ← PWM0

# 2) Start PWM channel 0 at 50 Hz and center duty for servo
#    NOTE: constructor requires ALL 4 positional args on this firmware.
pwm0 = PWM(0, 50, 7.5, enable=True)         # (channel, freq Hz, duty %, enable)


# Optional: helper for angles (0–180° ≈ 5–10% duty)
def write_angle(deg):
    deg = max(0, min(180, deg))
    pwm0.duty(5 + (deg/180)*5)

# Sweep
write_angle(0);   time.sleep(1)
write_angle(90);  time.sleep(1)
write_angle(180); time.sleep(1)
write_angle(90);  time.sleep(1)
