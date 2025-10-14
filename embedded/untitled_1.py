# main.py — CanMV K230 "deepest possible" low-power with MicroPython
# This is NOT true deep sleep; it aggressively powers down peripherals and DVFS.
import utime as time
import uos as os
import sys

SLEEP_MS = 60_000  # change as needed
RESTORE_AFTER = True  # set False to stay throttled after sleep
LED_PINS = []  # e.g., ["PA12", "PB3"] if you know your LED/backlight GPIOs

def log(*a):
    try:
        print(*a)
    except:
        pass

# ----------------- Power management helpers (CPU/KPU) -----------------
def list_profiles(kind="cpu"):
    try:
        from mpp import pm
        if kind == "cpu":
            return pm.cpu.list_profiles()
        else:
            return pm.kpu.list_profiles()
    except Exception as e:
        return None

def get_profile(kind="cpu"):
    try:
        from mpp import pm
        if kind == "cpu":
            return pm.cpu.get_freq()
        else:
            return pm.kpu.get_freq()
    except:
        return None

def set_profile(idx, kind="cpu"):
    try:
        from mpp import pm
        if kind == "cpu":
            pm.cpu.set_profile(idx)
        else:
            pm.kpu.set_profile(idx)
        return True
    except Exception as e:
        log("WARN: set_profile failed for", kind, idx, e)
        return False

def go_min_profiles():
    profs_cpu = list_profiles("cpu")
    profs_kpu = list_profiles("kpu")
    changed = {"cpu": None, "kpu": None}
    if profs_cpu:
        min_idx = len(profs_cpu) - 1
        set_profile(min_idx, "cpu")
        changed["cpu"] = min_idx
    if profs_kpu:
        min_idx = len(profs_kpu) - 1
        set_profile(min_idx, "kpu")
        changed["kpu"] = min_idx
    return changed

def restore_profiles():
    # conventionally index 0 is highest/default
    set_profile(0, "cpu")
    set_profile(0, "kpu")

# ----------------- Peripherals: displays/backlights -----------------
def blank_displays():
    did = False
    # Try CanMV/CanMV-lcd style
    try:
        import lcd
        try:
            if hasattr(lcd, "backlight"):
                lcd.backlight(0)
                did = True
        except:
            pass
        try:
            if hasattr(lcd, "clear"):
                lcd.clear(0)  # helps some panels reduce load
        except:
            pass
    except:
        pass
    # Try generic display module
    try:
        import display
        for name in ("set_backlight", "backlight", "brightness", "set_brightness"):
            if hasattr(display, name):
                try:
                    getattr(display, name)(0)
                    did = True
                except:
                    pass
    except:
        pass
    # Try ST77xx/ILI9xxx style (if user imported earlier)
    for modname in ("st7789", "st7735", "ili9341"):
        try:
            m = __import__(modname)
            for attr in ("backlight", "set_backlight", "brightness", "set_brightness"):
                if hasattr(m, attr):
                    try:
                        getattr(m, attr)(0)
                        did = True
                    except:
                        pass
        except:
            pass
    return did

# ----------------- Camera / sensor -----------------
def shutdown_sensor():
    try:
        import sensor
        # Try the strictest off first
        if hasattr(sensor, "shutdown"):
            sensor.shutdown(True)
            return True
        # Fallback to sleep if available
        if hasattr(sensor, "sleep"):
            sensor.sleep(True)
            return True
        # Some forks expose sensor.reset_power / set_power
        for name in ("set_power", "power", "reset_power"):
            if hasattr(sensor, name):
                try:
                    getattr(sensor, name)(False)
                    return True
                except:
                    pass
    except:
        pass
    return False

# ----------------- GPIOs (LEDs / misc rails) -----------------
def drive_pins_low(pin_names):
    ok = []
    try:
        from machine import Pin
    except:
        return ok
    for p in pin_names:
        try:
            pin = Pin(p, Pin.OUT)
            pin.value(0)
            ok.append(p)
        except Exception as e:
            log("WARN: couldn't drive pin", p, "low:", e)
    return ok

# ----------------- Main -----------------
def main():
    log("\n=== K230 Low-Power (MicroPython) ===")

    # 1) Shut down high-draw peripherals FIRST
    disp = blank_displays()
    if disp:
        log("Display/backlight: OFF")
    else:
        log("Display/backlight: not found or already off")

    cams = shutdown_sensor()
    log("Camera sensor shutdown:", cams)

    if LED_PINS:
        low = drive_pins_low(LED_PINS)
        log("LED/GPIO forced low:", low)

    # 2) Drop CPU/KPU to lowest profiles
    before_cpu = get_profile("cpu")
    before_kpu = get_profile("kpu")
    changed = go_min_profiles()
    after_cpu = get_profile("cpu")
    after_kpu = get_profile("kpu")
    log("CPU freq before/after:", before_cpu, "->", after_cpu)
    log("KPU freq before/after:", before_kpu, "->", after_kpu)
    log("Profiles set:", changed)

    # 3) Optional: sync FS, then idle sleep
    try:
        os.sync()
    except:
        pass
    log("Sleeping (low-power) for", SLEEP_MS, "ms ...")
    time.sleep_ms(SLEEP_MS)

    # 4) Restore if requested
    if RESTORE_AFTER:
        restore_profiles()
        blank_displays()  # keep off even after restore
        # Do not auto-wake sensor; keep power down unless user wants it
        log("Restored CPU/KPU to profile 0, peripherals remain off.")

    log("Done.")

if __name__ == "__main__":
    main()
