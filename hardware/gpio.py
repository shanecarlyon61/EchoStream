import time
from typing import Dict, Optional, Callable

try:
    import lgpio
    HAS_LGPIO = True
except Exception:
    HAS_LGPIO = False
    lgpio = None

GPIO_PINS = {
    20: 38,
    21: 40,
    23: 16,
    24: 18,
}

gpio_chip: Optional[int] = None
gpio_states: Dict[int, int] = {}


def init_gpio(chip: int = 0) -> bool:
    global gpio_chip

    if not HAS_LGPIO:
        print("[GPIO] ERROR: lgpio module not available")
        return False

    if gpio_chip is not None:
        print("[GPIO] GPIO already initialized")
        return True

    try:
        gpio_chip = lgpio.gpiochip_open(chip)
        if gpio_chip < 0:
            print(f"[GPIO] ERROR: Failed to open GPIO chip {chip}")
            gpio_chip = None
            return False

        print(f"[GPIO] GPIO chip {chip} opened successfully")

        for gpio_num, pin_num in GPIO_PINS.items():
            if not init_gpio_pin(gpio_num):
                print(f"[GPIO] WARNING: Failed to initialize GPIO {gpio_num} (pin {pin_num})")
            else:
                state = read_pin(gpio_num)
                gpio_states[gpio_num] = state
                status = "ACTIVE" if state == 0 else ("INACTIVE" if state == 1 else "UNKNOWN")
                print(f"[GPIO] PIN {pin_num} (GPIO {gpio_num}) initial state: {status}")

        return True
    except Exception as e:
        print(f"[GPIO] ERROR: Exception during GPIO initialization: {e}")
        gpio_chip = None
        return False


def init_gpio_pin(gpio_num: int) -> bool:
    global gpio_chip
    if gpio_chip is None:
        return False
    try:
        res = lgpio.gpio_claim_input(gpio_chip, gpio_num, lgpio.SET_PULL_UP)
        if res != 0:
            print(f"[GPIO] ERROR: gpio_claim_input({gpio_num}) returned {res}")
            return False
        return True
    except Exception as e:
        print(f"[GPIO] ERROR: Exception initializing GPIO {gpio_num}: {e}")
        return False


def read_pin(gpio_num: int) -> int:
    global gpio_chip
    if gpio_chip is None:
        return -1
    try:
        return lgpio.gpio_read(gpio_chip, gpio_num)
    except Exception as e:
        print(f"[GPIO] ERROR: Exception reading GPIO {gpio_num}: {e}")
        return -1


def cleanup_gpio():
    global gpio_chip, gpio_states
    if gpio_chip is not None:
        try:
            for gpio_num in GPIO_PINS.keys():
                try:
                    lgpio.gpio_free(gpio_chip, gpio_num)
                except Exception:
                    pass
            lgpio.gpiochip_close(gpio_chip)
            print("[GPIO] GPIO chip closed")
        except Exception as e:
            print(f"[GPIO] WARNING: Exception during GPIO cleanup: {e}")
    gpio_chip = None
    gpio_states.clear()


def monitor_gpio(poll_interval: float = 0.1, status_every: int = 100,
                 on_change: Optional[Callable[[int, int], None]] = None):
    if gpio_chip is None:
        print("[GPIO] ERROR: GPIO not initialized")
        return

    print("[GPIO] Monitoring GPIO pins for changes...")
    count = 0
    try:
        while True:
            changed = []
            for gpio_num, pin_num in GPIO_PINS.items():
                val = read_pin(gpio_num)
                if val < 0:
                    continue
                prev = gpio_states.get(gpio_num)
                if prev is None or prev != val:
                    gpio_states[gpio_num] = val
                    status = "ACTIVE" if val == 0 else "INACTIVE"
                    print(f"[GPIO] PIN {pin_num} (GPIO {gpio_num}): {status}")
                    changed.append((gpio_num, val))
                    if on_change is not None:
                        try:
                            on_change(gpio_num, val)
                        except Exception as e:
                            print(f"[GPIO] WARNING: on_change callback error for GPIO {gpio_num}: {e}")

            count += 1
            if count >= status_every:
                print("\n=== GPIO Status Report ===")
                for gpio_num, pin_num in GPIO_PINS.items():
                    val = gpio_states.get(gpio_num, -1)
                    status = "ACTIVE" if val == 0 else ("INACTIVE" if val == 1 else "UNKNOWN")
                    print(f"PIN {pin_num} (GPIO {gpio_num}): {status}")
                print("=" * 30 + "\n")
                count = 0

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[GPIO] ERROR: Exception in monitor loop: {e}")

