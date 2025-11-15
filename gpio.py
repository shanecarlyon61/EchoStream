
import lgpio
import threading
import time
from typing import Optional, Callable, Dict
from echostream import MAX_CHANNELS, global_interrupted

GPIO_PINS = {
    20: 38,
    21: 40,
    23: 16,
    24: 18,
}

PHYSICAL_TO_GPIO = {pin: gpio for gpio, pin in GPIO_PINS.items()}

gpio_chip: Optional[int] = None
gpio_mutex = threading.Lock()

gpio_states: Dict[int, int] = {}

gpio_callbacks: Dict[int, Callable[[int, int], None]] = {}

def init_gpio(chip: int = 0) -> bool:

    global gpio_chip

    with gpio_mutex:
        if gpio_chip is not None:
            print("[GPIO] GPIO already initialized")
            return True

        try:
            gpio_chip = lgpio.gpiochip_open(chip)
            if gpio_chip < 0:
                print(f"[GPIO] ERROR: Failed to open GPIO chip {chip}")
                return False

            print(f"[GPIO] GPIO chip {chip} opened successfully")

            for gpio_num, pin_num in GPIO_PINS.items():
                if init_gpio_pin(gpio_num):
                    gpio_states[gpio_num] = read_pin(gpio_num)
                    print(f"[GPIO] PIN {pin_num} (GPIO {gpio_num}) initialized: {'ACTIVE' if gpio_states[gpio_num] == 0 else 'INACTIVE'}")
                else:
                    print(f"[GPIO] WARNING: Failed to initialize GPIO {gpio_num}")

            return True

        except Exception as e:
            print(f"[GPIO] ERROR: Exception during GPIO initialization: {e}")
            gpio_chip = None
            return False

def init_gpio_pin(gpio_num: int) -> bool:

    global gpio_chip

    if gpio_chip is None:
        print("[GPIO] ERROR: GPIO chip not initialized")
        return False

    try:

        result = lgpio.gpio_claim_input(gpio_chip, gpio_num, lgpio.SET_PULL_UP)
        if result != 0:
            print(f"[GPIO] ERROR: Failed to claim GPIO {gpio_num}: {result}")
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
        value = lgpio.gpio_read(gpio_chip, gpio_num)

        return value

    except Exception as e:
        print(f"[GPIO] ERROR: Exception reading GPIO {gpio_num}: {e}")
        return -1

def register_callback(gpio_num: int, callback: Callable[[int, int], None]):

    with gpio_mutex:
        gpio_callbacks[gpio_num] = callback
        print(f"[GPIO] Registered callback for GPIO {gpio_num}")

def unregister_callback(gpio_num: int):
    with gpio_mutex:
        if gpio_num in gpio_callbacks:
            del gpio_callbacks[gpio_num]
            print(f"[GPIO] Unregistered callback for GPIO {gpio_num}")

def get_pin_state(gpio_num: int) -> Optional[int]:

    with gpio_mutex:
        return gpio_states.get(gpio_num)

def get_gpio_for_channel(channel_index: int) -> Optional[int]:

    gpio_list = list(GPIO_PINS.keys())
    if 0 <= channel_index < len(gpio_list):
        return gpio_list[channel_index]
    return None

def get_physical_pin(gpio_num: int) -> Optional[int]:

    return GPIO_PINS.get(gpio_num)

def monitor_gpio(callback: Optional[Callable[[int, int], None]] = None, poll_interval: float = 0.1):

    print("[GPIO] GPIO monitor worker started")

    if gpio_chip is None:
        print("[GPIO] ERROR: GPIO chip not initialized, cannot monitor")
        return

    with gpio_mutex:
        for gpio_num in GPIO_PINS.keys():
            state = read_pin(gpio_num)
            if state >= 0:
                gpio_states[gpio_num] = state
                pin_num = GPIO_PINS[gpio_num]
                status = "ACTIVE" if state == 0 else "INACTIVE"
                print(f"[GPIO] PIN {pin_num} (GPIO {gpio_num}) initial state: {status}")

    print("[GPIO] Monitoring GPIO pins for changes...")

    status_counter = 0

    while not global_interrupted.is_set():
        try:
            with gpio_mutex:

                for gpio_num in GPIO_PINS.keys():
                    current_state = read_pin(gpio_num)
                    if current_state < 0:
                        continue

                    previous_state = gpio_states.get(gpio_num)

                    if previous_state is None or current_state != previous_state:

                        gpio_states[gpio_num] = current_state
                        pin_num = GPIO_PINS[gpio_num]
                        status = "ACTIVE" if current_state == 0 else "INACTIVE"
                        print(f"[GPIO] PIN {pin_num} (GPIO {gpio_num}): {status}")

                        if gpio_num in gpio_callbacks:
                            try:
                                gpio_callbacks[gpio_num](gpio_num, current_state)
                            except Exception as e:
                                print(f"[GPIO] ERROR: Exception in callback for GPIO {gpio_num}: {e}")

                        if callback:
                            try:
                                callback(gpio_num, current_state)
                            except Exception as e:
                                print(f"[GPIO] ERROR: Exception in global callback: {e}")

            status_counter += 1
            if status_counter >= 100:
                print("\n=== GPIO Status Report ===")
                with gpio_mutex:
                    for gpio_num, pin_num in GPIO_PINS.items():
                        state = gpio_states.get(gpio_num, -1)
                        status = "ACTIVE" if state == 0 else ("INACTIVE" if state == 1 else "UNKNOWN")
                        print(f"PIN {pin_num} (GPIO {gpio_num}): {status}")
                print("=" * 30 + "\n")
                status_counter = 0

            time.sleep(poll_interval)

        except Exception as e:
            if not global_interrupted.is_set():
                print(f"[GPIO] ERROR: Exception in monitor loop: {e}")
            time.sleep(poll_interval)

    print("[GPIO] GPIO monitor worker stopped")

def cleanup_gpio():
    global gpio_chip, gpio_states, gpio_callbacks

    with gpio_mutex:
        if gpio_chip is not None:
            try:

                for gpio_num in GPIO_PINS.keys():
                    try:
                        lgpio.gpio_free(gpio_chip, gpio_num)
                    except Exception as e:
                        print(f"[GPIO] WARNING: Failed to free GPIO {gpio_num}: {e}")

                lgpio.gpiochip_close(gpio_chip)
                print("[GPIO] GPIO chip closed")

            except Exception as e:
                print(f"[GPIO] ERROR: Exception during GPIO cleanup: {e}")

            gpio_chip = None

        gpio_states.clear()
        gpio_callbacks.clear()
