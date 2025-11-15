"""
GPIO Controller - Manage GPIO pins for PTT (Push-To-Talk) control

This module handles GPIO pin initialization, monitoring, and state changes
using the lgpio library for Raspberry Pi GPIO access.
"""
import lgpio
import threading
import time
from typing import Optional, Callable, Dict
from echostream import MAX_CHANNELS, global_interrupted


# ============================================================================
# GPIO Pin Mapping
# ============================================================================

# GPIO pin definitions (GPIO number → physical pin number)
GPIO_PINS = {
    20: 38,  # Channel 1 - GPIO 20 → Physical pin 38
    21: 40,  # Channel 2 - GPIO 21 → Physical pin 40
    23: 16,  # Channel 3 - GPIO 23 → Physical pin 16
    24: 18,  # Channel 4 - GPIO 24 → Physical pin 18
}

# Reverse mapping (physical pin → GPIO number)
PHYSICAL_TO_GPIO = {pin: gpio for gpio, pin in GPIO_PINS.items()}

# GPIO chip handle
gpio_chip: Optional[int] = None
gpio_mutex = threading.Lock()


# ============================================================================
# GPIO State Tracking
# ============================================================================

# Pin state storage (GPIO number → state: 0=active/low, 1=inactive/high)
gpio_states: Dict[int, int] = {}

# State change callbacks (GPIO number → callback function)
gpio_callbacks: Dict[int, Callable[[int, int], None]] = {}


# ============================================================================
# GPIO Initialization and Management
# ============================================================================

def init_gpio(chip: int = 0) -> bool:
    """
    Initialize GPIO chip.
    
    Args:
        chip: GPIO chip number (default: 0)
        
    Returns:
        True if initialization successful, False otherwise
    """
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
            
            # Initialize all pins
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
    """
    Initialize a single GPIO pin as input with pull-up resistor.
    
    Args:
        gpio_num: GPIO number to initialize
        
    Returns:
        True if initialization successful, False otherwise
    """
    global gpio_chip
    
    if gpio_chip is None:
        print("[GPIO] ERROR: GPIO chip not initialized")
        return False
    
    try:
        # Claim pin as input with pull-up resistor
        # Note: lgpio uses SET_PULL_UP constant
        result = lgpio.gpio_claim_input(gpio_chip, gpio_num, lgpio.SET_PULL_UP)
        if result != 0:
            print(f"[GPIO] ERROR: Failed to claim GPIO {gpio_num}: {result}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[GPIO] ERROR: Exception initializing GPIO {gpio_num}: {e}")
        return False


def read_pin(gpio_num: int) -> int:
    """
    Read GPIO pin value.
    
    Args:
        gpio_num: GPIO number to read
        
    Returns:
        0 if pin is low/active (PTT pressed), 1 if high/inactive (PTT released)
        -1 on error
    """
    global gpio_chip
    
    if gpio_chip is None:
        return -1
    
    try:
        value = lgpio.gpio_read(gpio_chip, gpio_num)
        # lgpio returns 0 for low, 1 for high (pull-up means active=low=0)
        return value
        
    except Exception as e:
        print(f"[GPIO] ERROR: Exception reading GPIO {gpio_num}: {e}")
        return -1


def register_callback(gpio_num: int, callback: Callable[[int, int], None]):
    """
    Register a callback function for GPIO pin state changes.
    
    Args:
        gpio_num: GPIO number to monitor
        callback: Callback function(gpio_num, new_state) to call on state change
    """
    with gpio_mutex:
        gpio_callbacks[gpio_num] = callback
        print(f"[GPIO] Registered callback for GPIO {gpio_num}")


def unregister_callback(gpio_num: int):
    """Unregister callback for a GPIO pin."""
    with gpio_mutex:
        if gpio_num in gpio_callbacks:
            del gpio_callbacks[gpio_num]
            print(f"[GPIO] Unregistered callback for GPIO {gpio_num}")


def get_pin_state(gpio_num: int) -> Optional[int]:
    """
    Get current state of a GPIO pin (from cache).
    
    Args:
        gpio_num: GPIO number
        
    Returns:
        Current state (0=active, 1=inactive) or None if unknown
    """
    with gpio_mutex:
        return gpio_states.get(gpio_num)


def get_gpio_for_channel(channel_index: int) -> Optional[int]:
    """
    Get GPIO number for a channel index.
    
    Args:
        channel_index: Channel index (0-3)
        
    Returns:
        GPIO number or None if invalid
    """
    gpio_list = list(GPIO_PINS.keys())
    if 0 <= channel_index < len(gpio_list):
        return gpio_list[channel_index]
    return None


def get_physical_pin(gpio_num: int) -> Optional[int]:
    """
    Get physical pin number for a GPIO number.
    
    Args:
        gpio_num: GPIO number
        
    Returns:
        Physical pin number or None if invalid
    """
    return GPIO_PINS.get(gpio_num)


# ============================================================================
# GPIO Monitor Worker Thread
# ============================================================================

def monitor_gpio(callback: Optional[Callable[[int, int], None]] = None, poll_interval: float = 0.1):
    """
    Monitor GPIO pins for state changes (worker thread function).
    
    Args:
        callback: Optional global callback for all pin changes
        poll_interval: Polling interval in seconds (default: 0.1 = 100ms)
    """
    print("[GPIO] GPIO monitor worker started")
    
    if gpio_chip is None:
        print("[GPIO] ERROR: GPIO chip not initialized, cannot monitor")
        return
    
    # Read initial states
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
                # Check each GPIO pin for state changes
                for gpio_num in GPIO_PINS.keys():
                    current_state = read_pin(gpio_num)
                    if current_state < 0:
                        continue  # Skip on error
                    
                    previous_state = gpio_states.get(gpio_num)
                    
                    if previous_state is None or current_state != previous_state:
                        # State changed
                        gpio_states[gpio_num] = current_state
                        pin_num = GPIO_PINS[gpio_num]
                        status = "ACTIVE" if current_state == 0 else "INACTIVE"
                        print(f"[GPIO] PIN {pin_num} (GPIO {gpio_num}): {status}")
                        
                        # Call registered callback if exists
                        if gpio_num in gpio_callbacks:
                            try:
                                gpio_callbacks[gpio_num](gpio_num, current_state)
                            except Exception as e:
                                print(f"[GPIO] ERROR: Exception in callback for GPIO {gpio_num}: {e}")
                        
                        # Call global callback if provided
                        if callback:
                            try:
                                callback(gpio_num, current_state)
                            except Exception as e:
                                print(f"[GPIO] ERROR: Exception in global callback: {e}")
            
            # Display status periodically
            status_counter += 1
            if status_counter >= 100:  # Every 10 seconds (100 * 0.1s)
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
    """Release GPIO resources and cleanup."""
    global gpio_chip, gpio_states, gpio_callbacks
    
    with gpio_mutex:
        if gpio_chip is not None:
            try:
                # Free all GPIO pins
                for gpio_num in GPIO_PINS.keys():
                    try:
                        lgpio.gpio_free(gpio_chip, gpio_num)
                    except Exception as e:
                        print(f"[GPIO] WARNING: Failed to free GPIO {gpio_num}: {e}")
                
                # Close GPIO chip
                lgpio.gpiochip_close(gpio_chip)
                print("[GPIO] GPIO chip closed")
                
            except Exception as e:
                print(f"[GPIO] ERROR: Exception during GPIO cleanup: {e}")
            
            gpio_chip = None
        
        gpio_states.clear()
        gpio_callbacks.clear()
