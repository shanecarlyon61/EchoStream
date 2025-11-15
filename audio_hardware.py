"""
Audio Hardware Manager - Manage PyAudio/ALSA audio devices

This module handles PyAudio initialization, USB audio device detection,
and device assignment to channels.
"""
import pyaudio
from typing import Optional, List, Dict, Tuple
from echostream import MAX_CHANNELS


# ============================================================================
# Global State
# ============================================================================

# PyAudio instance
pa_instance: Optional[pyaudio.PyAudio] = None

# USB device mapping (channel_index → device_index)
usb_devices: List[int] = [-1] * MAX_CHANNELS

# Device assignment flag
device_assigned = False


# ============================================================================
# PyAudio Initialization
# ============================================================================

def init_portaudio() -> bool:
    """
    Initialize PyAudio instance.
    
    Returns:
        True if initialization successful, False otherwise
    """
    global pa_instance
    
    if pa_instance is not None:
        print("[AUDIO_HW] PyAudio already initialized")
        return True
    
    try:
        pa_instance = pyaudio.PyAudio()
        print("[AUDIO_HW] PyAudio initialized successfully")
        return True
        
    except Exception as e:
        print(f"[AUDIO_HW] ERROR: Failed to initialize PyAudio: {e}")
        pa_instance = None
        return False


def cleanup_portaudio():
    """Terminate PyAudio instance and cleanup."""
    global pa_instance
    
    if pa_instance is not None:
        try:
            pa_instance.terminate()
            print("[AUDIO_HW] PyAudio terminated")
        except Exception as e:
            print(f"[AUDIO_HW] ERROR: Exception during PyAudio cleanup: {e}")
        finally:
            pa_instance = None


# ============================================================================
# Device Detection and Management
# ============================================================================

def detect_usb_devices() -> int:
    """
    Scan for USB audio devices and assign them to channel slots.
    
    Returns:
        Number of USB devices found and assigned
    """
    global pa_instance, usb_devices, device_assigned
    
    if pa_instance is None:
        print("[AUDIO_HW] ERROR: PyAudio not initialized")
        return 0
    
    if device_assigned:
        print("[AUDIO_HW] USB devices already assigned")
        return sum(1 for dev in usb_devices if dev >= 0)
    
    print("[AUDIO_HW] Scanning for USB audio devices...")
    
    try:
        num_devices = pa_instance.get_device_count()
        print(f"[AUDIO_HW] Found {num_devices} audio device(s) total")
        
        usb_count = 0
        
        # Scan for USB audio devices
        for i in range(num_devices):
            if usb_count >= MAX_CHANNELS:
                break
            
            try:
                device_info = pa_instance.get_device_info_by_index(i)
                
                # Check if device has input capability
                if device_info['maxInputChannels'] > 0:
                    name = device_info['name'].lower()
                    
                    # Check if it's a USB audio device
                    if 'usb' in name or 'audio device' in name.lower() or 'headset' in name.lower():
                        usb_devices[usb_count] = i
                        print(f"[AUDIO_HW] USB Device {i} assigned to slot {usb_count}: {device_info['name']} (hw:{device_info.get('hostApi', 'unknown')},{device_info.get('index', 'unknown')})")
                        usb_count += 1
                        
            except Exception as e:
                # Skip devices that can't be queried
                continue
        
        # If no USB devices found, use default input device
        if usb_count == 0:
            try:
                default_device = pa_instance.get_default_input_device_info()
                default_index = default_device['index']
                
                print(f"[AUDIO_HW] No USB devices found, using default input device: {default_device['name']}")
                
                # Assign default device to all channels
                for i in range(MAX_CHANNELS):
                    usb_devices[i] = default_index
                usb_count = MAX_CHANNELS
                
            except Exception as e:
                print(f"[AUDIO_HW] ERROR: Failed to get default input device: {e}")
                return 0
        
        # If fewer USB devices than channels, cycle assignment
        elif usb_count < MAX_CHANNELS:
            print(f"[AUDIO_HW] Only {usb_count} USB device(s) found, some channels will share devices")
            for i in range(usb_count, MAX_CHANNELS):
                usb_devices[i] = usb_devices[i % usb_count]
        
        device_assigned = True
        
        # Print channel assignments
        print("[AUDIO_HW] Channel assignments:")
        for i in range(MAX_CHANNELS):
            if usb_devices[i] >= 0:
                try:
                    device_info = pa_instance.get_device_info_by_index(usb_devices[i])
                    print(f"[AUDIO_HW]   Channel {i + 1} → Device {usb_devices[i]}: {device_info['name']}")
                except Exception:
                    print(f"[AUDIO_HW]   Channel {i + 1} → Device {usb_devices[i]}: (unknown)")
        
        return usb_count
        
    except Exception as e:
        print(f"[AUDIO_HW] ERROR: Exception during USB device detection: {e}")
        import traceback
        traceback.print_exc()
        return 0


def assign_device_to_channel(channel_id: str, channel_index: int) -> bool:
    """
    Assign a device to a channel by channel ID and index.
    
    Args:
        channel_id: Channel ID string
        channel_index: Channel index (0-3)
        
    Returns:
        True if assignment successful, False otherwise
    """
    global usb_devices, device_assigned
    
    if not device_assigned:
        detect_usb_devices()
    
    if 0 <= channel_index < MAX_CHANNELS:
        device_index = usb_devices[channel_index]
        if device_index >= 0:
            print(f"[AUDIO_HW] Assigned device {device_index} to channel {channel_index + 1} ({channel_id})")
            return True
    
    print(f"[AUDIO_HW] WARNING: No device available for channel {channel_index + 1} ({channel_id})")
    return False


def get_device_for_channel(channel_index: int) -> int:
    """
    Get device index for a channel.
    
    Args:
        channel_index: Channel index (0-3)
        
    Returns:
        Device index, or -1 if not assigned
    """
    global usb_devices, device_assigned
    
    if not device_assigned:
        detect_usb_devices()
    
    if 0 <= channel_index < MAX_CHANNELS:
        return usb_devices[channel_index]
    
    return -1


def get_device_info(device_index: int) -> Optional[Dict]:
    """
    Get information about a device.
    
    Args:
        device_index: Device index
        
    Returns:
        Device info dictionary or None if error
    """
    global pa_instance
    
    if pa_instance is None:
        return None
    
    try:
        return pa_instance.get_device_info_by_index(device_index)
    except Exception as e:
        print(f"[AUDIO_HW] ERROR: Failed to get device info for index {device_index}: {e}")
        return None


def list_all_devices() -> List[Dict]:
    """
    List all available audio devices.
    
    Returns:
        List of device info dictionaries
    """
    global pa_instance
    
    if pa_instance is None:
        return []
    
    devices = []
    try:
        num_devices = pa_instance.get_device_count()
        for i in range(num_devices):
            try:
                device_info = pa_instance.get_device_info_by_index(i)
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'maxInputChannels': device_info['maxInputChannels'],
                    'maxOutputChannels': device_info['maxOutputChannels'],
                    'defaultSampleRate': device_info['defaultSampleRate'],
                    'hostApi': device_info.get('hostApi', 'unknown')
                })
            except Exception:
                continue
    except Exception as e:
        print(f"[AUDIO_HW] ERROR: Failed to list devices: {e}")
    
    return devices


def reset_device_assignment():
    """Reset device assignment (useful for testing)."""
    global device_assigned, usb_devices
    
    device_assigned = False
    usb_devices = [-1] * MAX_CHANNELS
    print("[AUDIO_HW] Device assignment reset")

