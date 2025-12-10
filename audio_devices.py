import threading
from typing import List, Optional, Dict, Tuple

try:
    import pyaudio

    HAS_PYAUDIO = True
except Exception:
    pyaudio = None
    HAS_PYAUDIO = False

_pa_instance_lock = threading.Lock()
_pa_instance: Optional["pyaudio.PyAudio"] = None


def _get_pa() -> Optional["pyaudio.PyAudio"]:
    if not HAS_PYAUDIO:
        print("[AUDIO] ERROR: PyAudio not available")
        return None
    global _pa_instance
    if _pa_instance is None:
        with _pa_instance_lock:
            if _pa_instance is None:
                _pa_instance = pyaudio.PyAudio()
    return _pa_instance


def list_output_devices() -> List[Dict[str, str]]:
    pa = _get_pa()
    if pa is None:
        return []
    results: List[Dict[str, str]] = []
    try:
        count = pa.get_device_count()
        for i in range(count):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxOutputChannels", 0)) > 0:
                results.append(
                    {
                        "index": str(i),
                        "name": str(info.get("name", "")),
                        "host_api": str(info.get("hostApi", "")),
                    }
                )
    except Exception as e:
        print(f"[AUDIO] ERROR: Failed to enumerate devices: {e}")
    return results


def select_output_device_for_channel(channel_index: int) -> Optional[int]:
    """
    Pick an output device for a channel index using simple heuristics:
    - Prefer devices whose name contains 'USB' or 'Headphones' by channel order.
    - Fallback to default output device.
    """
    pa = _get_pa()
    if pa is None:
        return None
    try:
        preferred_keywords = ["USB"]
        devices = list_output_devices()
        if not devices:
            print(f"[AUDIO] WARNING: No output devices found for channel {channel_index}")
            return None
        print(f"[AUDIO] Found {len(devices)} output device(s) for selection")
        ranked: List[Tuple[int, int]] = []  # (device_index, rank)
        for d in devices:
            idx = int(d["index"])
            name = d["name"].upper()
            rank = 100
            for k in preferred_keywords:
                if k in name:
                    rank -= 10
            ranked.append((idx, rank))
        ranked.sort(key=lambda x: (x[1], x[0]))
        if ranked:
            # Distribute channels across ranked devices using modulo to wrap around
            chosen_idx = channel_index % len(ranked)
            chosen = ranked[chosen_idx][0]
            print(f"[AUDIO] Selected output device {chosen} for channel index {channel_index} (from {len(ranked)} available devices)")
            return chosen
        # Fallback to default
        return pa.get_default_output_device_info().get("index")  # type: ignore
    except Exception as e:
        print(f"[AUDIO] ERROR: Device selection failed: {e}")
        return None


def open_output_stream(
    device_index: int,
    sample_rate: int = 48000,
    num_channels: int = 1,
    frames_per_buffer: int = 1024,
):
    pa = _get_pa()
    if pa is None:
        return None, None
    
    # Check if device is available before attempting to open
    try:
        device_info = pa.get_device_info_by_index(device_index)
        if device_info.get('maxOutputChannels', 0) == 0:
            print(f"[AUDIO] ERROR: Device {device_index} has no output channels")
            return None, None
    except Exception as e:
        print(f"[AUDIO] ERROR: Device {device_index} not available: {e}")
        return None, None
    
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=num_channels,
            rate=sample_rate,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
        )
        return pa, stream
    except Exception as e:
        print(
            f"[AUDIO] ERROR: Failed to open output stream on device {device_index}: {e}"
        )
        return None, None


def list_input_devices() -> List[Dict[str, str]]:
    pa = _get_pa()
    if pa is None:
        return []
    results: List[Dict[str, str]] = []
    try:
        count = pa.get_device_count()
        for i in range(count):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                results.append(
                    {
                        "index": str(i),
                        "name": str(info.get("name", "")),
                        "host_api": str(info.get("hostApi", "")),
                    }
                )
    except Exception as e:
        print(f"[AUDIO] ERROR: Failed to enumerate input devices: {e}")
    return results


def select_input_device_for_channel(channel_index: int) -> Optional[int]:
    """
    Pick an input device for a channel index using simple heuristics:
    - Prefer devices whose name contains 'USB' by channel order.
    - Fallback to default input device.
    """
    pa = _get_pa()
    if pa is None:
        return None
    try:
        preferred_keywords = ["USB"]
        devices = list_input_devices()
        if not devices:
            print(f"[AUDIO] WARNING: No input devices found for channel {channel_index}")
            return None
        print(f"[AUDIO] Found {len(devices)} input device(s) for selection")
        ranked: List[Tuple[int, int]] = []  # (device_index, rank)
        for d in devices:
            idx = int(d["index"])
            name = d["name"].upper()
            rank = 100
            for k in preferred_keywords:
                if k in name:
                    rank -= 10
            ranked.append((idx, rank))
        ranked.sort(key=lambda x: (x[1], x[0]))
        if ranked:
            # Distribute channels across ranked devices using modulo to wrap around
            chosen_idx = channel_index % len(ranked)
            chosen = ranked[chosen_idx][0]
            print(f"[AUDIO] Selected input device {chosen} for channel index {channel_index} (from {len(ranked)} available devices)")
            return chosen
        # Fallback to default
        try:
            default_info = pa.get_default_input_device_info()
            default_idx = default_info.get("index")
            print(f"[AUDIO] Using default input device {default_idx} for channel index {channel_index}")
            return default_idx  # type: ignore
        except Exception:
            print(f"[AUDIO] WARNING: No default input device available for channel {channel_index}")
            return None
    except Exception as e:
        print(f"[AUDIO] ERROR: Input device selection failed: {e}")
        return None


def open_input_stream(
    device_index: int,
    sample_rate: int = 48000,
    num_channels: int = 1,
    frames_per_buffer: int = 1024,
):
    pa = _get_pa()
    if pa is None:
        return None, None
    
    # Validate device exists and supports input before attempting to open
    try:
        device_info = pa.get_device_info_by_index(device_index)
        max_input_channels = device_info.get('maxInputChannels', 0)
        if max_input_channels == 0:
            device_name = device_info.get('name', 'unknown')
            print(f"[AUDIO] ERROR: Device {device_index} ({device_name}) has no input channels")
            return None, None
    except Exception as e:
        print(f"[AUDIO] ERROR: Device {device_index} not available or invalid: {e}")
        return None, None
    
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,  # 32-bit float like C code
            channels=num_channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frames_per_buffer,
        )
        return pa, stream
    except Exception as e:
        print(
            f"[AUDIO] ERROR: Failed to open input stream on device {device_index}: {e}"
        )
        return None, None


def close_stream(pa: "pyaudio.PyAudio", stream) -> None:
    try:
        if stream:
            stream.stop_stream()
            stream.close()
    except Exception:
        pass
    try:
        # Keep global instance for reuse; do not terminate here
        pass
    except Exception:
        pass
