
import json
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from echostream import MAX_CHANNELS, CHANNEL_ID_LEN, MAX_TONE_DEFINITIONS

@dataclass
class ToneDefinition:
    tone_id: str = ""
    tone_a_freq: float = 0.0
    tone_b_freq: float = 0.0
    tone_a_length_ms: int = 0
    tone_b_length_ms: int = 0
    tone_a_range_hz: int = 0
    tone_b_range_hz: int = 0
    record_length_ms: int = 0
    detection_tone_alert: str = ""
    valid: bool = False

@dataclass
class ToneDetectConfig:
    tone_passthrough: bool = False
    passthrough_channel: str = ""
    threshold: float = 0.0
    gain: float = 0.0
    db_threshold: int = -20
    detect_new_tones: bool = False
    new_tone_length_ms: int = 0
    new_tone_range_hz: int = 0
    valid: bool = False

@dataclass
class ChannelConfig:
    channel_id: str = ""
    input_low_one: bool = False
    input_low_two: bool = False
    input_high_one: bool = False
    input_high_two: bool = False
    tone_detect: bool = False
    tone_config: ToneDetectConfig = field(default_factory=ToneDetectConfig)
    valid: bool = False

global_app_config = {
    'channels': [ChannelConfig() for _ in range(MAX_CHANNELS)],
    'valid': False
}

def load_config(path: str = "/home/will/.an/config.json") -> bool:

    print(f"[CONFIG] Loading configuration from: {path}")

    try:
        if not os.path.exists(path):
            print(f"[CONFIG] ERROR: Config file not found: {path}")
            return False

        with open(path, 'r') as file:
            json_data = json.load(file)

        print(f"[CONFIG] Config file opened successfully ({os.path.getsize(path)} bytes)")

        shadow = json_data.get('shadow', {})
        state = shadow.get('state', {})
        desired = state.get('desired', {})
        software_config = desired.get('software_configuration', [])

        if not software_config or len(software_config) == 0:
            print("[CONFIG] ERROR: Could not find software_configuration in config")
            return False

        config_item = software_config[0]

        channel_keys = ["channel_one", "channel_two", "channel_three", "channel_four"]
        channels_loaded = 0

        for i, key in enumerate(channel_keys):
            if i >= MAX_CHANNELS:
                break

            channel_obj = config_item.get(key, {})
            if not channel_obj:
                continue

            channel_config = global_app_config['channels'][i]

            channel_id = channel_obj.get('channel_id', '')
            if channel_id and len(channel_id) < CHANNEL_ID_LEN:
                channel_config.channel_id = channel_id

            channel_config.input_low_one = channel_obj.get('input_low_one', False)
            channel_config.input_low_two = channel_obj.get('input_low_two', False)
            channel_config.input_high_one = channel_obj.get('input_high_one', False)
            channel_config.input_high_two = channel_obj.get('input_high_two', False)

            channel_config.tone_detect = channel_obj.get('tone_detect', False)

            if channel_config.tone_detect:
                _load_tone_detection_config(channel_config, channel_obj, i)

            channel_config.valid = True
            channels_loaded += 1
            print(f"[CONFIG] Loaded channel {i + 1}: ID={channel_config.channel_id}, tone_detect={channel_config.tone_detect}")

        if channels_loaded > 0:
            global_app_config['valid'] = True
            print(f"[CONFIG] Successfully loaded configuration for {channels_loaded} channel(s)")
            return True
        else:
            print("[CONFIG] WARNING: No channel configurations loaded")
            return False

    except FileNotFoundError:
        print(f"[CONFIG] ERROR: Could not open config file {path}")
        return False
    except json.JSONDecodeError as e:
        print(f"[CONFIG] ERROR: Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return False

def _load_tone_detection_config(channel_config: ChannelConfig, channel_obj: Dict[str, Any], channel_index: int):
    tone_detect_config_obj = channel_obj.get('tone_detect_configuration', {})
    if not tone_detect_config_obj:
        print(f"[CONFIG] WARNING: No tone_detect_configuration for channel {channel_index + 1}")
        return

    tone_config = channel_config.tone_config

    tone_config.tone_passthrough = tone_detect_config_obj.get('tone_passthrough', False)
    tone_config.passthrough_channel = tone_detect_config_obj.get('passthrough_channel', '')

    alert_details = tone_detect_config_obj.get('alert_details', {})
    if alert_details:
        tone_config.threshold = float(alert_details.get('threshold', 0.0))
        tone_config.gain = float(alert_details.get('gain', 0.0))
        tone_config.db_threshold = int(alert_details.get('db', -20))
        tone_config.detect_new_tones = alert_details.get('detect_new_tones', False)
        tone_config.new_tone_length_ms = int(alert_details.get('new_tone_length', 0))
        tone_config.new_tone_range_hz = int(alert_details.get('new_tone_range', 0))

    alert_tones = tone_detect_config_obj.get('alert_tones', [])
    if alert_tones:
        print(f"[CONFIG] Found {len(alert_tones)} tone definition(s) for channel {channel_index + 1}")

        try:
            import tone_detection
            tones_loaded = 0

            for tone_obj in alert_tones:
                tone_id = tone_obj.get('tone_id', '')
                tone_a = float(tone_obj.get('tone_a', 0.0))
                tone_b = float(tone_obj.get('tone_b', 0.0))

                tone_a_length = int(float(tone_obj.get('tone_a_length', 0.0)) * 1000)
                tone_b_length = int(float(tone_obj.get('tone_b_length', 0.0)) * 1000)
                tone_a_range = int(tone_obj.get('tone_a_range', 0))
                tone_b_range = int(tone_obj.get('tone_b_range', 0))

                record_length = int(tone_obj.get('record_length', 0)) * 1000
                detection_tone_alert = tone_obj.get('detection_tone_alert', '')

                if tone_id and tone_a > 0 and tone_b > 0:
                    if tone_detection.add_tone_definition(
                        tone_id, tone_a, tone_b,
                        tone_a_length, tone_b_length,
                        tone_a_range, tone_b_range,
                        record_length,
                        detection_tone_alert if detection_tone_alert else None
                    ):
                        tones_loaded += 1
                        print(f"[CONFIG] Loaded tone: ID={tone_id}, A={tone_a}Hz±{tone_a_range} ({tone_a_length}ms), B={tone_b}Hz±{tone_b_range} ({tone_b_length}ms)")

            if tones_loaded > 0:
                print(f"[CONFIG] Successfully loaded {tones_loaded} tone definition(s) for channel {channel_index + 1}")
        except ImportError as e:
            print(f"[CONFIG] WARNING: tone_detection module not available: {e}")
    else:
        print(f"[CONFIG] WARNING: No alert_tones array found for channel {channel_index + 1}")

    tone_config.valid = True
    print(f"[CONFIG] Tone detection config for channel {channel_index + 1}: passthrough={tone_config.tone_passthrough}, target={tone_config.passthrough_channel}")

def load_channel_config(channel_ids: List[str]) -> int:

    if not load_config():
        return 0

    loaded = 0
    for i in range(MAX_CHANNELS):
        channel_config = get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.channel_id:
            channel_ids[i] = channel_config.channel_id
            loaded += 1
            print(f"[CONFIG] Loaded channel {i + 1} ID: {channel_ids[i]}")

    return loaded

def load_complete_config() -> bool:

    return load_config()

def get_channel_config(channel_index: int) -> Optional[ChannelConfig]:

    if 0 <= channel_index < MAX_CHANNELS and global_app_config['valid']:
        return global_app_config['channels'][channel_index]
    return None

def get_tone_detect_config(channel_index: int) -> Optional[ToneDetectConfig]:

    channel_config = get_channel_config(channel_index)
    if channel_config and channel_config.tone_detect and channel_config.tone_config.valid:
        return channel_config.tone_config
    return None

def validate_config() -> bool:

    if not global_app_config['valid']:
        return False

    for i in range(MAX_CHANNELS):
        channel_config = global_app_config['channels'][i]
        if channel_config.valid and channel_config.channel_id:
            return True

    return False
