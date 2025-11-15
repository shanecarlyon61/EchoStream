"""
Configuration management - loads channel and tone detection settings from JSON

This module handles loading and managing configuration from the JSON config file.
All configuration data structures are defined here.
"""
import json
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# Import constants
from echostream import MAX_CHANNELS, CHANNEL_ID_LEN, MAX_TONE_DEFINITIONS


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class ToneDefinition:
    """Tone definition for DTMF-style tone detection."""
    tone_id: str = ""
    tone_a_freq: float = 0.0  # Frequency of tone A in Hz
    tone_b_freq: float = 0.0  # Frequency of tone B in Hz
    tone_a_length_ms: int = 0  # Required duration of tone A in ms
    tone_b_length_ms: int = 0  # Required duration of tone B in ms
    tone_a_range_hz: int = 0  # Frequency tolerance for tone A in Hz
    tone_b_range_hz: int = 0  # Frequency tolerance for tone B in Hz
    record_length_ms: int = 0  # Recording duration after detection in ms
    detection_tone_alert: str = ""  # Alert type identifier
    valid: bool = False


@dataclass
class ToneDetectConfig:
    """Tone detection configuration for a channel."""
    tone_passthrough: bool = False  # Enable passthrough routing
    passthrough_channel: str = ""  # Target channel for passthrough
    threshold: float = 0.0  # Detection threshold
    gain: float = 0.0  # Audio gain
    db_threshold: int = -20  # dB threshold for volume check
    detect_new_tones: bool = False  # Enable new tone detection
    new_tone_length_ms: int = 0  # Length for new tones
    new_tone_range_hz: int = 0  # Range for new tones
    valid: bool = False


@dataclass
class ChannelConfig:
    """Channel configuration."""
    channel_id: str = ""
    input_low_one: bool = False
    input_low_two: bool = False
    input_high_one: bool = False
    input_high_two: bool = False
    tone_detect: bool = False  # Enable tone detection for this channel
    tone_config: ToneDetectConfig = field(default_factory=ToneDetectConfig)
    valid: bool = False


# ============================================================================
# Global Configuration Storage
# ============================================================================

# Global application configuration
global_app_config = {
    'channels': [ChannelConfig() for _ in range(MAX_CHANNELS)],
    'valid': False
}


# ============================================================================
# Configuration Loading Functions
# ============================================================================

def load_config(path: str = "/home/will/.an/config.json") -> bool:
    """
    Load complete configuration from JSON file.
    
    Args:
        path: Path to configuration JSON file
        
    Returns:
        True if configuration loaded successfully, False otherwise
    """
    print(f"[CONFIG] Loading configuration from: {path}")
    
    try:
        if not os.path.exists(path):
            print(f"[CONFIG] ERROR: Config file not found: {path}")
            return False
        
        with open(path, 'r') as file:
            json_data = json.load(file)
        
        print(f"[CONFIG] Config file opened successfully ({os.path.getsize(path)} bytes)")
        
        # Navigate to the channel configuration
        shadow = json_data.get('shadow', {})
        state = shadow.get('state', {})
        desired = state.get('desired', {})
        software_config = desired.get('software_configuration', [])
        
        if not software_config or len(software_config) == 0:
            print("[CONFIG] ERROR: Could not find software_configuration in config")
            return False
        
        config_item = software_config[0]
        
        # Extract channel configurations
        channel_keys = ["channel_one", "channel_two", "channel_three", "channel_four"]
        channels_loaded = 0
        
        for i, key in enumerate(channel_keys):
            if i >= MAX_CHANNELS:
                break
                
            channel_obj = config_item.get(key, {})
            if not channel_obj:
                continue
            
            channel_config = global_app_config['channels'][i]
            
            # Load basic channel info
            channel_id = channel_obj.get('channel_id', '')
            if channel_id and len(channel_id) < CHANNEL_ID_LEN:
                channel_config.channel_id = channel_id
            
            # Load input settings
            channel_config.input_low_one = channel_obj.get('input_low_one', False)
            channel_config.input_low_two = channel_obj.get('input_low_two', False)
            channel_config.input_high_one = channel_obj.get('input_high_one', False)
            channel_config.input_high_two = channel_obj.get('input_high_two', False)
            
            # Load tone detection settings
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
    """Helper function to load tone detection configuration."""
    tone_detect_config_obj = channel_obj.get('tone_detect_configuration', {})
    if not tone_detect_config_obj:
        print(f"[CONFIG] WARNING: No tone_detect_configuration for channel {channel_index + 1}")
        return
    
    tone_config = channel_config.tone_config
    
    # Load tone passthrough settings
    tone_config.tone_passthrough = tone_detect_config_obj.get('tone_passthrough', False)
    tone_config.passthrough_channel = tone_detect_config_obj.get('passthrough_channel', '')
    
    # Load alert details
    alert_details = tone_detect_config_obj.get('alert_details', {})
    if alert_details:
        tone_config.threshold = float(alert_details.get('threshold', 0.0))
        tone_config.gain = float(alert_details.get('gain', 0.0))
        tone_config.db_threshold = int(alert_details.get('db', -20))
        tone_config.detect_new_tones = alert_details.get('detect_new_tones', False)
        tone_config.new_tone_length_ms = int(alert_details.get('new_tone_length', 0))
        tone_config.new_tone_range_hz = int(alert_details.get('new_tone_range', 0))
    
    # Load tone definitions (alert_tones array)
    alert_tones = tone_detect_config_obj.get('alert_tones', [])
    if alert_tones:
        print(f"[CONFIG] Found {len(alert_tones)} tone definition(s) for channel {channel_index + 1}")
        
        # Import tone_detect module (avoid circular import)
        try:
            import tone_detect
            tones_loaded = 0
            
            for tone_obj in alert_tones:
                tone_id = tone_obj.get('tone_id', '')
                tone_a = float(tone_obj.get('tone_a', 0.0))
                tone_b = float(tone_obj.get('tone_b', 0.0))
                # Convert seconds to milliseconds
                tone_a_length = int(float(tone_obj.get('tone_a_length', 0.0)) * 1000)
                tone_b_length = int(float(tone_obj.get('tone_b_length', 0.0)) * 1000)
                tone_a_range = int(tone_obj.get('tone_a_range', 0))
                tone_b_range = int(tone_obj.get('tone_b_range', 0))
                # Convert seconds to milliseconds
                record_length = int(tone_obj.get('record_length', 0)) * 1000
                detection_tone_alert = tone_obj.get('detection_tone_alert', '')
                
                if tone_id and tone_a > 0 and tone_b > 0:
                    if tone_detect.add_tone_definition(
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
        except ImportError:
            print("[CONFIG] WARNING: tone_detect module not available yet")
    else:
        print(f"[CONFIG] WARNING: No alert_tones array found for channel {channel_index + 1}")
    
    tone_config.valid = True
    print(f"[CONFIG] Tone detection config for channel {channel_index + 1}: passthrough={tone_config.tone_passthrough}, target={tone_config.passthrough_channel}")


def load_channel_config(channel_ids: List[str]) -> int:
    """
    Load channel IDs from config.json (legacy function for compatibility).
    
    Args:
        channel_ids: List to populate with channel IDs
        
    Returns:
        Number of channel IDs loaded
    """
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
    """
    Load complete configuration including tone detection settings.
    
    Returns:
        True if configuration loaded successfully
    """
    return load_config()


# ============================================================================
# Configuration Access Functions
# ============================================================================

def get_channel_config(channel_index: int) -> Optional[ChannelConfig]:
    """
    Get channel configuration by index.
    
    Args:
        channel_index: Channel index (0-3)
        
    Returns:
        ChannelConfig if found, None otherwise
    """
    if 0 <= channel_index < MAX_CHANNELS and global_app_config['valid']:
        return global_app_config['channels'][channel_index]
    return None


def get_tone_detect_config(channel_index: int) -> Optional[ToneDetectConfig]:
    """
    Get tone detection configuration by channel index.
    
    Args:
        channel_index: Channel index (0-3)
        
    Returns:
        ToneDetectConfig if found, None otherwise
    """
    channel_config = get_channel_config(channel_index)
    if channel_config and channel_config.tone_detect and channel_config.tone_config.valid:
        return channel_config.tone_config
    return None


def validate_config() -> bool:
    """
    Validate loaded configuration.
    
    Returns:
        True if configuration is valid
    """
    if not global_app_config['valid']:
        return False
    
    # Check that at least one channel is configured
    for i in range(MAX_CHANNELS):
        channel_config = global_app_config['channels'][i]
        if channel_config.valid and channel_config.channel_id:
            return True
    
    return False
