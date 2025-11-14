"""
Configuration management - loads channel and tone detection settings from JSON
"""
import json
import os
from typing import Optional, List, Dict, Any
from echostream import MAX_CHANNELS, CHANNEL_ID_LEN

# Tone detection configuration structure
class ToneDetectConfig:
    def __init__(self):
        self.tone_passthrough = False
        self.passthrough_channel = ""
        self.threshold = 0.0
        self.gain = 0.0
        self.db_threshold = 0
        self.detect_new_tones = False
        self.new_tone_length_ms = 0
        self.new_tone_range_hz = 0
        self.valid = False

# Channel configuration structure
class ChannelConfig:
    def __init__(self):
        self.channel_id = ""
        self.input_low_one = False
        self.input_low_two = False
        self.input_high_one = False
        self.input_high_two = False
        self.tone_detect = False
        self.tone_config = ToneDetectConfig()
        self.valid = False

# Global configuration instance
global_app_config = {
    'channels': [ChannelConfig() for _ in range(MAX_CHANNELS)],
    'valid': False
}

def load_channel_config(channel_ids: List[str]) -> int:
    """Load channel IDs from config.json"""
    config_path = "/home/will/.an/config.json"
    
    try:
        with open(config_path, 'r') as file:
            json_data = json.load(file)
        
        # Navigate to the channel configuration
        shadow = json_data.get('shadow', {})
        state = shadow.get('state', {})
        desired = state.get('desired', {})
        software_config = desired.get('software_configuration', [])
        
        if not software_config or len(software_config) == 0:
            print("Error: Could not find software_configuration in config")
            return 0
        
        config_item = software_config[0]
        
        # Extract channel IDs
        channel_keys = ["channel_one", "channel_two", "channel_three", "channel_four"]
        channels_loaded = 0
        
        for i, key in enumerate(channel_keys):
            channel_obj = config_item.get(key, {})
            channel_id = channel_obj.get('channel_id', '')
            
            if channel_id and len(channel_id) < CHANNEL_ID_LEN:
                channel_ids[i] = channel_id
                channels_loaded += 1
                print(f"Loaded channel {i + 1} ID: {channel_ids[i]}")
        
        print(f"Successfully loaded {channels_loaded} channel IDs from config")
        return channels_loaded
        
    except FileNotFoundError:
        print(f"Warning: Could not open config file {config_path}, using default channel IDs")
        return 0
    except Exception as e:
        print(f"Error loading config: {e}")
        return 0

def load_complete_config() -> bool:
    """Load complete configuration including tone detection settings"""
    config_path = "/home/will/.an/config.json"
    print(f"[CONFIG] Attempting to load configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            json_data = json.load(file)
        
        print("[CONFIG] Successfully opened config file")
        print(f"[CONFIG] Config file size: {os.path.getsize(config_path)} bytes")
        
        # Navigate to the channel configuration
        shadow = json_data.get('shadow', {})
        state = shadow.get('state', {})
        desired = state.get('desired', {})
        software_config = desired.get('software_configuration', [])
        
        if not software_config or len(software_config) == 0:
            print("Error: Could not find software_configuration in config")
            return False
        
        config_item = software_config[0]
        
        # Extract channel configurations
        channel_keys = ["channel_one", "channel_two", "channel_three", "channel_four"]
        channels_loaded = 0
        
        for i, key in enumerate(channel_keys):
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
                tone_detect_config_obj = channel_obj.get('tone_detect_configuration', {})
                if tone_detect_config_obj:
                    tone_config = channel_config.tone_config
                    
                    # Load tone passthrough settings
                    tone_config.tone_passthrough = tone_detect_config_obj.get('tone_passthrough', False)
                    tone_config.passthrough_channel = tone_detect_config_obj.get('passthrough_channel', '')
                    
                    # Load alert details
                    alert_details = tone_detect_config_obj.get('alert_details', {})
                    if alert_details:
                        tone_config.threshold = float(alert_details.get('threshold', 0.0))
                        tone_config.gain = float(alert_details.get('gain', 0.0))
                        tone_config.db_threshold = int(alert_details.get('db', 0))
                        tone_config.detect_new_tones = alert_details.get('detect_new_tones', False)
                        tone_config.new_tone_length_ms = int(alert_details.get('new_tone_length', 0))
                        tone_config.new_tone_range_hz = int(alert_details.get('new_tone_range', 0))
                    
                    tone_config.valid = True
                    
                    print(f"Loaded tone detection config for channel {i+1}: passthrough={tone_config.tone_passthrough}, channel={tone_config.passthrough_channel}")
                    print(f"Applied tone config: threshold={tone_config.threshold}, gain={tone_config.gain}, db={tone_config.db_threshold}, detect_new={tone_config.detect_new_tones}")
            
            channel_config.valid = True
            channels_loaded += 1
            print(f"Loaded channel {i+1} config: ID={channel_config.channel_id}, tone_detect={channel_config.tone_detect}")
        
        if channels_loaded > 0:
            global_app_config['valid'] = True
            print(f"Successfully loaded configuration for {channels_loaded} channels")
            return True
        else:
            print("Warning: No channel configurations loaded")
            return False
            
    except FileNotFoundError:
        print(f"[ERROR] Could not open config file {config_path}")
        print("Using default configuration")
        return False
    except Exception as e:
        print(f"Error loading complete config: {e}")
        return False

def get_channel_config(channel_index: int) -> Optional[ChannelConfig]:
    """Get channel configuration by index"""
    if 0 <= channel_index < MAX_CHANNELS and global_app_config['valid']:
        return global_app_config['channels'][channel_index]
    return None

def get_tone_detect_config(channel_index: int) -> Optional[ToneDetectConfig]:
    """Get tone detection configuration by channel index"""
    channel_config = get_channel_config(channel_index)
    if channel_config and channel_config.tone_detect and channel_config.tone_config.valid:
        return channel_config.tone_config
    
    print(f"[CONFIG] No valid tone detection configuration found for channel {channel_index}")
    print("[CONFIG] Please ensure config.json contains proper tone_detect_configuration for this channel")
    return None

