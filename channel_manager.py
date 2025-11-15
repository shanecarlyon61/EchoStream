"""
Channel Manager - Manage multiple audio channels

This module handles channel lifecycle management, GPIO to channel mapping,
and transmission control for multiple audio channels.
"""
import threading
from typing import Optional, Dict, List
from echostream import MAX_CHANNELS, global_channel_ids, global_channel_count
# Import at runtime to avoid circular dependencies
# from audio_stream import AudioStream, create_stream, start_stream, stop_stream, cleanup_stream, set_encryption_key
# from audio_hardware import get_device_for_channel, get_device_info, pa_instance
# from audio_workers import audio_input_worker, audio_output_worker
# from gpio import get_gpio_for_channel
# from websocket_client import register_channel, send_transmit_event, send_connect_message
# from udp_manager import send_audio_packet, is_udp_ready
import threading


# ============================================================================
# Channel Registry
# ============================================================================

class ChannelContext:
    """Channel context containing audio stream and state."""
    def __init__(self):
        self.audio: Optional[AudioStream] = None
        self.active: bool = False
        self.channel_id: str = ""
        self.device_index: int = -1
        self.gpio_num: Optional[int] = None
        self.thread_input: Optional[threading.Thread] = None
        self.thread_output: Optional[threading.Thread] = None


# Channel registry (channel_id → ChannelContext)
channels: Dict[str, ChannelContext] = {}

# Channel registry mutex
channels_mutex = threading.Lock()


# ============================================================================
# Channel Creation and Management
# ============================================================================

def create_channel(channel_id: str, device_index: int) -> bool:
    """
    Create a new channel.
    
    Args:
        channel_id: Channel ID string
        device_index: Audio device index for this channel
        
    Returns:
        True if channel created successfully, False otherwise
    """
    # Import at runtime to avoid circular dependencies
    import audio_stream
    import audio_hardware
    import gpio
    
    global channels, channels_mutex
    
    if audio_hardware.pa_instance is None:
        print(f"[CHANNEL_MGR] ERROR: PyAudio not initialized")
        return False
    
    with channels_mutex:
        # Check if channel already exists
        if channel_id in channels:
            print(f"[CHANNEL_MGR] Channel {channel_id} already exists")
            return True
        
        # Create new channel context
        ctx = ChannelContext()
        ctx.channel_id = channel_id
        ctx.device_index = device_index
        
        # Create audio stream
        stream = audio_stream.create_stream(channel_id, device_index, audio_hardware.pa_instance)
        if stream is None:
            print(f"[CHANNEL_MGR] ERROR: Failed to create audio stream for {channel_id}")
            return False
        
        ctx.audio = stream
        
        # Get GPIO number for this channel (if available)
        # Find channel index in global_channel_ids
        channel_index = -1
        for i in range(MAX_CHANNELS):
            if i < len(global_channel_ids) and global_channel_ids[i] == channel_id:
                channel_index = i
                break
        
        if channel_index >= 0:
            ctx.gpio_num = gpio.get_gpio_for_channel(channel_index)
        
        # Add to registry
        channels[channel_id] = ctx
        ctx.active = True
        
        print(f"[CHANNEL_MGR] Channel {channel_id} created (device {device_index})")
        return True


def start_channel(channel_id: str) -> bool:
    """
    Start a channel (start audio streams and workers).
    
    Args:
        channel_id: Channel ID string
        
    Returns:
        True if channel started successfully, False otherwise
    """
    # Import at runtime to avoid circular dependencies
    import audio_stream
    import audio_hardware
    import audio_workers
    
    global channels, channels_mutex
    
    if audio_hardware.pa_instance is None:
        print(f"[CHANNEL_MGR] ERROR: PyAudio not initialized")
        return False
    
    with channels_mutex:
        if channel_id not in channels:
            print(f"[CHANNEL_MGR] ERROR: Channel {channel_id} not found")
            return False
        
        ctx = channels[channel_id]
        if ctx.audio is None:
            print(f"[CHANNEL_MGR] ERROR: Audio stream not created for {channel_id}")
            return False
        
        # Start audio streams
        if not audio_stream.start_stream(ctx.audio, audio_hardware.pa_instance):
            print(f"[CHANNEL_MGR] ERROR: Failed to start streams for {channel_id}")
            return False
        
        # Start input worker thread
        if ctx.thread_input is None or not ctx.thread_input.is_alive():
            ctx.thread_input = threading.Thread(
                target=audio_workers.audio_input_worker,
                args=(ctx.audio,),
                daemon=True
            )
            ctx.thread_input.start()
            print(f"[CHANNEL_MGR] Input worker started for channel {channel_id}")
        
        # Start output worker thread (if output stream exists)
        if ctx.audio.output_stream and (ctx.thread_output is None or not ctx.thread_output.is_alive()):
            ctx.thread_output = threading.Thread(
                target=audio_workers.audio_output_worker,
                args=(ctx.audio,),
                daemon=True
            )
            ctx.thread_output.start()
            print(f"[CHANNEL_MGR] Output worker started for channel {channel_id}")
        
        print(f"[CHANNEL_MGR] Channel {channel_id} started")
        return True


def stop_channel(channel_id: str):
    """
    Stop a channel (stop audio streams and workers).
    
    Args:
        channel_id: Channel ID string
    """
    # Import at runtime to avoid circular dependencies
    import audio_stream
    
    global channels, channels_mutex
    
    with channels_mutex:
        if channel_id not in channels:
            return
        
        ctx = channels[channel_id]
        
        # Stop audio streams
        if ctx.audio:
            audio_stream.stop_stream(ctx.audio)
        
        # Threads will stop when stream.transmitting is False
        # Wait a bit for threads to finish
        if ctx.thread_input and ctx.thread_input.is_alive():
            ctx.thread_input.join(timeout=1.0)
        
        if ctx.thread_output and ctx.thread_output.is_alive():
            ctx.thread_output.join(timeout=1.0)
        
        print(f"[CHANNEL_MGR] Channel {channel_id} stopped")


def handle_gpio_change(channel_id: str, gpio_active: bool) -> bool:
    """
    Handle GPIO state change for a channel (PTT pressed/released).
    
    Args:
        channel_id: Channel ID string
        gpio_active: True if GPIO is active (PTT pressed), False otherwise
        
    Returns:
        True if handled successfully, False otherwise
    """
    # Import at runtime to avoid circular dependencies
    import udp_manager
    import websocket_client
    
    global channels, channels_mutex
    
    with channels_mutex:
        if channel_id not in channels:
            return False
        
        ctx = channels[channel_id]
        if ctx.audio is None:
            return False
        
        # Update GPIO state
        was_active = ctx.audio.gpio_active
        ctx.audio.gpio_active = gpio_active
        
        # If GPIO became active and UDP is ready, start transmission if not already started
        if gpio_active and not was_active:
            # Check if UDP is ready
            if udp_manager.is_udp_ready():
                if not ctx.audio.transmitting:
                    # Start transmission
                    if start_channel(channel_id):
                        print(f"[CHANNEL_MGR] Started transmission for channel {channel_id} (GPIO active, UDP ready)")
                    
                    # Send transmit_started to server
                    websocket_client.send_transmit_event(channel_id, True)
        
        # If GPIO became inactive, send transmit_stopped
        if not gpio_active and was_active:
            websocket_client.send_transmit_event(channel_id, False)
        
        return True


def set_channel_encryption_key(channel_id: str, key: bytes) -> bool:
    """
    Set encryption key for a channel.
    
    Args:
        channel_id: Channel ID string
        key: Encryption key bytes (32 bytes)
        
    Returns:
        True if key set successfully, False otherwise
    """
    # Import at runtime to avoid circular dependencies
    import audio_stream
    
    global channels, channels_mutex
    
    with channels_mutex:
        if channel_id not in channels:
            return False
        
        ctx = channels[channel_id]
        if ctx.audio is None:
            return False
        
        audio_stream.set_encryption_key(ctx.audio, key)
        return True


def get_channel(channel_id: str) -> Optional[ChannelContext]:
    """
    Get channel context by channel ID.
    
    Args:
        channel_id: Channel ID string
        
    Returns:
        ChannelContext if found, None otherwise
    """
    global channels, channels_mutex
    
    with channels_mutex:
        return channels.get(channel_id)


def get_all_channels() -> List[str]:
    """
    Get list of all channel IDs.
    
    Returns:
        List of channel ID strings
    """
    global channels, channels_mutex
    
    with channels_mutex:
        return list(channels.keys())


def cleanup_all_channels():
    """Cleanup and stop all channels."""
    # Import at runtime to avoid circular dependencies
    import audio_stream
    
    global channels, channels_mutex
    
    with channels_mutex:
        channel_ids = list(channels.keys())
        
        for channel_id in channel_ids:
            ctx = channels[channel_id]
            
            # Stop streams
            if ctx.audio:
                audio_stream.stop_stream(ctx.audio)
                audio_stream.cleanup_stream(ctx.audio)
            
            # Threads will stop when stream.transmitting is False
        
        # Clear registry
        channels.clear()
        
        print("[CHANNEL_MGR] All channels cleaned up")


# ============================================================================
# Channel Registration with WebSocket
# ============================================================================

def register_channels_with_websocket():
    """Register all channels with WebSocket server."""
    # Import at runtime to avoid circular dependencies
    import websocket_client
    
    global channels, channels_mutex
    
    with channels_mutex:
        for channel_id, ctx in channels.items():
            if ctx.active:
                # Register channel
                websocket_client.register_channel(channel_id)
                
                # Send connect message
                websocket_client.send_connect_message(channel_id)
                
                # If GPIO is active, send transmit_started
                if ctx.audio and ctx.audio.gpio_active:
                    websocket_client.send_transmit_event(channel_id, True)

