"""
Event Coordinator - Coordinate events between modules

This module provides event routing and coordination between different modules,
allowing loose coupling and event-driven architecture.
"""
import threading
from typing import Dict, List, Callable, Any, Optional
from echostream import global_interrupted


# ============================================================================
# Event Handler Registry
# ============================================================================

# Event handlers (event_type → list of handler functions)
event_handlers: Dict[str, List[Callable[[Any], None]]] = {}

# Event coordinator mutex
coordinator_mutex = threading.Lock()


# ============================================================================
# Event Registration and Emission
# ============================================================================

def register_handler(event_type: str, handler: Callable[[Any], None]):
    """
    Register an event handler for a specific event type.
    
    Args:
        event_type: Event type string (e.g., "gpio_change", "tone_detected", "udp_ready")
        handler: Handler function(event_data) -> None
    """
    with coordinator_mutex:
        if event_type not in event_handlers:
            event_handlers[event_type] = []
        
        event_handlers[event_type].append(handler)
        print(f"[EVENT] Registered handler for event type: {event_type}")


def unregister_handler(event_type: str, handler: Callable[[Any], None]):
    """
    Unregister an event handler.
    
    Args:
        event_type: Event type string
        handler: Handler function to remove
    """
    with coordinator_mutex:
        if event_type in event_handlers:
            if handler in event_handlers[event_type]:
                event_handlers[event_type].remove(handler)
                print(f"[EVENT] Unregistered handler for event type: {event_type}")


def emit_event(event_type: str, event_data: Any = None):
    """
    Emit an event to all registered handlers.
    
    Args:
        event_type: Event type string
        event_data: Event data (any type) to pass to handlers
    """
    with coordinator_mutex:
        if event_type not in event_handlers:
            return
        
        handlers = event_handlers[event_type].copy()  # Copy to avoid modification during iteration
    
    # Call handlers outside of mutex to avoid deadlocks
    for handler in handlers:
        try:
            handler(event_data)
        except Exception as e:
            print(f"[EVENT] ERROR: Exception in handler for {event_type}: {e}")


# ============================================================================
# Event Handlers for Specific Events
# ============================================================================

def handle_gpio_event(data: Dict[str, Any]):
    """
    Handle GPIO state change event.
    
    Args:
        data: Event data dictionary with 'channel_id' and 'state'
    """
    channel_id = data.get('channel_id', '')
    state = data.get('state', 1)
    gpio_active = (state == 0)
    
    print(f"[EVENT] GPIO event for channel {channel_id}: {'ACTIVE' if gpio_active else 'INACTIVE'}")
    
    # Route to channel manager
    try:
        import channel_manager
        channel_manager.handle_gpio_change(channel_id, gpio_active)
    except ImportError as e:
        print(f"[EVENT] WARNING: channel_manager module not available: {e}")
    except Exception as e:
        print(f"[EVENT] ERROR: Exception handling GPIO event: {e}")


def handle_tone_detected(tone_def):
    """
    Handle tone detection event.
    
    Args:
        tone_def: Tone definition that was detected
    """
    print(f"[EVENT] Tone detected: {tone_def.tone_id}")
    
    # Route to tone passthrough
    try:
        import tone_passthrough
        tone_passthrough.enable_passthrough(tone_def)
    except ImportError:
        print("[EVENT] WARNING: tone_passthrough module not available")
    except Exception as e:
        print(f"[EVENT] ERROR: Exception handling tone detected event: {e}")


def handle_udp_ready(udp_config: Dict[str, Any]):
    """
    Handle UDP ready event (when UDP connection info is received).
    
    Args:
        udp_config: UDP configuration dictionary
    """
    print("[EVENT] UDP ready event")
    
    # Setup UDP
    try:
        import udp_manager
        import channel_manager
        import websocket_client
        import crypto
        
        udp_host = udp_config.get('udp_host', '')
        udp_port = udp_config.get('udp_port', 0)
        aes_key_str = udp_config.get('aes_key', 'N/A')
        
        if udp_host and udp_port > 0:
            if udp_manager.setup_udp(udp_host, udp_port):
                print(f"[EVENT] UDP configured: {udp_host}:{udp_port}")
                
                # Start UDP listener
                udp_manager.start_udp_listener()
                
                # Start heartbeat
                udp_manager.start_heartbeat()
                
                # Decode and set encryption keys for all channels
                if aes_key_str and aes_key_str != 'N/A':
                    key_bytes = crypto.decode_base64(aes_key_str)
                    if key_bytes and len(key_bytes) == 32:
                        all_channels = channel_manager.get_all_channels()
                        for channel_id in all_channels:
                            channel_manager.set_channel_encryption_key(channel_id, key_bytes)
                
                # Setup UDP packet receive callback
                def handle_udp_packet(channel_id: str, encrypted_data: bytes):
                    """Handle received UDP packet."""
                    try:
                        import audio_workers
                        ctx = channel_manager.get_channel(channel_id)
                        if ctx and ctx.audio:
                            # Decrypt packet
                            decrypted = crypto.decrypt_aes(encrypted_data, ctx.audio.encryption_key)
                            if decrypted:
                                # Write Opus data to jitter buffer
                                audio_workers.write_to_jitter_buffer(ctx.audio, decrypted)
                    except Exception as e:
                        print(f"[EVENT] ERROR: Exception handling UDP packet: {e}")
                
                udp_manager.set_packet_receive_callback(handle_udp_packet)
                
                # Start transmission for channels with active GPIO
                all_channels = channel_manager.get_all_channels()
                
                for channel_id in all_channels:
                    ctx = channel_manager.get_channel(channel_id)
                    if ctx and ctx.audio and ctx.audio.gpio_active:
                        # Start transmission if not already started
                        if not ctx.audio.transmitting:
                            channel_manager.start_channel(channel_id)
                        
                        # Send transmit_started to server
                        websocket_client.send_transmit_event(channel_id, True)
            else:
                print("[EVENT] ERROR: Failed to setup UDP")
    except ImportError as e:
        print(f"[EVENT] WARNING: Module not available: {e}")
    except Exception as e:
        print(f"[EVENT] ERROR: Exception handling UDP ready event: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Event Registration Setup
# ============================================================================

def setup_event_handlers():
    """Setup default event handlers."""
    # Register default handlers
    register_handler("gpio_change", lambda data: handle_gpio_event(data['channel_id'], data['state']))
    register_handler("tone_detected", handle_tone_detected)
    register_handler("udp_ready", handle_udp_ready)
    
    print("[EVENT] Default event handlers registered")


def cleanup_event_coordinator():
    """Cleanup event coordinator."""
    with coordinator_mutex:
        event_handlers.clear()
    print("[EVENT] Event coordinator cleaned up")

