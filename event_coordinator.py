
import threading
from typing import Dict, List, Callable, Any, Optional
from echostream import global_interrupted

event_handlers: Dict[str, List[Callable[[Any], None]]] = {}

coordinator_mutex = threading.Lock()

def register_handler(event_type: str, handler: Callable[[Any], None]):

    with coordinator_mutex:
        if event_type not in event_handlers:
            event_handlers[event_type] = []

        event_handlers[event_type].append(handler)
        print(f"[EVENT] Registered handler for event type: {event_type}")

def unregister_handler(event_type: str, handler: Callable[[Any], None]):

    with coordinator_mutex:
        if event_type in event_handlers:
            if handler in event_handlers[event_type]:
                event_handlers[event_type].remove(handler)
                print(f"[EVENT] Unregistered handler for event type: {event_type}")

def emit_event(event_type: str, event_data: Any = None):

    with coordinator_mutex:
        if event_type not in event_handlers:
            return

        handlers = event_handlers[event_type].copy()

    for handler in handlers:
        try:
            handler(event_data)
        except Exception as e:
            print(f"[EVENT] ERROR: Exception in handler for {event_type}: {e}")

def handle_gpio_event(data: Dict[str, Any]):

    channel_id = data.get('channel_id', '')
    state = data.get('state', 1)
    gpio_active = (state == 0)

    print(f"[EVENT] GPIO event for channel {channel_id}: {'ACTIVE' if gpio_active else 'INACTIVE'}")

    try:
        import channel_manager
        channel_manager.handle_gpio_change(channel_id, gpio_active)
    except ImportError as e:
        print(f"[EVENT] WARNING: channel_manager module not available: {e}")
    except Exception as e:
        print(f"[EVENT] ERROR: Exception handling GPIO event: {e}")

def handle_tone_detected(tone_def):

    print(f"[EVENT] Tone detected: {tone_def.tone_id}")

    try:
        import tone_passthrough
        tone_passthrough.enable_passthrough(tone_def)
    except ImportError:
        print("[EVENT] WARNING: tone_passthrough module not available")
    except Exception as e:
        print(f"[EVENT] ERROR: Exception handling tone detected event: {e}")

def handle_udp_ready(udp_config: Dict[str, Any]):

    print("[EVENT] UDP ready event")

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

                udp_manager.start_udp_listener()

                udp_manager.start_heartbeat()

                if aes_key_str and aes_key_str != 'N/A':
                    key_bytes = crypto.decode_base64(aes_key_str)
                    if key_bytes and len(key_bytes) == 32:
                        all_channels = channel_manager.get_all_channels()
                        for channel_id in all_channels:
                            channel_manager.set_channel_encryption_key(channel_id, key_bytes)

                def handle_udp_packet(channel_id: str, encrypted_data: bytes):
                    try:
                        import audio_workers
                        import audio_stream

                        ctx = channel_manager.get_channel(channel_id)
                        if not ctx or not ctx.audio:

                            static_channel_not_found = getattr(handle_udp_packet, '_channel_not_found', set())
                            if channel_id not in static_channel_not_found:
                                static_channel_not_found.add(channel_id)
                                handle_udp_packet._channel_not_found = static_channel_not_found
                                print(f"[EVENT] WARNING: No active channel found for '{channel_id}'")
                                all_channels = channel_manager.get_all_channels()
                                if all_channels:
                                    print(f"[EVENT] Active channels: {', '.join(all_channels)}")
                            return

                        key_is_zero = all(b == 0 for b in ctx.audio.encryption_key)
                        if key_is_zero:
                            static_zero_key_warned = getattr(handle_udp_packet, '_zero_key_warned', set())
                            if channel_id not in static_zero_key_warned:
                                static_zero_key_warned.add(channel_id)
                                handle_udp_packet._zero_key_warned = static_zero_key_warned
                                print(f"[EVENT] WARNING: AES key not set for channel {channel_id}; dropping encrypted audio until key is provisioned")
                            return

                        decrypted = crypto.decrypt_aes(encrypted_data, ctx.audio.encryption_key)
                        if not decrypted:
                            static_decrypt_failures = getattr(handle_udp_packet, '_decrypt_failures', {})
                            count = static_decrypt_failures.get(channel_id, 0) + 1
                            static_decrypt_failures[channel_id] = count
                            handle_udp_packet._decrypt_failures = static_decrypt_failures

                            if count <= 3 or count % 50 == 0:
                                print(f"[EVENT] WARNING: Decryption failed for channel {channel_id} (failure #{count})")
                            return

                        import audio_stream
                        import numpy as np
                        from echostream import SAMPLES_PER_FRAME

                        if ctx.audio.decoder is None:
                            return

                        try:

                            decoded_bytes = ctx.audio.decoder.decode(decrypted, SAMPLES_PER_FRAME)

                            pcm_data = np.frombuffer(decoded_bytes, dtype=np.int16)
                            samples_count = len(pcm_data)

                            if samples_count > 0:

                                float_samples = (pcm_data.astype(np.float32) / 32767.0) * 20.0

                                float_samples = np.clip(float_samples, -1.0, 1.0)

                                if len(float_samples) < SAMPLES_PER_FRAME:
                                    padded = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                                    padded[:len(float_samples)] = float_samples
                                    float_samples = padded

                                success = audio_workers.write_to_jitter_buffer(ctx.audio, float_samples, samples_count)
                            else:
                                success = False
                        except Exception as e:
                            static_decode_failures = getattr(handle_udp_packet, '_decode_failures', {})
                            count = static_decode_failures.get(channel_id, 0) + 1
                            static_decode_failures[channel_id] = count
                            handle_udp_packet._decode_failures = static_decode_failures

                            if count <= 3 or count % 50 == 0:
                                print(f"[EVENT] WARNING: Opus decode failed for channel {channel_id} (failure #{count}): {e}")
                            success = False

                        if not success:

                            static_buffer_full = getattr(handle_udp_packet, '_buffer_full', {})
                            count = static_buffer_full.get(channel_id, 0) + 1
                            static_buffer_full[channel_id] = count
                            handle_udp_packet._buffer_full = static_buffer_full

                            if count <= 3 or count % 100 == 0:
                                print(f"[EVENT] WARNING: Jitter buffer full for channel {channel_id} (drops: {count})")

                    except Exception as e:
                        print(f"[EVENT] ERROR: Exception handling UDP packet for channel {channel_id}: {e}")
                        import traceback
                        traceback.print_exc()

                udp_manager.set_packet_receive_callback(handle_udp_packet)

                print("[EVENT] Starting output workers for all channels...")
                all_channels = channel_manager.get_all_channels()

                for channel_id in all_channels:
                    ctx = channel_manager.get_channel(channel_id)
                    if ctx and ctx.audio:

                        if not ctx.audio.transmitting:

                            channel_manager.start_channel(channel_id)
                            print(f"[EVENT] Started output for channel {channel_id}")
                        else:

                            print(f"[EVENT] Output already running for channel {channel_id}")

                for channel_id in all_channels:
                    ctx = channel_manager.get_channel(channel_id)
                    if ctx and ctx.audio and ctx.audio.gpio_active:

                        websocket_client.send_transmit_event(channel_id, True)
                        print(f"[EVENT] Sent transmit_started for channel {channel_id} (GPIO active)")
            else:
                print("[EVENT] ERROR: Failed to setup UDP")
    except ImportError as e:
        print(f"[EVENT] WARNING: Module not available: {e}")
    except Exception as e:
        print(f"[EVENT] ERROR: Exception handling UDP ready event: {e}")
        import traceback
        traceback.print_exc()

def setup_event_handlers():

    register_handler("gpio_change", lambda data: handle_gpio_event(data['channel_id'], data['state']))
    register_handler("tone_detected", handle_tone_detected)
    register_handler("udp_ready", handle_udp_ready)

    print("[EVENT] Default event handlers registered")

def cleanup_event_coordinator():
    with coordinator_mutex:
        event_handlers.clear()
    print("[EVENT] Event coordinator cleaned up")

