"""
EchoStream Main - Main entry point for the EchoStream application
"""
import signal
import sys
import time
import threading
from echostream import global_interrupted, global_channel_ids, global_channel_count, MAX_CHANNELS, CHANNEL_ID_LEN
import config
import audio
import websocket
import gpio
import udp
import mqtt
import tone_detect
import s3_upload

def handle_interrupt(sig, frame):
    """Handle interrupt signal (Ctrl+C)"""
    print("\nShutdown signal received, cleaning up...")
    global_interrupted.set()
    
    # Cleanup audio devices
    audio.cleanup_audio_devices()
    
    # Close WebSocket
    if websocket.global_ws_client:
        websocket.global_ws_client = None
    
    # Stop audio streams
    for i in range(MAX_CHANNELS):
        if audio.channels[i].active:
            audio.channels[i].audio.transmitting = False
            
            if audio.channels[i].audio.input_stream:
                try:
                    audio.channels[i].audio.input_stream.stop_stream()
                    audio.channels[i].audio.input_stream.close()
                except Exception:
                    pass
                audio.channels[i].audio.input_stream = None
            
            if audio.channels[i].audio.output_stream:
                try:
                    audio.channels[i].audio.output_stream.stop_stream()
                    audio.channels[i].audio.output_stream.close()
                except Exception:
                    pass
                audio.channels[i].audio.output_stream = None
    
    # Close UDP socket
    if udp.global_udp_socket:
        try:
            udp.global_udp_socket.close()
        except Exception:
            pass
        udp.global_udp_socket = None
    
    # Stop audio passthrough
    audio.stop_audio_passthrough()
    
    # Stop tone detection
    tone_detect.stop_tone_detection()
    
    # Cleanup MQTT
    mqtt.cleanup_mqtt()
    
    # Terminate PyAudio
    if audio.pa_instance:
        try:
            audio.pa_instance.terminate()
        except Exception:
            pass

def main():
    """Main function"""
    # Initialize global variables
    global_interrupted.clear()
    
    # Load channel configuration from JSON file
    print("Loading channel configuration from /home/will/.an/config.json...")
    channel_ids = [""] * MAX_CHANNELS
    global_channel_count = config.load_channel_config(channel_ids)
    
    if global_channel_count > 0:
        for i in range(global_channel_count):
            global_channel_ids[i] = channel_ids[i]
        print(f"Successfully loaded {global_channel_count} channels from config")
    else:
        print("No channels loaded from config, using generic defaults")
        for i in range(4):
            global_channel_ids[i] = f"channel_{i + 1}"
        global_channel_count = 4
    
    # Initialize tone detection system FIRST (before loading config)
    # This creates the global_tone_detection object that config will populate
    if not tone_detect.init_tone_detection():
        print("Failed to initialize tone detection system", file=sys.stderr)
        return 1
    
    # Load complete configuration including tone detection settings
    # This will add tone definitions to the already-initialized global_tone_detection
    print("[MAIN] Loading complete configuration from /home/will/.an/config.json...")
    if config.load_complete_config():
        print("[MAIN] Complete configuration loaded successfully")
    else:
        print("[MAIN] ERROR: Failed to load JSON config - NO TONE DETECTION AVAILABLE")
        print("[MAIN] Please check /home/will/.an/config.json file exists and is readable")
        return 1
    
    if not audio.initialize_portaudio():
        print("PortAudio initialization failed", file=sys.stderr)
        return 1
    
    # Initialize audio devices
    if not audio.initialize_audio_devices():
        print("Audio device initialization failed", file=sys.stderr)
        return 1
    
    # Initialize tone detection control
    if not audio.init_tone_detect_control():
        print("Failed to initialize tone detection control", file=sys.stderr)
        return 1
    
    # Initialize shared audio buffer
    if not audio.init_shared_audio_buffer():
        print("Failed to initialize shared audio buffer", file=sys.stderr)
        return 1
    
    # Setup signal handler
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    
    # Start GPIO monitor thread
    gpio_thread = threading.Thread(target=gpio.gpio_monitor_worker, daemon=True)
    gpio_thread.start()
    
    print(f"Setting up {global_channel_count} channels...")
    
    # Setup channels
    for i in range(global_channel_count):
        print(f"Setting up channel {i + 1} with ID: {global_channel_ids[i]}")
        if not audio.setup_channel(audio.channels[i], global_channel_ids[i]):
            print(f"Failed to setup channel {i + 1} ({global_channel_ids[i]})", file=sys.stderr)
            return 1
    
    # Initialize audio passthrough
    if not audio.init_audio_passthrough():
        print("Failed to initialize audio passthrough", file=sys.stderr)
        return 1
    
    # Connect global WebSocket
    if not websocket.connect_global_websocket():
        print("Failed to connect WebSocket", file=sys.stderr)
        return 1
    
    # Start audio passthrough
    if not audio.start_audio_passthrough():
        print("Failed to start audio passthrough", file=sys.stderr)
        return 1
    
    # Start tone detection
    if not tone_detect.start_tone_detection():
        print("Failed to start tone detection", file=sys.stderr)
        return 1
    
    # Start tone detection worker thread
    def tone_detection_worker():
        """Worker thread that processes shared audio buffer for tone detection"""
        import audio
        import tone_detect
        from echostream import global_interrupted
        
        print("[INFO] Tone detection thread started")
        
        process_count = 0
        while not global_interrupted.is_set():
            try:
                # Wait for data in shared buffer
                with audio.global_shared_buffer.mutex:
                    if not audio.global_shared_buffer.valid:
                        audio.global_shared_buffer.data_ready.wait(timeout=0.1)
                    
                    if audio.global_shared_buffer.valid and audio.global_shared_buffer.sample_count > 0:
                        # Process audio for tone detection
                        samples = audio.global_shared_buffer.samples[:audio.global_shared_buffer.sample_count].copy()
                        sample_count = audio.global_shared_buffer.sample_count
                        
                        # Process audio
                        tone_detect.process_audio_python_approach(samples, sample_count)
                        process_count += 1
                    else:
                        # Debug: Log if buffer is empty
                        if process_count == 0:
                            static_empty_count = getattr(tone_detection_worker, '_empty_count', 0)
                            tone_detection_worker._empty_count = static_empty_count + 1
                            if static_empty_count % 1000 == 0:  # Log every 1000th empty check
                                print(f"[TONE DEBUG] Shared buffer empty (valid={audio.global_shared_buffer.valid}, count={audio.global_shared_buffer.sample_count})")
                
                time.sleep(0.01)  # Small delay to avoid busy waiting
            except Exception as e:
                if not global_interrupted.is_set():
                    print(f"[ERROR] Tone detection worker error: {e}")
                time.sleep(0.1)
        
        print("[INFO] Tone detection thread stopped")
    
    tone_thread = threading.Thread(target=tone_detection_worker, daemon=True)
    tone_thread.start()
    
    # Start WebSocket thread
    ws_thread = threading.Thread(target=websocket.global_websocket_thread, daemon=True)
    ws_thread.start()
    
    print(f"All {global_channel_count} channels running with single WebSocket. Press Ctrl+C to stop.")
    print("\n=== SYSTEM BEHAVIOR ===")
    print("Channel Configuration:")
    for i in range(global_channel_count):
        print(f"  Channel {i + 1} ({global_channel_ids[i]}):")
        print("    - Output: ALWAYS plays EchoStream audio (unaffected by tone detection)")
        
        # Check if this channel has tone detection enabled
        has_tone_detect = False
        for j in range(MAX_CHANNELS):
            channel_config = config.get_channel_config(j)
            if channel_config and channel_config.valid and channel_config.channel_id == global_channel_ids[i]:
                has_tone_detect = channel_config.tone_detect
                break
        
        if has_tone_detect:
            print(f"    - Input: {'ENABLED' if audio.is_card1_input_enabled() else 'DISABLED'} (for tone detection and passthrough)")
        else:
            print("    - Input: ENABLED (no tone detection)")
    
    # Reflect configured passthrough channel
    tone_cfg = config.get_tone_detect_config(0)
    if tone_cfg and tone_cfg.tone_passthrough:
        print(f"Passthrough output target: {tone_cfg.passthrough_channel}")
    
    print("\nTone detection control available:")
    print("  - Call enable_tone_detection() to enable tone detect mode")
    print("  - Call disable_tone_detection() to disable tone detect mode")
    print(f"  - Current mode: {'ENABLED' if audio.is_tone_detect_enabled() else 'DISABLED'}")
    
    # Wait for threads
    try:
        while not global_interrupted.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        handle_interrupt(None, None)
    
    # Cleanup
    tone_detect.stop_tone_detection()
    audio.stop_audio_passthrough()
    audio.cleanup_audio_devices()
    mqtt.cleanup_mqtt()
    
    if audio.pa_instance:
        audio.pa_instance.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

