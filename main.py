"""
EchoStream Main - Application entry point

This module initializes all components, starts worker threads,
and manages the application lifecycle.
"""
import signal
import sys
import time
import threading
from echostream import (
    global_interrupted, global_channel_ids, global_channel_count,
    MAX_CHANNELS
)
import config
import gpio
import audio_hardware
import channel_manager
import websocket_client
import udp_manager
import event_coordinator
import tone_detection
import audio_workers


# ============================================================================
# Signal Handlers
# ============================================================================

def handle_shutdown(sig, frame):
    """Handle shutdown signal (SIGINT, SIGTERM)."""
    if global_interrupted.is_set():
        print("\n[MAIN] Force exit...")
        import os
        os._exit(1)
    
    print("\n[MAIN] Shutdown signal received, cleaning up...")
    global_interrupted.set()
    cleanup_all()


# ============================================================================
# Initialization Functions
# ============================================================================

def initialize_system() -> bool:
    """
    Initialize all system components.
    
    Returns:
        True if initialization successful, False otherwise
    """
    print("[MAIN] Initializing EchoStream system...")
    
    # 1. Load configuration
    print("[MAIN] Loading configuration...")
    channel_ids = [""] * MAX_CHANNELS
    channel_count = config.load_channel_config(channel_ids)
    
    if channel_count > 0:
        for i in range(channel_count):
            if i < MAX_CHANNELS:
                global_channel_ids[i] = channel_ids[i]
        global_channel_count = channel_count
        print(f"[MAIN] Loaded {channel_count} channel(s) from config")
    else:
        print("[MAIN] WARNING: No channels loaded from config, using defaults")
        for i in range(min(4, MAX_CHANNELS)):
            global_channel_ids[i] = f"channel_{i + 1}"
        global_channel_count = min(4, MAX_CHANNELS)
    
    # 2. Initialize tone detection (must be done before loading complete config)
    print("[MAIN] Initializing tone detection system...")
    if not tone_detection.init_tone_detection():
        print("[MAIN] ERROR: Failed to initialize tone detection system", file=sys.stderr)
        return False
    
    # 3. Load complete configuration (includes tone definitions)
    print("[MAIN] Loading complete configuration...")
    if not config.load_complete_config():
        print("[MAIN] WARNING: Failed to load complete config - tone detection may not work")
        # Don't fail, continue with defaults
    
    # 4. Initialize GPIO
    print("[MAIN] Initializing GPIO...")
    if not gpio.init_gpio():
        print("[MAIN] ERROR: Failed to initialize GPIO", file=sys.stderr)
        return False
    
    # 5. Initialize audio hardware
    print("[MAIN] Initializing audio hardware...")
    if not audio_hardware.init_portaudio():
        print("[MAIN] ERROR: Failed to initialize PyAudio", file=sys.stderr)
        return False
    
    # Detect USB devices
    audio_hardware.detect_usb_devices()
    
    # 6. Create channels from configuration
    print(f"[MAIN] Setting up {global_channel_count} channel(s)...")
    for i in range(global_channel_count):
        channel_id = global_channel_ids[i]
        print(f"[MAIN] Setting up channel {i + 1} with ID: {channel_id}")
        
        # Get device index for this channel
        device_index = audio_hardware.get_device_for_channel(i)
        
        if device_index < 0:
            print(f"[MAIN] ERROR: No device available for channel {i + 1} ({channel_id})", file=sys.stderr)
            return False
        
        # Create channel
        if not channel_manager.create_channel(channel_id, device_index):
            print(f"[MAIN] ERROR: Failed to create channel {i + 1} ({channel_id})", file=sys.stderr)
            return False
    
    # 7. Initialize shared audio buffer
    print("[MAIN] Initializing shared audio buffer...")
    if not audio_workers.init_shared_audio_buffer():
        print("[MAIN] ERROR: Failed to initialize shared audio buffer", file=sys.stderr)
        return False
    
    # 8. Setup event coordinator
    print("[MAIN] Setting up event coordinator...")
    event_coordinator.setup_event_handlers()
    
    # 9. Setup WebSocket (will be connected in worker thread)
    print("[MAIN] Setting up WebSocket...")
    ws_url = "wss://audio.redenes.org/ws/"
    
    # Setup UDP config callback (will be called when WebSocket receives UDP config)
    websocket_client.set_udp_config_callback(
        lambda udp_config: event_coordinator.handle_udp_ready(udp_config)
    )
    
    # Note: WebSocket connection will be established in worker thread
    
    print("[MAIN] System initialization complete")
    return True


def start_all_workers() -> bool:
    """
    Start all worker threads.
    
    Returns:
        True if all workers started successfully
    """
    print("[MAIN] Starting worker threads...")
    
    # Start GPIO monitor thread
    gpio_thread = threading.Thread(
        target=lambda: gpio.monitor_gpio(
            callback=lambda gpio_num, state: _handle_gpio_callback(gpio_num, state)
        ),
        daemon=True
    )
    gpio_thread.start()
    print("[MAIN] GPIO monitor thread started")
    
    # Start tone detection worker thread
    tone_thread = threading.Thread(
        target=tone_detection_worker,
        daemon=True
    )
    tone_thread.start()
    print("[MAIN] Tone detection worker thread started")
    
    # Start WebSocket handler thread
    ws_thread = threading.Thread(
        target=lambda: websocket_client.global_websocket_thread("wss://audio.redenes.org/ws/"),
        daemon=True
    )
    ws_thread.start()
    print("[MAIN] WebSocket handler thread started")
    
    print("[MAIN] All worker threads started")
    return True


def tone_detection_worker():
    """Tone detection worker thread - processes shared audio buffer."""
    print("[MAIN] Tone detection worker thread started")
    
    process_count = 0
    
    while not global_interrupted.is_set():
        try:
            # Wait for data in shared buffer
            with audio_workers.global_shared_buffer.mutex:
                if not audio_workers.global_shared_buffer.valid:
                    audio_workers.global_shared_buffer.data_ready.wait(timeout=0.1)
                
                if audio_workers.global_shared_buffer.valid and audio_workers.global_shared_buffer.sample_count > 0:
                    # Read samples from shared buffer
                    samples, sample_count = audio_workers.global_shared_buffer.read()
                    
                    if samples is not None and sample_count > 0:
                        # Process audio for tone detection
                        tone_detection.process_audio_samples(samples, sample_count)
                        process_count += 1
            
            time.sleep(0.01)  # Small delay to avoid busy waiting
            
        except Exception as e:
            if not global_interrupted.is_set():
                print(f"[MAIN] ERROR: Tone detection worker error: {e}")
            time.sleep(0.1)
    
    print("[MAIN] Tone detection worker thread stopped")


def _handle_gpio_callback(gpio_num: int, state: int):
    """Handle GPIO state change callback."""
    # Find channel ID for this GPIO
    channel_id = _find_channel_for_gpio(gpio_num)
    
    if channel_id:
        # Route GPIO event through event coordinator
        event_coordinator.emit_event("gpio_change", {
            'channel_id': channel_id,
            'state': state
        })


def _find_channel_for_gpio(gpio_num: int):
    """Find channel ID for a GPIO number."""
    import gpio
    
    # Find which channel index this GPIO belongs to
    gpio_list = list(gpio.GPIO_PINS.keys())
    if gpio_num in gpio_list:
        channel_index = gpio_list.index(gpio_num)
        if channel_index < len(global_channel_ids) and channel_index < global_channel_count:
            return global_channel_ids[channel_index]
    
    return None


# ============================================================================
# Cleanup Functions
# ============================================================================

def cleanup_all():
    """Cleanup all resources and stop all threads."""
    print("[MAIN] Starting cleanup...")
    
    # Stop tone detection
    tone_detection.stop_tone_detection()
    
    # Stop all channels
    channel_manager.cleanup_all_channels()
    
    # Cleanup GPIO
    gpio.cleanup_gpio()
    
    # Cleanup audio hardware
    audio_hardware.cleanup_portaudio()
    
    # Disconnect WebSocket
    websocket_client.disconnect()
    
    # Cleanup UDP
    udp_manager.cleanup_udp()
    udp_manager.stop_heartbeat()
    
    # Cleanup event coordinator
    event_coordinator.cleanup_event_coordinator()
    
    print("[MAIN] Cleanup complete")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main application entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Initialize global interrupted flag
    global_interrupted.clear()
    
    # Initialize system
    if not initialize_system():
        print("[MAIN] ERROR: System initialization failed", file=sys.stderr)
        cleanup_all()
        return 1
    
    # Start all worker threads
    if not start_all_workers():
        print("[MAIN] ERROR: Failed to start worker threads", file=sys.stderr)
        cleanup_all()
        return 1
    
    # Start tone detection
    tone_detection.start_tone_detection()
    
    # Print system status
    # Wait a moment for WebSocket to connect and register channels
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("[MAIN] EchoStream system ready")
    
    # Get actual active channel count
    import channel_manager
    active_channels = channel_manager.get_all_channels()
    print(f"[MAIN] {len(active_channels)} channel(s) active")
    
    print("[MAIN] Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Print channel configuration
    print("=== SYSTEM BEHAVIOR ===")
    print("Channel Configuration:")
    for i in range(global_channel_count):
        channel_id = global_channel_ids[i]
        print(f"  Channel {i + 1} ({channel_id}):")
        print("    - Output: ALWAYS plays EchoStream audio")
        
        # Check if channel has tone detection enabled
        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            print("    - Input: ENABLED (for tone detection and passthrough)")
            
            # Show passthrough target
            tone_cfg = channel_config.tone_config
            if tone_cfg and tone_cfg.tone_passthrough:
                print(f"    - Passthrough target: {tone_cfg.passthrough_channel}")
        else:
            print("    - Input: ENABLED (no tone detection)")
    
    print("\nTone detection:")
    print(f"  Status: {'ENABLED' if tone_detection.is_tone_detect_enabled() else 'DISABLED'}")
    valid_tones = sum(1 for td in tone_detection.global_tone_detection.tone_definitions if td.valid)
    print(f"  Loaded tone definitions: {valid_tones}")
    print("=" * 60 + "\n")
    
    # Main loop - wait for shutdown
    try:
        while not global_interrupted.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        handle_shutdown(None, None)
    
    # Cleanup
    cleanup_all()
    
    print("[MAIN] Application terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
