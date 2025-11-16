import sys
from gpio_monitor import init_gpio, monitor_gpio, cleanup_gpio, GPIO_PINS
from config import load_config, get_channel_ids, get_tone_detect_config
from websocket_client import start_websocket


def main() -> int:
    print("[MAIN] Loading configuration...")
    cfg = load_config()
    ch_ids = get_channel_ids(cfg)
    tone_map = get_tone_detect_config(cfg)
    if ch_ids:
        print(f"[MAIN] Loaded {len(ch_ids)} channel ID(s) from config:")
        for idx, cid in enumerate(ch_ids, 1):
            print(f"  Channel {idx}: {cid}")
    else:
        print("[MAIN] No channel IDs found in config")
    if tone_map:
        print("[MAIN] Tone detection per channel:")
        for cid, td in tone_map:
            print(f"  {cid}: tone_detect={'ENABLED' if td else 'DISABLED'}")

    # Start WS without auto-register; GPIO activity will trigger channel registration
    start_websocket("wss://audio.redenes.org/ws/", ch_ids)

    print("[MAIN] GPIO monitor starting...")
    if not init_gpio(0):
        print("[MAIN] Failed to initialize GPIO (chip 0)")
        return 1
    try:
        # Build mapping from GPIO number to channel_id by index order
        gpio_keys = list(GPIO_PINS.keys())
        gpio_to_channel = {}
        for idx, gpio_num in enumerate(gpio_keys):
            if idx < len(ch_ids):
                gpio_to_channel[gpio_num] = ch_ids[idx]
        # Callback to (lazily) register/connect channel when its GPIO becomes ACTIVE (value 0)
        from websocket_client import send_transmit_event, register_channel, send_connect_message
        def _on_gpio_change(gpio_num: int, state: int):
            ch_id = gpio_to_channel.get(gpio_num)
            if not ch_id:
                return
            if state == 0:
                send_connect_message(ch_id)
                register_channel(ch_id)
                send_transmit_event(ch_id, True)
            elif state == 1:
                send_transmit_event(ch_id, False)
        # Proactively register channels for GPIOs that are already ACTIVE at startup
        from gpio_monitor import gpio_states
        for gpio_num, state in list(gpio_states.items()):
            if state == 0:
                _on_gpio_change(gpio_num, state)
        monitor_gpio(poll_interval=0.1, status_every=100, on_change=_on_gpio_change)
    finally:
        cleanup_gpio()
        print("[MAIN] GPIO monitor stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())


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

def handle_shutdown(sig, frame):
    if global_interrupted.is_set():
        print("\n[MAIN] Force exit...")
        import os
        os._exit(1)

    print("\n[MAIN] Shutdown signal received, cleaning up...")
    global_interrupted.set()
    cleanup_all()

def initialize_system() -> bool:

    print("[MAIN] Initializing EchoStream system...")

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

    print("[MAIN] Initializing tone detection system...")
    if not tone_detection.init_tone_detection():
        print("[MAIN] ERROR: Failed to initialize tone detection system", file=sys.stderr)
        return False

    print("[MAIN] Loading complete configuration...")
    if not config.load_complete_config():
        print("[MAIN] WARNING: Failed to load complete config - tone detection may not work")

    print("[MAIN] Initializing GPIO...")
    if not gpio.init_gpio():
        print("[MAIN] ERROR: Failed to initialize GPIO", file=sys.stderr)
        return False

    print("[MAIN] Initializing audio hardware...")
    if not audio_hardware.init_portaudio():
        print("[MAIN] ERROR: Failed to initialize PyAudio", file=sys.stderr)
        return False

    audio_hardware.detect_usb_devices()

    print(f"[MAIN] Setting up {global_channel_count} channel(s)...")
    for i in range(global_channel_count):
        channel_id = global_channel_ids[i]
        print(f"[MAIN] Setting up channel {i + 1} with ID: {channel_id}")

        device_index = audio_hardware.get_device_for_channel(i)

        if device_index < 0:
            print(f"[MAIN] ERROR: No device available for channel {i + 1} ({channel_id})", file=sys.stderr)
            return False

        if not channel_manager.create_channel(channel_id, device_index):
            print(f"[MAIN] ERROR: Failed to create channel {i + 1} ({channel_id})", file=sys.stderr)
            return False

    print("[MAIN] Initializing shared audio buffer...")
    if not audio_workers.init_shared_audio_buffer():
        print("[MAIN] ERROR: Failed to initialize shared audio buffer", file=sys.stderr)
        return False

    print("[MAIN] Setting up event coordinator...")
    event_coordinator.setup_event_handlers()

    print("[MAIN] Setting up WebSocket...")
    ws_url = "wss://audio.redenes.org/ws/"

    websocket_client.set_udp_config_callback(
        lambda udp_config: event_coordinator.handle_udp_ready(udp_config)
    )

    print("[MAIN] System initialization complete")
    return True

def start_all_workers() -> bool:

    print("[MAIN] Starting worker threads...")

    gpio_thread = threading.Thread(
        target=lambda: gpio.monitor_gpio(
            callback=lambda gpio_num, state: _handle_gpio_callback(gpio_num, state)
        ),
        daemon=True
    )
    gpio_thread.start()
    print("[MAIN] GPIO monitor thread started")

    tone_thread = threading.Thread(
        target=tone_detection_worker,
        daemon=True
    )
    tone_thread.start()
    print("[MAIN] Tone detection worker thread started")

    ws_thread = threading.Thread(
        target=lambda: websocket_client.global_websocket_thread("wss://audio.redenes.org/ws/"),
        daemon=True
    )
    ws_thread.start()
    print("[MAIN] WebSocket handler thread started")

    print("[MAIN] All worker threads started")
    return True

def tone_detection_worker():
    print("[MAIN] Tone detection worker thread started")

    process_count = 0

    while not global_interrupted.is_set():
        try:

            with audio_workers.global_shared_buffer.mutex:
                if not audio_workers.global_shared_buffer.valid:
                    audio_workers.global_shared_buffer.data_ready.wait(timeout=0.1)

                if audio_workers.global_shared_buffer.valid and audio_workers.global_shared_buffer.sample_count > 0:

                    samples, sample_count = audio_workers.global_shared_buffer.read()

                    if samples is not None and sample_count > 0:

                        tone_detection.process_audio_samples(samples, sample_count)
                        process_count += 1

            time.sleep(0.01)

        except Exception as e:
            if not global_interrupted.is_set():
                print(f"[MAIN] ERROR: Tone detection worker error: {e}")
            time.sleep(0.1)

    print("[MAIN] Tone detection worker thread stopped")

def _handle_gpio_callback(gpio_num: int, state: int):

    channel_id = _find_channel_for_gpio(gpio_num)

    if channel_id:

        event_coordinator.emit_event("gpio_change", {
            'channel_id': channel_id,
            'state': state
        })

def _find_channel_for_gpio(gpio_num: int):
    import gpio

    gpio_list = list(gpio.GPIO_PINS.keys())
    if gpio_num in gpio_list:
        channel_index = gpio_list.index(gpio_num)
        if channel_index < len(global_channel_ids) and channel_index < global_channel_count:
            return global_channel_ids[channel_index]

    return None

def cleanup_all():
    print("[MAIN] Starting cleanup...")

    tone_detection.stop_tone_detection()

    channel_manager.cleanup_all_channels()

    gpio.cleanup_gpio()

    audio_hardware.cleanup_portaudio()

    websocket_client.disconnect()

    udp_manager.cleanup_udp()
    udp_manager.stop_heartbeat()

    event_coordinator.cleanup_event_coordinator()

    print("[MAIN] Cleanup complete")

def main():

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    global_interrupted.clear()

    if not initialize_system():
        print("[MAIN] ERROR: System initialization failed", file=sys.stderr)
        cleanup_all()
        return 1

    if not start_all_workers():
        print("[MAIN] ERROR: Failed to start worker threads", file=sys.stderr)
        cleanup_all()
        return 1

    tone_detection.start_tone_detection()

    time.sleep(2)

    print("\n" + "=" * 60)
    print("[MAIN] EchoStream system ready")

    import channel_manager
    active_channels = channel_manager.get_all_channels()
    print(f"[MAIN] {len(active_channels)} channel(s) active")

    print("[MAIN] Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    print("=== SYSTEM BEHAVIOR ===")
    print("Channel Configuration:")
    for i in range(global_channel_count):
        channel_id = global_channel_ids[i]
        print(f"  Channel {i + 1} ({channel_id}):")
        print("    - Output: ALWAYS plays EchoStream audio")

        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            print("    - Input: ENABLED (for tone detection and passthrough)")

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

    try:
        while not global_interrupted.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        handle_shutdown(None, None)

    cleanup_all()

    print("[MAIN] Application terminated")
    return 0

if __name__ == "__main__":
    sys.exit(main())
