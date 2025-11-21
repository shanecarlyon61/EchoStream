import sys
from gpio_monitor import init_gpio, monitor_gpio, cleanup_gpio, GPIO_PINS
from config import load_config, get_channel_ids, get_tone_detect_config
from websocket_client import start_websocket
from websocket_client import set_udp_config_callback
from websocket_client import set_output_device_map
from audio_devices import list_output_devices, select_output_device_for_channel


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

    # Enumerate audio output devices and align to channels
    try:
        devices = list_output_devices()
        if devices:
            print("[MAIN] Detected audio output devices:")
            for d in devices:
                print(
                    f"  - index={d.get('index')} name={d.get('name')} host_api={d.get('host_api')}"
                )
        else:
            print("[MAIN] WARNING: No audio output devices detected")
        ch_idx_to_device = {}
        for ch_index in range(len(ch_ids)):
            dev_index = select_output_device_for_channel(ch_index)
            if dev_index is not None:
                ch_idx_to_device[ch_index] = int(dev_index)
                print(
                    f"[MAIN] Channel {ch_index + 1} ({ch_ids[ch_index]}) -> audio_device_index {dev_index}"
                )
            else:
                print(
                    f"[MAIN] WARNING: No audio device selected for channel {ch_index + 1} ({ch_ids[ch_index]})"
                )
        if ch_idx_to_device:
            set_output_device_map(ch_idx_to_device)
    except Exception as e:
        print(f"[MAIN] WARNING: Audio device alignment failed: {e}")

    # Start WS without auto-register; GPIO activity will trigger channel registration
    start_websocket("wss://audio.redenes.org/ws/", ch_ids)

    # When UDP connection info arrives from WS, just log; the WS thread now starts its own UDP listener
    def _on_udp_ready(cfg: dict) -> None:
        try:
            udp_port = int(cfg.get("udp_port", 0) or 0)
            udp_host = str(cfg.get("udp_host", ""))
            if udp_port <= 0:
                print(f"[MAIN] WARNING: Invalid UDP port in config: {cfg}")
                return
            print(f"[MAIN] UDP config received: host={udp_host}, port={udp_port}")
        except Exception as e:
            print(f"[MAIN] ERROR: UDP config handling failed: {e}")

    set_udp_config_callback(_on_udp_ready)

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
        from websocket_client import send_transmit_event, request_register_channel

        def _on_gpio_change(gpio_num: int, state: int):
            ch_id = gpio_to_channel.get(gpio_num)
            if not ch_id:
                return
            if state == 0:
                request_register_channel(ch_id)
                send_transmit_event(ch_id, True)
            elif state == 1:
                send_transmit_event(ch_id, False)

        # Proactively check all configured channels against current GPIO states and connect ACTIVE ones
        from gpio_monitor import gpio_states

        print("[MAIN] Evaluating GPIO states for initial channel connections...")
        for idx, gpio_num in enumerate(gpio_keys):
            ch_id = gpio_to_channel.get(gpio_num)
            if not ch_id:
                continue
            state = gpio_states.get(gpio_num, -1)
            status = (
                "ACTIVE" if state == 0 else ("INACTIVE" if state == 1 else "UNKNOWN")
            )
            print(
                f"[MAIN] Channel {idx + 1} ({ch_id}) mapped to GPIO {gpio_num}: {status}"
            )
            if state == 0:
                print(f"[MAIN] Connecting ACTIVE channel {ch_id} via WebSocket")
                request_register_channel(ch_id)
                send_transmit_event(ch_id, True)
        print("[MAIN] Initial channel connection evaluation complete")
        monitor_gpio(poll_interval=0.1, status_every=100, on_change=_on_gpio_change)
    finally:
        cleanup_gpio()
        print("[MAIN] GPIO monitor stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
