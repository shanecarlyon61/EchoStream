
import threading
from typing import Optional, Dict, List
from echostream import MAX_CHANNELS, global_channel_ids, global_channel_count

import threading

class ChannelContext:
    def __init__(self):
        self.audio: Optional[AudioStream] = None
        self.active: bool = False
        self.channel_id: str = ""
        self.device_index: int = -1
        self.gpio_num: Optional[int] = None
        self.thread_input: Optional[threading.Thread] = None
        self.thread_output: Optional[threading.Thread] = None

channels: Dict[str, ChannelContext] = {}

channels_mutex = threading.Lock()

def create_channel(channel_id: str, device_index: int) -> bool:

    import audio_stream
    import audio_hardware
    import gpio

    global channels, channels_mutex

    if audio_hardware.pa_instance is None:
        print(f"[CHANNEL_MGR] ERROR: PyAudio not initialized")
        return False

    with channels_mutex:

        if channel_id in channels:
            print(f"[CHANNEL_MGR] Channel {channel_id} already exists")
            return True

        ctx = ChannelContext()
        ctx.channel_id = channel_id
        ctx.device_index = device_index

        stream = audio_stream.create_stream(channel_id, device_index, audio_hardware.pa_instance)
        if stream is None:
            print(f"[CHANNEL_MGR] ERROR: Failed to create audio stream for {channel_id}")
            return False

        ctx.audio = stream

        channel_index = -1
        for i in range(MAX_CHANNELS):
            if i < len(global_channel_ids) and global_channel_ids[i] == channel_id:
                channel_index = i
                break

        if channel_index >= 0:
            ctx.gpio_num = gpio.get_gpio_for_channel(channel_index)

        channels[channel_id] = ctx
        ctx.active = True

        print(f"[CHANNEL_MGR] Channel {channel_id} created (device {device_index})")
        return True

def start_channel(channel_id: str) -> bool:

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

        if not audio_stream.start_stream(ctx.audio, audio_hardware.pa_instance):
            print(f"[CHANNEL_MGR] ERROR: Failed to start streams for {channel_id}")
            return False

        if ctx.thread_input is None or not ctx.thread_input.is_alive():
            ctx.thread_input = threading.Thread(
                target=audio_workers.audio_input_worker,
                args=(ctx.audio,),
                daemon=True
            )
            ctx.thread_input.start()
            print(f"[CHANNEL_MGR] Input worker started for channel {channel_id}")

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

    import audio_stream

    global channels, channels_mutex

    with channels_mutex:
        if channel_id not in channels:
            return

        ctx = channels[channel_id]

        if ctx.audio:
            audio_stream.stop_stream(ctx.audio)

        if ctx.thread_input and ctx.thread_input.is_alive():
            ctx.thread_input.join(timeout=1.0)

        if ctx.thread_output and ctx.thread_output.is_alive():
            ctx.thread_output.join(timeout=1.0)

        print(f"[CHANNEL_MGR] Channel {channel_id} stopped")

def handle_gpio_change(channel_id: str, gpio_active: bool) -> bool:

    import udp_manager
    import websocket_client

    global channels, channels_mutex

    with channels_mutex:
        if channel_id not in channels:
            return False

        ctx = channels[channel_id]
        if ctx.audio is None:
            return False

        was_active = ctx.audio.gpio_active
        ctx.audio.gpio_active = gpio_active

        if gpio_active and not was_active:

            if udp_manager.is_udp_ready():
                if not ctx.audio.transmitting:

                    if start_channel(channel_id):
                        print(f"[CHANNEL_MGR] Started transmission for channel {channel_id} (GPIO active, UDP ready)")

                    websocket_client.send_transmit_event(channel_id, True)

        if not gpio_active and was_active:
            websocket_client.send_transmit_event(channel_id, False)

        return True

def set_channel_encryption_key(channel_id: str, key: bytes) -> bool:

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

    global channels, channels_mutex

    with channels_mutex:
        return channels.get(channel_id)

def get_all_channels() -> List[str]:

    global channels, channels_mutex

    with channels_mutex:
        return list(channels.keys())

def cleanup_all_channels():

    import audio_stream

    global channels, channels_mutex

    with channels_mutex:
        channel_ids = list(channels.keys())

        for channel_id in channel_ids:
            ctx = channels[channel_id]

            if ctx.audio:
                audio_stream.stop_stream(ctx.audio)
                audio_stream.cleanup_stream(ctx.audio)

        channels.clear()

        print("[CHANNEL_MGR] All channels cleaned up")

def register_channels_with_websocket():

    import websocket_client

    global channels, channels_mutex

    with channels_mutex:
        for channel_id, ctx in channels.items():
            if ctx.active:

                websocket_client.register_channel(channel_id)

                websocket_client.send_connect_message(channel_id)

                if ctx.audio and ctx.audio.gpio_active:
                    websocket_client.send_transmit_event(channel_id, True)

