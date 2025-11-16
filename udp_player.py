import socket
import threading
from typing import Dict, Optional, List

from audio_devices import (
    select_output_device_for_channel,
    open_output_stream,
    close_stream,
)
from gpio_monitor import GPIO_PINS, gpio_states


class UDPPlayer:
    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._streams: Dict[int, Dict[str, object]] = {}  # channel_index -> { 'pa': pa, 'stream': stream }
        self._lock = threading.Lock()

    def _ensure_stream_for_channel(self, channel_index: int) -> None:
        with self._lock:
            if channel_index in self._streams:
                return
            device_index = select_output_device_for_channel(channel_index)
            if device_index is None:
                print(f"[AUDIO] WARNING: No device available for channel index {channel_index}")
                return
            pa, stream = open_output_stream(device_index)
            if pa is None or stream is None:
                return
            self._streams[channel_index] = {"pa": pa, "stream": stream}  # keep PA for lifecycle consistency
            print(f"[AUDIO] Output stream ready on device {device_index} for channel index {channel_index}")

    def _close_all_streams(self) -> None:
        with self._lock:
            for ch_idx, bundle in list(self._streams.items()):
                pa = bundle.get("pa")
                stream = bundle.get("stream")
                if pa and stream:
                    close_stream(pa, stream)
            self._streams.clear()

    def _receiver_loop(self) -> None:
        print("[UDP] Receiver thread started")
        while self._running.is_set():
            try:
                data, _addr = self._sock.recvfrom(8192)  # nosec - UDP audio frames
                if not data:
                    continue
                # Heuristic: write the same audio to all GPIO ACTIVE channels for now
                active_gpio = [g for g, val in gpio_states.items() if val == 0]
                # Channel index is by GPIO order
                gpio_keys: List[int] = list(GPIO_PINS.keys())
                for g in active_gpio:
                    if g in gpio_keys:
                        ch_index = gpio_keys.index(g)
                        self._ensure_stream_for_channel(ch_index)
                        bundle = self._streams.get(ch_index)
                        if bundle:
                            try:
                                stream = bundle["stream"]
                                stream.write(data)  # type: ignore[attr-defined]
                            except Exception as e:
                                print(f"[UDP] WARNING: write failed for channel index {ch_index}: {e}")
            except OSError:
                if self._running.is_set():
                    print("[UDP] WARNING: socket error while running")
                break
            except Exception as e:
                print(f"[UDP] ERROR: receiver loop exception: {e}")
        print("[UDP] Receiver thread exiting")

    def start(self, udp_port: int, host_hint: str = "") -> bool:
        if self._sock is not None:
            print("[UDP] Already running")
            return True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Bind locally to receive server audio; host_hint logged for debugging
            self._sock.bind(("0.0.0.0", int(udp_port)))
            self._sock.settimeout(1.0)
            print(f"[UDP] Listening for audio on 0.0.0.0:{udp_port} (server={host_hint})")
            self._running.set()
            self._recv_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self._recv_thread.start()
            return True
        except Exception as e:
            print(f"[UDP] ERROR: Failed to start UDP listener on {udp_port}: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        try:
            self._running.clear()
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
            if self._recv_thread and self._recv_thread.is_alive():
                self._recv_thread.join(timeout=1.0)
            self._recv_thread = None
        finally:
            self._close_all_streams()
            print("[UDP] Stopped")


# Singleton instance used by main via websocket callback
global_udp_player = UDPPlayer()


