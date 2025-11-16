import socket
import threading
import base64
import json
import os
from typing import Dict, Optional, List, Tuple

from audio_devices import (
    select_output_device_for_channel,
    open_output_stream,
    close_stream,
)
from gpio_monitor import GPIO_PINS, gpio_states

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    HAS_AES = True
except Exception:
    AESGCM = None  # type: ignore
    HAS_AES = False

try:
    import opuslib  # type: ignore
    HAS_OPUS = True
except Exception:
    opuslib = None  # type: ignore
    HAS_OPUS = False


class UDPPlayer:
    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._streams: Dict[int, Dict[str, object]] = {}  # channel_index -> { 'pa': pa, 'stream': stream }
        self._lock = threading.Lock()
        self._aesgcm: Optional[AESGCM] = None
        self._opus_decoder = None
        self._channel_ids: List[str] = []
        self._server_addr: Optional[Tuple[str, int]] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_running = threading.Event()
        self._hb_count = 0
        # If ES_UDP_HEARTBEAT_LOG=0, completely suppress heartbeat logs
        self._suppress_hb_log = os.getenv("ES_UDP_HEARTBEAT_LOG", "1") == "0"

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

    def set_channel_ids(self, channel_ids: List[str]) -> None:
        # Preserve order to map index
        self._channel_ids = [str(c).strip() for c in channel_ids if str(c).strip()]

    def _map_channel_id_to_index(self, channel_id: str) -> Optional[int]:
        if not channel_id:
            return None
        if self._channel_ids:
            try:
                return self._channel_ids.index(channel_id)
            except ValueError:
                return None
        return None

    def _close_all_streams(self) -> None:
        with self._lock:
            for ch_idx, bundle in list(self._streams.items()):
                pa = bundle.get("pa")
                stream = bundle.get("stream")
                if pa and stream:
                    close_stream(pa, stream)
            self._streams.clear()

    def _decrypt_and_decode(self, b64_data: str) -> Optional[bytes]:
        try:
            enc = base64.b64decode(b64_data)
        except Exception:
            return None
        data = enc
        if self._aesgcm:
            if len(enc) < 28:
                return None
            iv = enc[:12]
            ct = enc[12:-16]
            tag = enc[-16:]
            try:
                data = self._aesgcm.decrypt(iv, ct + tag, None)
            except Exception:
                return None
        if not HAS_OPUS or self._opus_decoder is None:
            return None
        try:
            pcm = self._opus_decoder.decode(data, frame_size=1920, decode_fec=0)  # type: ignore[attr-defined]
            import array
            arr = array.array("h", pcm)  # type: ignore[arg-type]
            return arr.tobytes()
        except Exception:
            return None

    def _receiver_loop(self) -> None:
        print("[UDP] Receiver thread started")
        while self._running.is_set():
            try:
                data = self._sock.recv(8192)  # nosec
                if not data:
                    continue
                try:
                    msg = json.loads(data.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") != "audio":
                    continue
                ch_id = str(msg.get("channel_id", "")).strip()
                b64 = msg.get("data", "")
                if not ch_id or not b64:
                    continue
                pcm = self._decrypt_and_decode(b64)
                if not pcm:
                    continue
                # Prefer explicit channel_id mapping; fallback to GPIO active mapping
                target_index = self._map_channel_id_to_index(ch_id)
                target_indices: List[int]
                if target_index is not None:
                    target_indices = [target_index]
                else:
                    # Fallback: mirror to all ACTIVE GPIO channels
                    active_gpio = [g for g, val in gpio_states.items() if val == 0]
                    gpio_keys: List[int] = list(GPIO_PINS.keys())
                    target_indices = [gpio_keys.index(g) for g in active_gpio if g in gpio_keys]
                for ch_index in target_indices:
                    self._ensure_stream_for_channel(ch_index)
                    bundle = self._streams.get(ch_index)
                    if bundle and pcm:
                        try:
                            stream = bundle["stream"]
                            stream.write(pcm)  # type: ignore[attr-defined]
                        except Exception as e:
                            print(f"[UDP] WARNING: write failed for channel index {ch_index}: {e}")
            except socket.timeout:
                continue
            except OSError as e:
                if not self._running.is_set():
                    break
                print(f"[UDP] WARNING: socket error while running: {e}")
                continue
            except Exception as e:
                print(f"[UDP] ERROR: receiver loop exception: {e}")
        print("[UDP] Receiver thread exiting")

    def _heartbeat_loop(self) -> None:
        if not self._server_addr or not self._sock:
            return
        print("[UDP] Heartbeat thread started")
        while self._hb_running.is_set():
            try:
                # Keep NAT mapping alive; server expects JSON KEEP_ALIVE
                self._sock.send(b'{"type":"KEEP_ALIVE"}')  # nosec
                # Throttle heartbeat logs (every 60th ~10 minutes) or suppress via env
                if not self._suppress_hb_log:
                    self._hb_count += 1
                    if self._hb_count % 60 == 1 or os.getenv("ES_UDP_DEBUG"):
                        try:
                            la = self._sock.getsockname()
                            print(f"[UDP] HEARTBEAT sent from {la} to {self._server_addr}")
                        except Exception:
                            print("[UDP] HEARTBEAT sent")
            except Exception:
                pass
            self._hb_running.wait(10.0)
        print("[UDP] Heartbeat thread stopped")

    def start(self, udp_port: int, host_hint: str = "", aes_key_b64: str = "") -> bool:
        if self._sock is not None:
            print("[UDP] Already running")
            return True
        try:
            # Init crypto/decoder if available
            if aes_key_b64 and aes_key_b64 not in ("", "N/A") and HAS_AES:
                try:
                    key = base64.b64decode(aes_key_b64)
                    self._aesgcm = AESGCM(key)
                except Exception as e:
                    print(f"[UDP] WARNING: AES key init failed: {e}")
                    self._aesgcm = None
            if HAS_OPUS:
                try:
                    self._opus_decoder = opuslib.Decoder(48000, 1)  # mono
                except Exception as e:
                    print(f"[UDP] WARNING: Opus decoder init failed: {e}")
                    self._opus_decoder = None
            else:
                self._opus_decoder = None

            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(1.0)

            # Connect to server host:port using UDP so replies return to this socket.
            # The OS picks an ephemeral local port that the server will reply to.
            # Allow environment override for UDP host (useful when server advertises public IP but audio flows on overlay)
            host_override = os.getenv("ES_UDP_HOST_OVERRIDE", "").strip()
            host_used = host_override or host_hint
            if not host_used:
                print("[UDP] ERROR: No udp_host provided by server")
                self.stop()
                return False
            self._server_addr = (host_used, int(udp_port))
            try:
                self._sock.connect(self._server_addr)
            except Exception as e:
                print(f"[UDP] ERROR: UDP connect failed to {self._server_addr}: {e}")
                self.stop()
                return False

            # Immediate heartbeat to open the path
            try:
                self._sock.send(b'{"type":"KEEP_ALIVE"}')  # nosec
                try:
                    la = self._sock.getsockname()
                    print(f"[UDP] Local addr: {la} -> {self._server_addr}")
                except Exception:
                    pass
            except Exception as e:
                print(f"[UDP] WARNING: initial heartbeat failed: {e}")

            # Start periodic heartbeat
            self._hb_running.set()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

            print(f"[UDP] Connected to server {host_used}:{udp_port} (ephemeral local port in use)")
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
            if self._hb_running.is_set():
                self._hb_running.clear()
            if self._hb_thread and self._hb_thread.is_alive():
                self._hb_thread.join(timeout=10.0)
            self._hb_thread = None
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


