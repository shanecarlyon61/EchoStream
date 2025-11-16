import socket
import threading
import base64
import json
import os
import time
import errno
from typing import Dict, Optional, List, Tuple

from audio_devices import (
    select_output_device_for_channel,
    open_output_stream,
    close_stream,
)

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
        self._receive_count = 0  # Track received packets for logging
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
        print(f"[UDP] Channel IDs configured: {self._channel_ids}")
        print(f"[UDP] AES decryptor: {'READY' if self._aesgcm else 'NOT AVAILABLE'}")
        print(f"[UDP] Opus decoder: {'READY' if self._opus_decoder else 'NOT AVAILABLE'}")
        if self._sock:
            try:
                local_addr = self._sock.getsockname()
                print(f"[UDP] Socket bound to: {local_addr}")
            except Exception as e:
                print(f"[UDP] WARNING: Could not get socket address: {e}")
        
        loop_count = 0
        last_log_time = time.time()
        while self._running.is_set():
            loop_count += 1
            # Log every 10 seconds that we're still listening
            current_time = time.time()
            if current_time - last_log_time >= 10.0:
                print(f"[UDP] Still listening... (loop iterations: {loop_count}, packets received: {self._receive_count})")
                last_log_time = current_time
                if self._sock:
                    try:
                        local_addr = self._sock.getsockname()
                        print(f"[UDP] Socket still bound to: {local_addr}")
                    except Exception:
                        pass
            
            try:
                # Use recvfrom() like C code - receives from ANY address
                # This is important because server may send from different IP/port
                data, addr = self._sock.recvfrom(8192)  # nosec
                
                if not data:
                    continue
                
                # Log every packet initially, then every 1000th (for debugging)
                self._receive_count += 1
                if self._receive_count <= 10 or self._receive_count % 1000 == 0:
                    print(f"[UDP] Received {len(data)} bytes from {addr} (count: {self._receive_count})")
                
                try:
                    msg = json.loads(data.decode("utf-8", errors="ignore"))
                except Exception as e:
                    if self._receive_count <= 10:
                        print(f"[UDP] JSON decode failed: {e}, data preview: {data[:100]}")
                    continue
                    
                if not isinstance(msg, dict):
                    if self._receive_count <= 10:
                        print(f"[UDP] Message is not a dict: {type(msg)}")
                    continue
                    
                msg_type = msg.get("type", "")
                if msg_type != "audio":
                    if self._receive_count <= 10:
                        print(f"[UDP] Non-audio message type: '{msg_type}', ignoring")
                    continue
                    
                ch_id = str(msg.get("channel_id", "")).strip()
                b64 = msg.get("data", "")
                
                if not ch_id or not b64:
                    if self._receive_count <= 10:
                        print(f"[UDP] Missing channel_id or data: ch_id='{ch_id}', has_data={bool(b64)}")
                    continue
                
                if self._receive_count <= 10:
                    print(f"[UDP] Processing audio packet: channel_id='{ch_id}', data_len={len(b64)}")
                    
                pcm = self._decrypt_and_decode(b64)
                if not pcm:
                    if self._receive_count <= 10:
                        print(f"[UDP] Decrypt/decode failed for channel '{ch_id}'")
                    continue
                
                if self._receive_count <= 10:
                    print(f"[UDP] Successfully decoded {len(pcm)} bytes PCM for channel '{ch_id}'")
                    
                # Find channel by channel_id (matching C code behavior)
                target_index = self._map_channel_id_to_index(ch_id)
                if target_index is None:
                    # Debug: log when channel not found (like C code)
                    if os.getenv("ES_UDP_DEBUG"):
                        print(f"[UDP] No active channel found for '{ch_id}'")
                        print(f"[UDP] Active channels: {self._channel_ids}")
                    continue
                    
                self._ensure_stream_for_channel(target_index)
                bundle = self._streams.get(target_index)
                if bundle and pcm:
                    try:
                        stream = bundle["stream"]
                        stream.write(pcm)  # type: ignore[attr-defined]
                    except Exception as e:
                        print(f"[UDP] WARNING: write failed for channel index {target_index}: {e}")
                        
            except OSError as e:
                if not self._running.is_set():
                    break
                # Handle EAGAIN/EWOULDBLOCK like C code
                if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                    # No data available - continue waiting
                    continue
                print(f"[UDP] WARNING: socket error: {e} (errno={e.errno})")
                import traceback
                traceback.print_exc()
                continue
            except Exception as e:
                print(f"[UDP] ERROR: receiver loop exception: {e}")
                import traceback
                traceback.print_exc()
        print("[UDP] Receiver thread exiting")

    def _heartbeat_loop(self) -> None:
        if not self._server_addr or not self._sock:
            return
        print("[UDP] Heartbeat thread started")
        while self._hb_running.is_set():
            try:
                # Use sendto() like C code - keeps NAT mapping alive
                heartbeat_msg = b'{"type":"KEEP_ALIVE"}'
                self._sock.sendto(heartbeat_msg, self._server_addr)  # nosec
                
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
            # Sleep for 10 seconds, checking event state periodically
            for _ in range(100):  # Check every 0.1 seconds for responsiveness
                if not self._hb_running.is_set():
                    break
                time.sleep(0.1)
        print("[UDP] Heartbeat thread stopped")

    def start(self, udp_port: int, host_hint: str = "", aes_key_b64: str = "") -> bool:
        if self._sock is not None:
            print("[UDP] Already running")
            return True
        try:
            # Init crypto/decoder if available
            # Use hardcoded key from C code if server sends 'N/A' or empty
            # C code uses: "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0="
            if HAS_AES:
                # Always try to init AES - use hardcoded key if server doesn't provide one
                key_b64_to_use = aes_key_b64 if (aes_key_b64 and aes_key_b64 not in ("", "N/A")) else "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0="
                try:
                    key = base64.b64decode(key_b64_to_use)
                    if len(key) == 32:  # AES-256 requires 32 bytes
                        self._aesgcm = AESGCM(key)
                        key_source = 'server' if (aes_key_b64 and aes_key_b64 not in ("", "N/A")) else 'hardcoded'
                        print(f"[UDP] AES key initialized (using {key_source} key)")
                    else:
                        print(f"[UDP] WARNING: AES key length invalid: {len(key)} bytes (expected 32)")
                        self._aesgcm = None
                except Exception as e:
                    print(f"[UDP] WARNING: AES key init failed: {e}")
                    self._aesgcm = None
            else:
                print("[UDP] WARNING: AES library not available")
                self._aesgcm = None
            if HAS_OPUS:
                try:
                    self._opus_decoder = opuslib.Decoder(48000, 1)  # mono
                except Exception as e:
                    print(f"[UDP] WARNING: Opus decoder init failed: {e}")
                    self._opus_decoder = None
            else:
                self._opus_decoder = None

            # Create UDP socket WITHOUT connecting (like C code)
            # This allows receiving from any address using recvfrom()
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Don't set timeout - let it block like C code does
            # This ensures we're always ready to receive packets immediately
            
            # Don't bind to a specific port - let OS assign ephemeral port
            # This matches C code behavior where socket uses ephemeral port
            
            # Allow environment override for UDP host (useful when server advertises public IP but audio flows on overlay)
            host_override = os.getenv("ES_UDP_HOST_OVERRIDE", "").strip()
            host_used = host_override or host_hint
            if not host_used:
                print("[UDP] ERROR: No udp_host provided by server")
                self.stop()
                return False
                
            self._server_addr = (host_used, int(udp_port))
            
            # Send initial heartbeat using sendto() (like C code)
            # This establishes NAT mapping and tells server where to send packets
            try:
                heartbeat_msg = b'{"type":"KEEP_ALIVE"}'
                self._sock.sendto(heartbeat_msg, self._server_addr)  # nosec
                
                # Log local address (ephemeral port assigned by OS)
                try:
                    la = self._sock.getsockname()
                    print(f"[UDP] Local addr: {la} -> Server: {self._server_addr}")
                    print("[UDP] Initial heartbeat sent to establish NAT mapping")
                except Exception:
                    pass
            except Exception as e:
                print(f"[UDP] WARNING: initial heartbeat failed: {e}")

            # Start periodic heartbeat
            self._hb_running.set()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

            print(f"[UDP] Listening on ephemeral port, server: {host_used}:{udp_port}")
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


