import socket
import threading
import base64
import json
import os
import time
import errno
import queue
import numpy as np
from typing import Dict, Optional, List, Tuple, Any

from audio_devices import (
    select_output_device_for_channel,
    open_output_stream,
    close_stream,
    select_input_device_for_channel,
    open_input_stream,
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

try:
    from config import (
        load_config,
        get_frequency_filters,
        get_tone_detect_config,
        get_tone_definitions,
        get_new_tone_config,
        get_passthrough_config,
        get_channel_ids,
    )
    from frequency_filter import apply_audio_frequency_filters
    from tone_detection import (
        init_channel_detector,
        process_audio_for_channel,
        add_audio_samples_for_channel,
    )
    from passthrough import global_passthrough_manager
    from recording import global_recording_manager

    HAS_FREQ_FILTER = True
    HAS_TONE_DETECT = True
    HAS_PASSTHROUGH = True
    HAS_RECORDING = True
except Exception as e:
    print(f"[UDP] WARNING: Frequency filtering/tone detection not available: {e}")
    HAS_FREQ_FILTER = False
    HAS_TONE_DETECT = False
    HAS_PASSTHROUGH = False
    load_config = None
    get_frequency_filters = None
    get_tone_detect_config = None
    get_tone_definitions = None
    get_new_tone_config = None
    get_passthrough_config = None
    get_channel_ids = None
    apply_audio_frequency_filters = None
    init_channel_detector = None
    process_audio_for_channel = None
    add_audio_samples_for_channel = None
    global_passthrough_manager = None
    global_recording_manager = None
    HAS_RECORDING = False


class UDPPlayer:
    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        # channel_index -> { 'pa': pa, 'stream': stream }
        self._streams: Dict[int, Dict[str, object]] = {}
        # channel_index -> { 'pa': pa, 'stream': stream, 'encoder': encoder,
        # 'buffer': buffer, 'buffer_pos': int, 'thread': thread }
        self._input_streams: Dict[int, Dict[str, object]] = {}
        self._lock = threading.Lock()
        self._input_lock = threading.Lock()
        self._aesgcm: Optional[AESGCM] = None
        self._aes_key: Optional[bytes] = None  # Store key for encryption
        self._opus_decoder = None
        self._channel_ids: List[str] = []
        self._server_addr: Optional[Tuple[str, int]] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_running = threading.Event()
        self._hb_count = 0
        self._receive_count = 0  # Track received packets for logging
        # channel_index -> transmitting state
        self._transmitting: Dict[int, bool] = {}
        # If ES_UDP_HEARTBEAT_LOG=0, completely suppress heartbeat logs
        self._suppress_hb_log = os.getenv("ES_UDP_HEARTBEAT_LOG", "1") == "0"
        self._frequency_filters: Dict[str, List[Dict[str, Any]]] = {}
        self._tone_detect_enabled: Dict[str, bool] = {}
        self._config_cache: Optional[Dict[str, Any]] = None

        # Tone detection threads with queues (no buffer copying)
        # Broadcasting thread puts audio references in queue
        # Tone detection thread processes from queue
        self._tone_detect_queues: Dict[str, queue.Queue] = {}
        self._tone_detect_threads: Dict[str, threading.Thread] = {}
        self._tone_detect_running: Dict[str, threading.Event] = {}
        
        # Track input worker threads for health monitoring
        self._input_worker_threads: Dict[int, threading.Thread] = {}
        
        # Track last packet send time for each channel to detect stalled threads
        self._last_packet_time: Dict[int, float] = {}
        
        # Health monitoring thread
        self._health_monitor_thread: Optional[threading.Thread] = None
        
        # Start health monitor immediately
        self._start_health_monitor()

    def _ensure_stream_for_channel(self, channel_index: int) -> None:
        with self._lock:
            if channel_index in self._streams:
                bundle = self._streams[channel_index]
                stream = bundle.get("stream")
                # Ensure stream is started
                if stream and hasattr(stream, "is_active") and not stream.is_active():
                    try:
                        stream.start_stream()  # type: ignore[attr-defined]
                        print(
                            f"[AUDIO] Started output stream for channel index {channel_index}"
                        )
                    except Exception as e:
                        print(
                            f"[AUDIO] WARNING: Failed to start stream for channel {channel_index}: {e}"
                        )
                return
            device_index = select_output_device_for_channel(channel_index)
            if device_index is None:
                print(
                    f"[AUDIO] WARNING: No device available for channel index {channel_index}"
                )
                return
            pa, stream = open_output_stream(device_index)
            if pa is None or stream is None:
                print(
                    f"[AUDIO] ERROR: Failed to open output stream for channel index {channel_index}"
                )
                return
            # Start the stream immediately
            try:
                stream.start_stream()  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[AUDIO] WARNING: Failed to start stream: {e}")
            # keep PA for lifecycle consistency
            self._streams[channel_index] = {"pa": pa, "stream": stream}
            print(
                f"[AUDIO] Output stream ready and started on device {device_index} for channel index {channel_index}"
            )

    def set_channel_ids(self, channel_ids: List[str]) -> None:
        # Preserve order to map index
        self._channel_ids = [str(c).strip() for c in channel_ids if str(c).strip()]
        if HAS_PASSTHROUGH and global_passthrough_manager:
            all_channel_ids = []
            if load_config and get_channel_ids:
                if self._config_cache is None:
                    self._config_cache = load_config()
                all_channel_ids = get_channel_ids(self._config_cache)
            if all_channel_ids:
                global_passthrough_manager.set_channel_mapping(all_channel_ids)
            else:
                global_passthrough_manager.set_channel_mapping(self._channel_ids)
        self._load_frequency_filters()

    def _load_frequency_filters(self) -> None:
        if not HAS_FREQ_FILTER:
            return
        try:
            if self._config_cache is None:
                self._config_cache = load_config()
            self._frequency_filters.clear()
            self._tone_detect_enabled.clear()
            tone_detect_map = {}
            if get_tone_detect_config:
                tone_detect_map = dict(get_tone_detect_config(self._config_cache))
            for channel_id in self._channel_ids:
                is_tone_detect_enabled = tone_detect_map.get(channel_id, False)
                self._tone_detect_enabled[channel_id] = is_tone_detect_enabled
                if is_tone_detect_enabled:
                    filters = get_frequency_filters(self._config_cache, channel_id)
                    if filters:
                        self._frequency_filters[channel_id] = filters
                        print(
                            f"[UDP] Loaded {len(filters)} frequency filter(s) "
                            f"for tone detection on channel {channel_id}"
                        )
                    else:
                        self._frequency_filters[channel_id] = []

                    if (
                        HAS_TONE_DETECT
                        and get_tone_definitions
                        and init_channel_detector
                    ):
                        tone_defs = get_tone_definitions(self._config_cache, channel_id)
                        new_tone_cfg = None
                        if get_new_tone_config:
                            new_tone_cfg = get_new_tone_config(
                                self._config_cache, channel_id
                            )
                        passthrough_cfg = None
                        if get_passthrough_config:
                            passthrough_cfg = get_passthrough_config(
                                self._config_cache, channel_id
                            )
                        if tone_defs or (
                            new_tone_cfg and new_tone_cfg.get("detect_new_tones", False)
                        ):
                            freq_filters = self._frequency_filters.get(channel_id, [])
                            init_channel_detector(
                                channel_id, tone_defs, new_tone_cfg, passthrough_cfg, freq_filters
                            )
                            print(
                                f"[UDP] Initialized tone detection for channel {channel_id} "
                                f"with {len(tone_defs)} tone definition(s)"
                            )
                        else:
                            print(
                                f"[UDP] WARNING: Tone detection enabled for channel {channel_id} "
                                f"but no tone definitions found in config.json"
                            )
                else:
                    self._frequency_filters[channel_id] = []
        except Exception as e:
            print(f"[UDP] WARNING: Failed to load frequency filters: {e}")

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
            pcm = self._opus_decoder.decode(
                data, frame_size=1920, decode_fec=0
            )  # type: ignore[attr-defined]
            # Convert to bytes directly (opuslib returns bytes)
            return pcm if isinstance(pcm, bytes) else bytes(pcm)
        except Exception:
            return None

    def _start_health_monitor(self) -> None:
        """Start health monitor thread if not already running."""
        if self._health_monitor_thread is None or not self._health_monitor_thread.is_alive():
            self._health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True,
                name="UDPHealthMonitor"
            )
            self._health_monitor_thread.start()
            print("[UDP] Started health monitoring thread")
    
    def _health_monitor_loop(self) -> None:
        """Periodically check health of input worker threads and restart dead/stalled ones."""
        print("[UDP] Health monitor thread started")
        status_report_count = 0
        while self._running.is_set():
            try:
                time.sleep(5.0)  # Check every 5 seconds
                current_time = time.time()
                status_report_count += 1
                
                # Log status every 12 checks (1 minute)
                if status_report_count % 12 == 0:
                    with self._input_lock:
                        active_threads = sum(1 for t in self._input_worker_threads.values() if t.is_alive())
                        transmitting_count = sum(1 for v in self._transmitting.values() if v)
                        print(f"[UDP HEALTH] Status: {active_threads} active threads, {transmitting_count} transmitting channels")
                
                with self._input_lock:
                    # Check all input worker threads
                    dead_threads = []
                    stalled_threads = []
                    
                    for channel_index, thread in list(self._input_worker_threads.items()):
                        if channel_index >= len(self._channel_ids):
                            continue
                            
                        channel_id = self._channel_ids[channel_index]
                        
                        # Check if thread is dead
                        if not thread.is_alive():
                            dead_threads.append(channel_index)
                            print(f"[UDP HEALTH] Thread for channel {channel_id} is DEAD")
                        # Check if thread is alive but not sending packets (stalled)
                        elif self._transmitting.get(channel_index, False):
                            last_packet_time = self._last_packet_time.get(channel_index, 0)
                            time_since_last_packet = current_time - last_packet_time
                            
                            # If no packets sent in last 10 seconds, thread is likely stalled
                            # OR if last_packet_time is 0 and thread has been running, it's stuck
                            if (time_since_last_packet > 10.0 and last_packet_time > 0) or (last_packet_time == 0 and time_since_last_packet > 5.0):
                                stalled_threads.append(channel_index)
                                if last_packet_time == 0:
                                    print(f"[UDP HEALTH] Thread for channel {channel_id} is STUCK (never sent packets)")
                                else:
                                    print(f"[UDP HEALTH] Thread for channel {channel_id} is STALLED (no packets for {time_since_last_packet:.1f}s)")
                    
                    # Restart dead or stalled threads
                    threads_to_restart = dead_threads + stalled_threads
                    for channel_index in threads_to_restart:
                        if channel_index < len(self._channel_ids):
                            channel_id = self._channel_ids[channel_index]
                            print(f"[UDP HEALTH] Restarting transmission for channel {channel_id}...")
                            
                            # Check if channel should be transmitting
                            if self._transmitting.get(channel_index, False):
                                try:
                                    # Remove old thread reference
                                    if channel_index in self._input_worker_threads:
                                        del self._input_worker_threads[channel_index]
                                    
                                    # Restart transmission
                                    if self.start_transmission_for_channel(channel_index):
                                        print(f"[UDP HEALTH] ✓ Successfully restarted transmission for channel {channel_id}")
                                        # Reset last packet time
                                        self._last_packet_time[channel_index] = current_time
                                    else:
                                        print(f"[UDP HEALTH] ✗ Failed to restart transmission for channel {channel_id}")
                                except Exception as e:
                                    print(f"[UDP HEALTH] ERROR restarting channel {channel_id}: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
            except Exception as e:
                print(f"[UDP HEALTH] ERROR in health monitor: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5.0)
        
        print("[UDP] Health monitor thread stopped")
    
    def _receiver_loop(self) -> None:
        print("[UDP] Receiver thread started")
        print(f"[UDP] Channel IDs configured: {self._channel_ids}")
        print(
            f"[UDP] AES decryptor: " f"{'READY' if self._aesgcm else 'NOT AVAILABLE'}"
        )
        print(
            f"[UDP] Opus decoder: "
            f"{'READY' if self._opus_decoder else 'NOT AVAILABLE'}"
        )
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
                print(
                    f"[UDP] Still listening... (loop iterations: {loop_count}, "
                    f"packets received: {self._receive_count})"
                )
                last_log_time = current_time
                if self._sock:
                    try:
                        local_addr = self._sock.getsockname()
                        print(f"[UDP] Socket still bound to: {local_addr}")
                    except Exception:
                        pass

            try:
                # Use recvfrom() like C code - receives from ANY address
                # This is important because server may send from different
                # IP/port
                data, addr = self._sock.recvfrom(8192)  # nosec

                if not data:
                    continue

                # Log every packet initially, then every 1000th (for debugging)
                self._receive_count += 1
                if self._receive_count <= 10 or self._receive_count % 1000 == 0:
                    print(
                        f"[UDP] Received {len(data)} bytes from {addr} "
                        f"(count: {self._receive_count})"
                    )

                try:
                    msg = json.loads(data.decode("utf-8", errors="ignore"))
                except Exception as e:
                    if self._receive_count <= 10:
                        print(
                            f"[UDP] JSON decode failed: {e}, data preview: {data[:100]}"
                        )
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
                        print(
                            f"[UDP] Missing channel_id or data: ch_id='{ch_id}', "
                            f"has_data={bool(b64)}"
                        )
                    continue

                if self._receive_count <= 10:
                    print(
                        f"[UDP] Processing audio packet: channel_id='{ch_id}', "
                        f"data_len={len(b64)}"
                    )

                pcm = self._decrypt_and_decode(b64)
                if not pcm:
                    if self._receive_count <= 10:
                        print(f"[UDP] Decrypt/decode failed for channel '{ch_id}'")
                    continue

                if self._receive_count <= 10:
                    print(
                        f"[UDP] Successfully decoded {len(pcm)} bytes PCM "
                        f"for channel '{ch_id}'"
                    )

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
                        # Ensure stream is started
                        if not stream.is_active():  # type: ignore[attr-defined]
                            stream.start_stream()  # type: ignore[attr-defined]
                        # Write PCM data in chunks to avoid buffer issues
                        # 1920 samples = 3840 bytes (int16), write in
                        # 1024-sample chunks
                        pcm_len = len(pcm)
                        bytes_per_sample = 2
                        samples_per_chunk = 1024
                        bytes_per_chunk = samples_per_chunk * bytes_per_sample
                        written = 0
                        while written < pcm_len:
                            chunk_size = min(bytes_per_chunk, pcm_len - written)
                            chunk = pcm[written : written + chunk_size]
                            try:
                                # type: ignore[attr-defined]
                                stream.write(chunk)
                                written += chunk_size
                            except Exception as write_err:
                                if self._receive_count <= 10:
                                    print(
                                        f"[UDP] ERROR: stream.write() failed: {write_err}"
                                    )
                                break
                        if self._receive_count <= 10:
                            print(
                                f"[UDP] Wrote {written} bytes PCM to channel {target_index} (channel_id: {ch_id})"
                            )
                    except Exception as e:
                        print(
                            f"[UDP] WARNING: write failed for channel index {target_index}: {e}"
                        )
                        import traceback

                        traceback.print_exc()

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

                # Throttle heartbeat logs (every 60th ~10 minutes) or suppress
                # via env
                if not self._suppress_hb_log:
                    self._hb_count += 1
                    if self._hb_count % 60 == 1 or os.getenv("ES_UDP_DEBUG"):
                        try:
                            la = self._sock.getsockname()
                            print(
                                f"[UDP] HEARTBEAT sent from {la} to "
                                f"{self._server_addr}"
                            )
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
        host_override = os.getenv("ES_UDP_HOST_OVERRIDE", "").strip()
        host_used = host_override or host_hint
        new_server_addr = (host_used, int(udp_port)) if host_used else None
        
        if self._sock is not None and self._server_addr is not None:
            # Verify socket is still valid
            try:
                # Try to get socket info to verify it's still open
                self._sock.getsockname()
                
                # Check if server address has changed
                if new_server_addr and self._server_addr != new_server_addr:
                    print(f"[UDP] Server address changed from {self._server_addr} to {new_server_addr}, updating...")
                    self._server_addr = new_server_addr
                    # Send a heartbeat to the new address to verify connectivity
                    try:
                        heartbeat_msg = b'{"type":"KEEP_ALIVE"}'
                        self._sock.sendto(heartbeat_msg, self._server_addr)  # nosec
                        print(f"[UDP] Updated server address to {self._server_addr}")
                    except Exception as e:
                        print(f"[UDP] ERROR: Failed to send heartbeat to new address: {e}")
                        # If send fails, recreate socket
                        try:
                            self._sock.close()
                        except Exception:
                            pass
                        self._sock = None
                        self._server_addr = None
                        # Fall through to create new socket
                    else:
                        return True
                else:
                    print("[UDP] Already running and socket is valid")
                    return True
            except (OSError, AttributeError):
                # Socket is closed or invalid, need to recreate
                print("[UDP] Socket exists but is invalid, recreating...")
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                self._server_addr = None
                # Fall through to create new socket
        try:
            # Init crypto/decoder if available
            # Use hardcoded key from C code if server sends 'N/A' or empty
            # C code uses: "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0="
            if HAS_AES:
                # Always try to init AES - use hardcoded key if server doesn't
                # provide one
                key_b64_to_use = (
                    aes_key_b64
                    if (aes_key_b64 and aes_key_b64 not in ("", "N/A"))
                    else "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0="
                )
                try:
                    key = base64.b64decode(key_b64_to_use)
                    if len(key) == 32:  # AES-256 requires 32 bytes
                        self._aesgcm = AESGCM(key)
                        self._aes_key = key  # Store key for encryption
                        key_source = (
                            "server"
                            if (aes_key_b64 and aes_key_b64 not in ("", "N/A"))
                            else "hardcoded"
                        )
                        print(f"[UDP] AES key initialized (using {key_source} key)")
                    else:
                        print(
                            f"[UDP] WARNING: AES key length invalid: "
                            f"{len(key)} bytes (expected 32)"
                        )
                        self._aesgcm = None
                        self._aes_key = None
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

            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            host_override = os.getenv("ES_UDP_HOST_OVERRIDE", "").strip()
            host_used = host_override or host_hint
            if not host_used:
                print("[UDP] ERROR: No udp_host provided by server")
                self.stop()
                return False

            self._server_addr = (host_used, int(udp_port))
            
            # Ensure health monitoring thread is running
            self._start_health_monitor()

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

            self._hb_running.set()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

            print(f"[UDP] Listening on ephemeral port, server: {host_used}:{udp_port}")
            self._running.set()
            self._recv_thread = threading.Thread(
                target=self._receiver_loop, daemon=True
            )
            self._recv_thread.start()
            return True
        except Exception as e:
            print(f"[UDP] ERROR: Failed to start UDP listener on {udp_port}: {e}")
            self.stop()
            return False

    def _close_all_input_streams(self) -> None:
        with self._input_lock:
            for ch_idx, bundle in list(self._input_streams.items()):
                thread = bundle.get("thread")
                if thread and thread.is_alive():
                    # Thread will check _running and exit
                    pass
                pa = bundle.get("pa")
                stream = bundle.get("stream")
                if pa and stream:
                    close_stream(pa, stream)
                encoder = bundle.get("encoder")
                if encoder and HAS_OPUS:
                    try:
                        encoder.destroy()
                    except Exception:
                        pass
            self._input_streams.clear()

    def _tone_detection_worker(self, channel_id: str) -> None:
        print(f"[TONE DETECT] Worker thread started for channel {channel_id}")

        running_flag = self._tone_detect_running.get(channel_id)
        audio_queue = self._tone_detect_queues.get(channel_id)

        if not running_flag or not audio_queue:
            print(f"[TONE DETECT] ERROR: Missing queue or flag for {channel_id}")
            return

        chunk_count = 0
        process_count = 0
        last_process_time = time.time()
        PROCESS_INTERVAL = 0.25  # Process tone detection every 250ms instead of every chunk

        while running_flag.is_set():
            try:
                # Get audio buffer reference from queue (timeout 0.1s)
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunk_count += 1
                
                # Get queue size early to determine batch processing strategy
                current_time = time.time()
                time_since_last = current_time - last_process_time
                queue_size = audio_queue.qsize()
                
                if chunk_count <= 5 or chunk_count % 100 == 0:
                    print(f"[TONE DETECT DEBUG] Channel {channel_id}: Received chunk #{chunk_count} from queue (qsize={queue_size})")

                # Apply frequency filters (fast operation)
                if apply_audio_frequency_filters:
                    filters = self._frequency_filters.get(channel_id, [])
                    if filters:
                        filtered_audio = apply_audio_frequency_filters(
                            audio_chunk, filters, sample_rate=48000
                        )
                    else:
                        filtered_audio = audio_chunk
                else:
                    filtered_audio = audio_chunk

                # ALWAYS add samples to buffer (fast operation, just appending to list)
                if add_audio_samples_for_channel:
                    add_audio_samples_for_channel(channel_id, filtered_audio)
                    if chunk_count <= 5:
                        print(f"[TONE DETECT DEBUG] Channel {channel_id}: Added {len(filtered_audio)} samples to buffer")

                # Handle passthrough (in tone detection thread, not broadcasting)
                if HAS_PASSTHROUGH and global_passthrough_manager:
                    try:
                        global_passthrough_manager.cleanup_expired_sessions()
                        if global_passthrough_manager.is_active(channel_id):
                            try:
                                global_passthrough_manager.route_audio(
                                    channel_id, audio_chunk
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Handle recording (in tone detection thread, not broadcasting)
                if HAS_RECORDING and global_recording_manager:
                    try:
                        global_recording_manager.cleanup_expired_sessions()
                        if global_recording_manager.is_active(channel_id):
                            try:
                                global_recording_manager.route_audio(
                                    channel_id, audio_chunk
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

                # FIX: When queue is very full, process multiple chunks in batch before doing expensive tone detection
                # This allows us to drain the queue faster by processing chunks quickly, then doing tone detection
                chunks_to_process_before_tone_detect = 1  # Default: process one chunk at a time
                if queue_size > 80:
                    # Queue is almost full - process 5 chunks before doing tone detection
                    chunks_to_process_before_tone_detect = 5
                elif queue_size > 50:
                    # Queue is half full - process 3 chunks before doing tone detection
                    chunks_to_process_before_tone_detect = 3
                elif queue_size > 20:
                    # Queue is backing up - process 2 chunks before doing tone detection
                    chunks_to_process_before_tone_detect = 2
                
                # Only do expensive tone detection after processing N chunks OR if enough time has passed
                should_do_tone_detection = False
                adaptive_interval = PROCESS_INTERVAL
                
                if queue_size > 80:
                    # Queue is almost full (80%+) - process every N chunks to drain faster
                    should_do_tone_detection = (chunk_count % chunks_to_process_before_tone_detect == 0)
                    adaptive_interval = 0.05  # Process every 50ms
                elif queue_size > 50:
                    # Queue is half full - process every N chunks and more frequently
                    should_do_tone_detection = (chunk_count % chunks_to_process_before_tone_detect == 0)
                    adaptive_interval = 0.1  # Process every 100ms
                elif queue_size > 20:
                    # Queue is starting to back up - process every N chunks more often
                    should_do_tone_detection = (chunk_count % chunks_to_process_before_tone_detect == 0)
                    adaptive_interval = 0.15  # Process every 150ms
                elif queue_size > 10:
                    # Queue is getting backed up - process every N chunks sooner
                    should_do_tone_detection = (chunk_count % chunks_to_process_before_tone_detect == 0)
                    adaptive_interval = 0.2  # Process every 200ms
                else:
                    # Normal operation - use standard interval with lower threshold
                    adaptive_interval = PROCESS_INTERVAL
                    should_do_tone_detection = (time_since_last >= adaptive_interval) or (queue_size > 5)
                
                if should_do_tone_detection:
                    process_count += 1
                    last_process_time = current_time
                    
                    if process_count <= 5 or process_count % 20 == 0:
                        print(f"[TONE DETECT DEBUG] Channel {channel_id}: Processing tone detection (process_count={process_count}, chunks_processed={chunk_count}, qsize={queue_size}, batch_size={chunks_to_process_before_tone_detect})")
                    
                    try:
                        # Import detector access from tone_detection module
                        from tone_detection import _channel_detectors, _detectors_mutex
                        
                        with _detectors_mutex:
                            detector = _channel_detectors.get(channel_id)
                            if detector:
                                # Call process_audio() directly without adding samples again
                                detected_tone = detector.process_audio()
                                if detected_tone:
                                    print(f"\n[TONE DETECT] *** TONE DETECTED ON {channel_id} ***")
                                    print(
                                        f"[TONE DETECT] Tone ID: {detected_tone.get('tone_id', 'unknown')}\n"
                                    )
                    except Exception as tone_err:
                        print(f"[TONE DETECT] ERROR in tone detection for {channel_id}: {tone_err}")
                        import traceback
                        traceback.print_exc()

                audio_queue.task_done()

            except Exception as e:
                print(f"[TONE DETECT] ERROR in worker for {channel_id}: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.5)

        print(f"[TONE DETECT] Worker thread stopped for channel {channel_id}")

    def _audio_input_worker(self, channel_index: int, channel_id: str) -> None:
        print(
            f"[AUDIO TX] Input worker started for channel {channel_id} (index {channel_index})"
        )

        bundle = self._input_streams.get(channel_index)
        if not bundle:
            print(
                f"[AUDIO TX] ERROR: No input stream bundle for channel {channel_index}"
            )
            return

        stream = bundle.get("stream")
        encoder = bundle.get("encoder")
        input_buffer = bundle.get("buffer")
        buffer_pos_key = "buffer_pos"

        if not stream or not encoder or input_buffer is None:
            print(
                f"[AUDIO TX] ERROR: Missing stream/encoder/buffer for channel {channel_index}"
            )
            return

        send_count = 0
        read_count = 0

        if send_count == 0:
            print(
                f"[AUDIO TX] Worker started for channel {channel_id} (index {channel_index})"
            )
            print(
                f"[AUDIO TX] Transmission flag: {self._transmitting.get(channel_index, False)}"
            )
            print(f"[AUDIO TX] Running flag: {self._running.is_set()}")

        loop_exit_reason = "unknown"
        while self._running.is_set() and self._transmitting.get(channel_index, False):
            try:
                from gpio_monitor import GPIO_PINS, gpio_states

                gpio_keys = list(GPIO_PINS.keys())
                if channel_index < len(gpio_keys):
                    gpio_num = gpio_keys[channel_index]
                    gpio_state = gpio_states.get(gpio_num, -1)
                    if gpio_state != 0:
                        if send_count == 0:
                            print(
                                f"[AUDIO TX] Channel {channel_id}: GPIO {gpio_num} not active (state={gpio_state}), waiting..."
                            )
                        time.sleep(0.1)
                        continue
            except Exception as e:
                if send_count % 100 == 0:
                    print(f"[AUDIO TX] WARNING: GPIO check failed: {e}")

            try:
                data = stream.read(1024, exception_on_overflow=False)
                if len(data) == 0:
                    if send_count <= 10:
                        print(f"[AUDIO TX DEBUG] Channel {channel_id}: stream.read() returned 0 bytes")
                    time.sleep(0.01)
                    continue

                read_count += 1
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                if send_count <= 10 or send_count % 1000 == 0:
                    print(f"[AUDIO TX DEBUG] Channel {channel_id}: Read {len(data)} bytes from stream (read_count={read_count})")

                for sample in audio_data:
                    input_buffer[bundle[buffer_pos_key]] = sample
                    bundle[buffer_pos_key] += 1

                    if bundle[buffer_pos_key] >= 1920:
                        audio_chunk = input_buffer[:1920].copy()
                        passthrough_active = False
                        pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(
                            np.int16
                        )
                        pcm_bytes = pcm.tobytes()

                        try:
                            opus_data = encoder.encode(pcm_bytes, 1920)

                            if opus_data and len(opus_data) > 0:
                                if self._aes_key and HAS_AES:
                                    try:
                                        iv = os.urandom(12)
                                        aesgcm_enc = AESGCM(self._aes_key)
                                        encrypted = aesgcm_enc.encrypt(
                                            iv, opus_data, None
                                        )

                                        encrypted_with_iv = iv + encrypted

                                        b64_data = base64.b64encode(
                                            encrypted_with_iv
                                        ).decode("utf-8")

                                        if self._sock and self._server_addr:
                                            try:
                                                # Verify socket is still valid before sending
                                                self._sock.getsockname()
                                            except (OSError, AttributeError):
                                                # Socket is closed/invalid
                                                if send_count <= 10 or send_count % 1000 == 0:
                                                    print(f"[AUDIO TX ERROR] Channel {channel_id}: Socket is invalid, cannot send (send_count={send_count})")
                                                # Skip this packet
                                                continue
                                            
                                            msg = json.dumps(
                                                {
                                                    "channel_id": channel_id,
                                                    "type": "audio",
                                                    "data": b64_data,
                                                }
                                            )
                                            try:
                                                msg_bytes = msg.encode("utf-8")
                                                bytes_sent = self._sock.sendto(
                                                    msg_bytes, self._server_addr
                                                )  # nosec

                                                send_count += 1
                                                # Update last packet time for health monitoring
                                                self._last_packet_time[channel_index] = time.time()
                                                
                                                # Log more frequently to help debug audio transmission issues
                                                if (
                                                    send_count <= 10
                                                    or send_count % 100 == 0  # Log every 100 packets for better visibility
                                                ):
                                                    print(
                                                        f"[AUDIO TX] Channel {channel_id}: Sent audio packet #{send_count} "
                                                        f"({bytes_sent}/{len(msg_bytes)} bytes) to {self._server_addr}"
                                                    )
                                                # Verify bytes sent matches message length
                                                if bytes_sent != len(msg_bytes):
                                                    print(
                                                        f"[AUDIO TX WARNING] Channel {channel_id}: Partial send - "
                                                        f"sent {bytes_sent} of {len(msg_bytes)} bytes"
                                                    )
                                            except (OSError, AttributeError) as sock_err:
                                                # Socket error - log and continue
                                                if send_count <= 10 or send_count % 100 == 0:
                                                    print(f"[AUDIO TX ERROR] Channel {channel_id}: Socket send failed: {sock_err} (send_count={send_count})")
                                                    print(f"[AUDIO TX ERROR] Socket: {self._sock}, Server addr: {self._server_addr}")
                                                # Don't increment send_count on failure
                                                continue
                                    except Exception as e:
                                        if send_count <= 10:
                                            print(
                                                f"[AUDIO TX ERROR] Channel {channel_id}: Encryption/send failed: {e}"
                                            )
                                else:
                                    if send_count <= 10:
                                        print(
                                            f"[AUDIO TX ERROR] Channel {channel_id}: No AES key available"
                                        )
                        except Exception as e:
                            if send_count <= 10:
                                print(
                                    f"[AUDIO TX ERROR] Channel {channel_id}: Opus encode failed: {e}"
                                )

                        if HAS_TONE_DETECT and self._tone_detect_enabled.get(
                            channel_id, False
                        ):
                            tone_queue = self._tone_detect_queues.get(channel_id)
                            detect_thread = self._tone_detect_threads.get(channel_id)
                            running_flag = self._tone_detect_running.get(channel_id)
                            
                            # Health check: verify tone detection thread is alive and restart if needed
                            # Check more frequently (every 1000 packets instead of 10000) to catch issues sooner
                            if send_count > 0 and send_count % 1000 == 0:
                                thread_needs_restart = False
                                if not detect_thread or not detect_thread.is_alive():
                                    print(f"[TONE DETECT] WARNING: Tone detection thread is dead for {channel_id}, attempting restart...")
                                    thread_needs_restart = True
                                elif running_flag and not running_flag.is_set():
                                    print(f"[TONE DETECT] WARNING: Tone detection flag is cleared for {channel_id}, attempting restart...")
                                    thread_needs_restart = True
                                # Also check if queue is consistently full (might indicate thread is stuck)
                                if tone_queue and tone_queue.full():
                                    if send_count % 5000 == 0:  # Log less frequently for queue full
                                        print(f"[TONE DETECT] WARNING: Tone queue is consistently full for {channel_id}, thread may be stuck")
                                
                                if thread_needs_restart:
                                    try:
                                        # Ensure queue exists
                                        if channel_id not in self._tone_detect_queues:
                                            self._tone_detect_queues[channel_id] = queue.Queue(maxsize=100)  # Increased from 50 to handle bursts
                                        
                                        # Create new running flag
                                        running_flag = threading.Event()
                                        running_flag.set()
                                        self._tone_detect_running[channel_id] = running_flag
                                        
                                        # Start new thread
                                        detect_thread = threading.Thread(
                                            target=self._tone_detection_worker,
                                            args=(channel_id,),
                                            daemon=True,
                                            name=f"ToneDetect-{channel_id}",
                                        )
                                        self._tone_detect_threads[channel_id] = detect_thread
                                        detect_thread.start()
                                        print(f"[TONE DETECT] Auto-restarted worker thread for {channel_id}")
                                    except Exception as e:
                                        print(f"[TONE DETECT] ERROR: Failed to auto-restart tone detection thread: {e}")
                                        import traceback
                                        traceback.print_exc()
                            
                            if tone_queue:
                                try:
                                    tone_queue.put_nowait(audio_chunk)
                                    if send_count <= 10 or send_count % 1000 == 0:
                                        print(f"[AUDIO TX DEBUG] Channel {channel_id}: Put audio chunk in tone queue (send_count={send_count}, qsize={tone_queue.qsize()})")
                                except queue.Full:
                                    if send_count <= 10 or send_count % 100 == 0:
                                        print(f"[AUDIO TX DEBUG] Channel {channel_id}: Tone queue FULL, skipping (send_count={send_count})")
                                    pass
                            else:
                                # Queue missing - this shouldn't happen but log it
                                if send_count <= 10 or send_count % 10000 == 0:
                                    print(f"[TONE DETECT] WARNING: Tone queue missing for {channel_id} (send_count={send_count})")

                        bundle[buffer_pos_key] = 0
            except Exception as e:
                if not self._running.is_set():
                    loop_exit_reason = "_running is False after exception"
                    break
                if send_count <= 10 or send_count % 100 == 0:
                    print(f"[AUDIO TX] Input error for {channel_id}: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

        # Check why loop exited
        if not self._running.is_set():
            loop_exit_reason = "_running is False"
        elif not self._transmitting.get(channel_index, False):
            loop_exit_reason = "_transmitting is False"
        
        print(f"[AUDIO TX] Input worker stopped for channel {channel_id}")
        print(f"[AUDIO TX] Loop exit reason: {loop_exit_reason}")
        print(f"[AUDIO TX] Final state - running: {self._running.is_set()}, transmitting: {self._transmitting.get(channel_index, False)}, send_count: {send_count}")

    def start_transmission_for_channel(self, channel_index: int) -> bool:
        if channel_index < 0 or channel_index >= len(self._channel_ids):
            print(f"[AUDIO TX] ERROR: Invalid channel index {channel_index}")
            return False

        channel_id = self._channel_ids[channel_index]

        with self._input_lock:
            if channel_index in self._input_streams:
                print(f"[AUDIO TX] Channel {channel_id} already transmitting")
                
                # Check if input worker thread is still alive and actually sending packets
                input_thread = self._input_worker_threads.get(channel_index)
                current_time = time.time()
                last_packet_time = self._last_packet_time.get(channel_index, 0)
                time_since_last_packet = current_time - last_packet_time
                
                thread_dead = not input_thread or not input_thread.is_alive()
                # FIX: If thread exists but last_packet_time is 0, it means thread never sent packets
                # OR if thread exists and no packets for >10s, it's stalled
                thread_stalled = False
                if input_thread and input_thread.is_alive():
                    if last_packet_time == 0:
                        # Thread exists but never sent a packet - likely stuck or just started
                        # If it's been running for more than 5 seconds without sending, treat as stalled
                        thread_stalled = True  # Treat as stalled if no packets ever sent
                    elif time_since_last_packet > 10.0:
                        thread_stalled = True
                elif input_thread is None:
                    # No thread reference at all - force restart
                    thread_dead = True
                
                if thread_dead:
                    if input_thread is None:
                        print(f"[AUDIO TX] WARNING: No thread reference for {channel_id}, creating new thread...")
                    else:
                        print(f"[AUDIO TX] WARNING: Input worker thread is DEAD for {channel_id}, restarting...")
                elif thread_stalled:
                    if last_packet_time == 0:
                        print(f"[AUDIO TX] WARNING: Input worker thread for {channel_id} exists but never sent packets, restarting...")
                    else:
                        print(f"[AUDIO TX] WARNING: Input worker thread is STALLED for {channel_id} (no packets for {time_since_last_packet:.1f}s), restarting...")
                
                if thread_dead or thread_stalled:
                    try:
                        bundle = self._input_streams.get(channel_index)
                        if bundle:
                            # FIX: Properly clean up old thread and stream before restarting
                            old_thread = bundle.get("thread")
                            old_stream = bundle.get("stream")
                            old_pa = bundle.get("pa")
                            old_encoder = bundle.get("encoder")
                            
                            # Wait for old thread to finish (with timeout)
                            if old_thread and old_thread.is_alive():
                                print(f"[AUDIO TX] Waiting for old thread to finish for {channel_id}...")
                                old_thread.join(timeout=1.0)
                                if old_thread.is_alive():
                                    print(f"[AUDIO TX] WARNING: Old thread for {channel_id} did not stop within timeout")
                            
                            # Remove old thread reference
                            if channel_index in self._input_worker_threads:
                                del self._input_worker_threads[channel_index]
                            
                            # Close and cleanup old stream/encoder
                            if old_stream:
                                try:
                                    if hasattr(old_stream, 'is_active') and old_stream.is_active():
                                        old_stream.stop_stream()
                                    old_stream.close()
                                    print(f"[AUDIO TX] Closed old stream for {channel_id}")
                                except Exception as e:
                                    print(f"[AUDIO TX] WARNING: Error closing old stream for {channel_id}: {e}")
                            
                            if old_encoder:
                                try:
                                    old_encoder.destroy()
                                    print(f"[AUDIO TX] Destroyed old encoder for {channel_id}")
                                except Exception as e:
                                    print(f"[AUDIO TX] WARNING: Error destroying old encoder for {channel_id}: {e}")
                            
                            # Recreate the stream/encoder/bundle from scratch
                            device_index = select_input_device_for_channel(channel_index)
                            if device_index is None:
                                print(f"[AUDIO TX] ERROR: No input device available for channel {channel_index}")
                                return False
                            
                            pa, stream = open_input_stream(device_index, frames_per_buffer=1024)
                            if pa is None or stream is None:
                                print(f"[AUDIO TX] ERROR: Failed to open input stream for channel {channel_index}")
                                return False
                            
                            encoder = None
                            if HAS_OPUS:
                                try:
                                    encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_VOIP)
                                    encoder.bitrate = 64000
                                    encoder.vbr = True
                                except Exception as e:
                                    print(f"[AUDIO TX] ERROR: Opus encoder init failed for channel {channel_index}: {e}")
                                    close_stream(pa, stream)
                                    return False
                            else:
                                print("[AUDIO TX] ERROR: Opus library not available")
                                close_stream(pa, stream)
                                return False
                            
                            # Start input stream
                            try:
                                stream.start_stream()
                            except Exception as e:
                                print(f"[AUDIO TX] ERROR: Failed to start input stream: {e}")
                                close_stream(pa, stream)
                                if encoder:
                                    try:
                                        encoder.destroy()
                                    except Exception:
                                        pass
                                return False
                            
                            input_buffer = np.zeros(1920, dtype=np.float32)
                            
                            # Update bundle with new stream/encoder
                            bundle["pa"] = pa
                            bundle["stream"] = stream
                            bundle["encoder"] = encoder
                            bundle["buffer"] = input_buffer
                            bundle["buffer_pos"] = 0
                            
                            # Start new thread
                            thread = threading.Thread(
                                target=self._audio_input_worker,
                                args=(channel_index, channel_id),
                                daemon=True,
                                name=f"AudioInput-{channel_id}",
                            )
                            bundle["thread"] = thread
                            self._input_worker_threads[channel_index] = thread
                            self._transmitting[channel_index] = True
                            self._last_packet_time[channel_index] = current_time  # Reset timer
                            thread.start()
                            print(f"[AUDIO TX] ✓ Restarted input worker thread for channel {channel_id}")
                            
                            # Verify thread started successfully
                            time.sleep(0.1)  # Give thread a moment to start
                            if not thread.is_alive():
                                print(f"[AUDIO TX] ERROR: Restarted thread for {channel_id} died immediately!")
                                return False
                        else:
                            print(f"[AUDIO TX] ERROR: No bundle found for channel {channel_id}, cannot restart")
                    except Exception as e:
                        print(f"[AUDIO TX] ERROR: Failed to restart input worker thread: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    # Thread is alive and sending packets, just ensure transmitting flag is set
                    self._transmitting[channel_index] = True
                
                # Verify and restart tone detection thread if needed
                if HAS_TONE_DETECT and self._tone_detect_enabled.get(channel_id, False):
                    detect_thread = self._tone_detect_threads.get(channel_id)
                    running_flag = self._tone_detect_running.get(channel_id)
                    
                    # Check if thread is missing or dead
                    if not detect_thread or not detect_thread.is_alive():
                        print(f"[TONE DETECT] Tone detection thread missing or dead for {channel_id}, restarting...")
                        try:
                            # Ensure queue exists
                            if channel_id not in self._tone_detect_queues:
                                self._tone_detect_queues[channel_id] = queue.Queue(maxsize=100)  # Increased from 50 to handle bursts
                            
                            # Create new running flag if missing
                            if not running_flag:
                                running_flag = threading.Event()
                                running_flag.set()
                                self._tone_detect_running[channel_id] = running_flag
                            elif not running_flag.is_set():
                                # Restart if flag was cleared
                                running_flag.set()
                            
                            # Start new thread
                            detect_thread = threading.Thread(
                                target=self._tone_detection_worker,
                                args=(channel_id,),
                                daemon=True,
                                name=f"ToneDetect-{channel_id}",
                            )
                            self._tone_detect_threads[channel_id] = detect_thread
                            detect_thread.start()
                            print(f"[TONE DETECT] Restarted worker thread for {channel_id}")
                        except Exception as e:
                            print(f"[TONE DETECT] WARNING: Failed to restart tone detection thread: {e}")
                            import traceback
                            traceback.print_exc()
                    elif running_flag and not running_flag.is_set():
                        # Thread exists but flag is cleared, restart it
                        print(f"[TONE DETECT] Tone detection flag cleared for {channel_id}, restarting thread...")
                        try:
                            running_flag.set()
                            # Wait a bit and check if thread is still alive
                            time.sleep(0.1)
                            if not detect_thread.is_alive():
                                # Thread died, restart it
                                self._tone_detect_queues[channel_id] = queue.Queue(maxsize=100)  # Increased from 50 to handle bursts
                                running_flag = threading.Event()
                                running_flag.set()
                                self._tone_detect_running[channel_id] = running_flag
                                detect_thread = threading.Thread(
                                    target=self._tone_detection_worker,
                                    args=(channel_id,),
                                    daemon=True,
                                    name=f"ToneDetect-{channel_id}",
                                )
                                self._tone_detect_threads[channel_id] = detect_thread
                                detect_thread.start()
                                print(f"[TONE DETECT] Restarted worker thread for {channel_id}")
                        except Exception as e:
                            print(f"[TONE DETECT] WARNING: Failed to restart tone detection: {e}")
                
                return True

            device_index = select_input_device_for_channel(channel_index)
            if device_index is None:
                print(
                    f"[AUDIO TX] ERROR: No input device available for channel {channel_index}"
                )
                return False

            pa, stream = open_input_stream(device_index, frames_per_buffer=1024)
            if pa is None or stream is None:
                print(
                    f"[AUDIO TX] ERROR: Failed to open input stream for channel {channel_index}"
                )
                return False

            encoder = None
            if HAS_OPUS:
                try:
                    encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_VOIP)
                    encoder.bitrate = 64000
                    encoder.vbr = True
                except Exception as e:
                    print(
                        f"[AUDIO TX] ERROR: Opus encoder init failed for channel {channel_index}: {e}"
                    )
                    close_stream(pa, stream)
                    return False
            else:
                print("[AUDIO TX] ERROR: Opus library not available")
                close_stream(pa, stream)
                return False

            # Start input stream
            try:
                stream.start_stream()
            except Exception as e:
                print(f"[AUDIO TX] ERROR: Failed to start input stream: {e}")
                close_stream(pa, stream)
                if encoder:
                    try:
                        encoder.destroy()
                    except Exception:
                        pass
                return False

            input_buffer = np.zeros(1920, dtype=np.float32)

            # Store bundle
            bundle = {
                "pa": pa,
                "stream": stream,
                "encoder": encoder,
                "buffer": input_buffer,
                "buffer_pos": 0,
            }
            self._input_streams[channel_index] = bundle

            # Start input worker thread
            thread = threading.Thread(
                target=self._audio_input_worker,
                args=(channel_index, channel_id),
                daemon=True,
                name=f"AudioInput-{channel_id}",
            )
            bundle["thread"] = thread
            self._input_worker_threads[channel_index] = thread
            self._transmitting[channel_index] = True
            self._last_packet_time[channel_index] = time.time()  # Initialize last packet time
            thread.start()
            print(f"[AUDIO TX] Started input worker thread for channel {channel_id} (index {channel_index})")

            # Start tone detection thread
            if HAS_TONE_DETECT and self._tone_detect_enabled.get(channel_id, False):
                try:
                    print("[TONE DETECT] This is the final step to start tone detectio thread")
                    self._tone_detect_queues[channel_id] = queue.Queue(maxsize=100)  # Increased from 50 to handle bursts

                    # Start tone detection worker thread
                    running_flag = threading.Event()
                    running_flag.set()
                    self._tone_detect_running[channel_id] = running_flag

                    detect_thread = threading.Thread(
                        target=self._tone_detection_worker,
                        args=(channel_id,),
                        daemon=True,
                        name=f"ToneDetect-{channel_id}",
                    )
                    self._tone_detect_threads[channel_id] = detect_thread
                    detect_thread.start()

                    print(f"[TONE DETECT] Started worker thread for {channel_id}")
                except Exception as e:
                    print(f"[TONE DETECT] WARNING: Failed to start thread: {e}")

            print(
                f"[AUDIO TX] Started transmission for channel {channel_id} (index {channel_index}, device {device_index})"
            )
            return True

    def stop_transmission_for_channel(self, channel_index: int) -> None:
        """Stop audio transmission for a channel"""
        if channel_index < 0 or channel_index >= len(self._channel_ids):
            return

        channel_id = self._channel_ids[channel_index]
        print(f"[AUDIO TX DEBUG] stop_transmission_for_channel called for {channel_id} (index {channel_index})")
        import traceback
        traceback.print_stack()

        # Stop tone detection thread
        if HAS_TONE_DETECT and channel_id in self._tone_detect_running:
            try:
                running_flag = self._tone_detect_running[channel_id]
                running_flag.clear()  # Signal thread to stop

                detect_thread = self._tone_detect_threads.get(channel_id)
                if detect_thread and detect_thread.is_alive():
                    detect_thread.join(timeout=2.0)

                # Cleanup resources
                self._tone_detect_running.pop(channel_id, None)
                self._tone_detect_threads.pop(channel_id, None)
                self._tone_detect_queues.pop(channel_id, None)

                print(f"[TONE DETECT] Stopped thread for {channel_id}")
            except Exception as e:
                print(f"[TONE DETECT] WARNING: Error stopping thread: {e}")

        with self._input_lock:
            if channel_index not in self._input_streams:
                return
            self._transmitting[channel_index] = False
            bundle = self._input_streams.get(channel_index)
            if bundle:
                thread = bundle.get("thread")
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)
                pa = bundle.get("pa")
                stream = bundle.get("stream")
                if pa and stream:
                    close_stream(pa, stream)
                encoder = bundle.get("encoder")
                if encoder and HAS_OPUS:
                    try:
                        encoder.destroy()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                del self._input_streams[channel_index]
            if channel_index in self._transmitting:
                del self._transmitting[channel_index]
            # Remove thread reference
            if channel_index in self._input_worker_threads:
                del self._input_worker_threads[channel_index]

    def stop(self) -> None:
        try:
            self._running.clear()

            # Stop all tone detection threads
            if HAS_TONE_DETECT:
                for channel_id, running_flag in list(self._tone_detect_running.items()):
                    try:
                        running_flag.clear()
                    except Exception:
                        pass

                for channel_id, detect_thread in list(
                    self._tone_detect_threads.items()
                ):
                    try:
                        if detect_thread and detect_thread.is_alive():
                            detect_thread.join(timeout=1.0)
                    except Exception:
                        pass

                self._tone_detect_running.clear()
                self._tone_detect_threads.clear()
                self._tone_detect_queues.clear()

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
            self._close_all_input_streams()
            print("[UDP] Stopped")


# Singleton instance used by main via websocket callback
global_udp_player = UDPPlayer()
