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
        if self._sock is not None:
            print("[UDP] Already running")
            return True
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
                
                if chunk_count <= 5 or chunk_count % 100 == 0:
                    print(f"[TONE DETECT DEBUG] Channel {channel_id}: Received chunk #{chunk_count} from queue")

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

                # Process tone detection periodically (expensive FFT operations)
                # Note: We already added samples above, so we call detector.process_audio() directly
                # to avoid adding samples twice
                current_time = time.time()
                if (current_time - last_process_time) >= PROCESS_INTERVAL:
                    process_count += 1
                    last_process_time = current_time
                    
                    if process_count <= 5 or process_count % 20 == 0:
                        print(f"[TONE DETECT DEBUG] Channel {channel_id}: Processing tone detection (process_count={process_count}, chunks_processed={chunk_count})")
                    
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
                                            msg = json.dumps(
                                                {
                                                    "channel_id": channel_id,
                                                    "type": "audio",
                                                    "data": b64_data,
                                                }
                                            )
                                            self._sock.sendto(
                                                msg.encode("utf-8"), self._server_addr
                                            )  # nosec

                                            send_count += 1
                                            if (
                                                send_count <= 5
                                                or send_count % 10000 == 0
                                            ):
                                                print(
                                                    f"[AUDIO TX] Channel {channel_id}: Sent audio packet #{send_count} "
                                                    f"({len(msg)} bytes) to {self._server_addr}"
                                                )
                                        else:
                                            if send_count <= 10:
                                                print(
                                                    f"[AUDIO TX ERROR] Channel {channel_id}: "
                                                    f"Cannot send - sock={self._sock is not None}, "
                                                    f"addr={self._server_addr}"
                                                )
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
                            if tone_queue:
                                try:
                                    tone_queue.put_nowait(audio_chunk)
                                    if send_count <= 10 or send_count % 1000 == 0:
                                        print(f"[AUDIO TX DEBUG] Channel {channel_id}: Put audio chunk in tone queue (send_count={send_count}, qsize={tone_queue.qsize()})")
                                except queue.Full:
                                    if send_count <= 10 or send_count % 100 == 0:
                                        print(f"[AUDIO TX DEBUG] Channel {channel_id}: Tone queue FULL, skipping (send_count={send_count})")
                                    pass

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
                self._transmitting[channel_index] = True
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
            )
            bundle["thread"] = thread
            self._transmitting[channel_index] = True
            thread.start()

            # Start tone detection thread
            if HAS_TONE_DETECT and self._tone_detect_enabled.get(channel_id, False):
                try:
                    print("[TONE DETECT] This is the final step to start tone detectio thread")
                    self._tone_detect_queues[channel_id] = queue.Queue(maxsize=50)  # Increased from 10 to prevent queue full errors

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
