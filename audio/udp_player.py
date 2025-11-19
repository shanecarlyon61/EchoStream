import socket
import threading
import base64
import json
import os
import time
import errno
import numpy as np
from typing import Dict, Optional, List, Tuple, Any

from audio.devices import (
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
    from core.config import load_config, get_frequency_filters, get_tone_detect_config, get_tone_definitions, get_new_tone_config, get_passthrough_config, get_channel_ids
    from audio.filter import apply_audio_frequency_filters
    from processing.tone_detection import init_channel_detector, process_audio_for_channel
    from processing.passthrough import global_passthrough_manager
    from processing.recording import global_recording_manager
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
    global_passthrough_manager = None
    global_recording_manager = None
    HAS_RECORDING = False


class UDPPlayer:
    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._streams: Dict[int, Dict[str, object]] = {}  # channel_index -> { 'pa': pa, 'stream': stream }
        self._input_streams: Dict[int, Dict[str, object]] = {}  # channel_index -> { 'pa': pa, 'stream': stream, 'encoder': encoder, 'buffer': buffer, 'buffer_pos': int, 'thread': thread }
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
        self._transmitting: Dict[int, bool] = {}  # channel_index -> transmitting state
        # If ES_UDP_HEARTBEAT_LOG=0, completely suppress heartbeat logs
        self._suppress_hb_log = os.getenv("ES_UDP_HEARTBEAT_LOG", "1") == "0"
        self._frequency_filters: Dict[str, List[Dict[str, Any]]] = {}
        self._tone_detect_enabled: Dict[str, bool] = {}
        self._config_cache: Optional[Dict[str, Any]] = None

    def _ensure_stream_for_channel(self, channel_index: int) -> None:
        with self._lock:
            if channel_index in self._streams:
                bundle = self._streams[channel_index]
                stream = bundle.get("stream")
                # Ensure stream is started
                if stream and hasattr(stream, 'is_active') and not stream.is_active():
                    try:
                        stream.start_stream()  # type: ignore[attr-defined]
                        print(f"[AUDIO] Started output stream for channel index {channel_index}")
                    except Exception as e:
                        print(f"[AUDIO] WARNING: Failed to start stream for channel {channel_index}: {e}")
                return
            device_index = select_output_device_for_channel(channel_index)
            if device_index is None:
                print(f"[AUDIO] WARNING: No device available for channel index {channel_index}")
                return
            pa, stream = open_output_stream(device_index)
            if pa is None or stream is None:
                print(f"[AUDIO] ERROR: Failed to open output stream for channel index {channel_index}")
                return
            # Start the stream immediately
            try:
                stream.start_stream()  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[AUDIO] WARNING: Failed to start stream: {e}")
            self._streams[channel_index] = {"pa": pa, "stream": stream}  # keep PA for lifecycle consistency
            print(f"[AUDIO] Output stream ready and started on device {device_index} for channel index {channel_index}")

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
                        print(f"[UDP] Loaded {len(filters)} frequency filter(s) for tone detection on channel {channel_id}")
                    else:
                        self._frequency_filters[channel_id] = []
                    
                    if HAS_TONE_DETECT and get_tone_definitions and init_channel_detector:
                        tone_defs = get_tone_definitions(self._config_cache, channel_id)
                        new_tone_cfg = None
                        if get_new_tone_config:
                            new_tone_cfg = get_new_tone_config(self._config_cache, channel_id)
                        passthrough_cfg = None
                        if get_passthrough_config:
                            passthrough_cfg = get_passthrough_config(self._config_cache, channel_id)
                        if tone_defs or (new_tone_cfg and new_tone_cfg.get("detect_new_tones", False)):
                            init_channel_detector(channel_id, tone_defs, new_tone_cfg, passthrough_cfg)
                            print(f"[UDP] Initialized tone detection for channel {channel_id} "
                                  f"with {len(tone_defs)} tone definition(s)")
                        else:
                            print(f"[UDP] WARNING: Tone detection enabled for channel {channel_id} "
                                  f"but no tone definitions found in config.json")
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
            pcm = self._opus_decoder.decode(data, frame_size=1920, decode_fec=0)  # type: ignore[attr-defined]
            # Convert to bytes directly (opuslib returns bytes)
            return pcm if isinstance(pcm, bytes) else bytes(pcm)
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
                        # Ensure stream is started
                        if not stream.is_active():  # type: ignore[attr-defined]
                            stream.start_stream()  # type: ignore[attr-defined]
                        # Write PCM data in chunks to avoid buffer issues
                        # 1920 samples = 3840 bytes (int16), write in 1024-sample chunks
                        pcm_len = len(pcm)
                        bytes_per_sample = 2
                        samples_per_chunk = 1024
                        bytes_per_chunk = samples_per_chunk * bytes_per_sample
                        written = 0
                        while written < pcm_len:
                            chunk_size = min(bytes_per_chunk, pcm_len - written)
                            chunk = pcm[written:written + chunk_size]
                            try:
                                stream.write(chunk)  # type: ignore[attr-defined]
                                written += chunk_size
                            except Exception as write_err:
                                if self._receive_count <= 10:
                                    print(f"[UDP] ERROR: stream.write() failed: {write_err}")
                                break
                        if self._receive_count <= 10:
                            print(f"[UDP] Wrote {written} bytes PCM to channel {target_index} (channel_id: {ch_id})")
                    except Exception as e:
                        print(f"[UDP] WARNING: write failed for channel index {target_index}: {e}")
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
                        self._aes_key = key  # Store key for encryption
                        key_source = 'server' if (aes_key_b64 and aes_key_b64 not in ("", "N/A")) else 'hardcoded'
                        print(f"[UDP] AES key initialized (using {key_source} key)")
                    else:
                        print(f"[UDP] WARNING: AES key length invalid: {len(key)} bytes (expected 32)")
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
                        encoder.destroy()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            self._input_streams.clear()

    def _audio_input_worker(self, channel_index: int, channel_id: str) -> None:
        """Audio input worker thread - captures audio and sends it via UDP"""
        print(f"[AUDIO TX] Input worker started for channel {channel_id} (index {channel_index})")
        
        bundle = self._input_streams.get(channel_index)
        if not bundle:
            print(f"[AUDIO TX] ERROR: No input stream bundle for channel {channel_index}")
            return
        
        stream = bundle.get("stream")
        encoder = bundle.get("encoder")
        input_buffer = bundle.get("buffer")
        buffer_pos_key = "buffer_pos"
        
        if not stream or not encoder or input_buffer is None:
            print(f"[AUDIO TX] ERROR: Missing stream/encoder/buffer for channel {channel_index}")
            return
        
        send_count = 0
        read_count = 0
        
        if send_count == 0:
            print(f"[AUDIO TX] Worker started for channel {channel_id} (index {channel_index})")
            print(f"[AUDIO TX] Transmission flag: {self._transmitting.get(channel_index, False)}")
            print(f"[AUDIO TX] Running flag: {self._running.is_set()}")
        
        while self._running.is_set() and self._transmitting.get(channel_index, False):
            # Check GPIO state - only send when GPIO is active (value 0)
            try:
                from hardware.gpio import GPIO_PINS, gpio_states
                gpio_keys = list(GPIO_PINS.keys())
                if channel_index < len(gpio_keys):
                    gpio_num = gpio_keys[channel_index]
                    gpio_state = gpio_states.get(gpio_num, -1)
                    if gpio_state != 0:  # GPIO not active
                        if send_count == 0:
                            print(f"[AUDIO TX] Channel {channel_id}: GPIO {gpio_num} not active (state={gpio_state}), waiting...")
                        time.sleep(0.1)
                        continue
            except Exception as e:
                # If GPIO check fails, continue anyway (for testing)
                if send_count % 100 == 0:
                    print(f"[AUDIO TX] WARNING: GPIO check failed: {e}")
            
            try:
                # Read audio data (1024 frames like C code)
                data = stream.read(1024, exception_on_overflow=False)  # type: ignore[attr-defined]
                if len(data) == 0:
                    time.sleep(0.01)
                    continue
                
                read_count += 1
                # Convert bytes to float32 array
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Accumulate samples for 1920 samples (40ms at 48kHz)
                for sample in audio_data:
                    input_buffer[bundle[buffer_pos_key]] = sample
                    bundle[buffer_pos_key] += 1
                    
                    if bundle[buffer_pos_key] >= 1920:
                        audio_chunk = input_buffer[:1920].copy()
                        
                        if (HAS_TONE_DETECT and
                            self._tone_detect_enabled.get(channel_id, False) and
                            process_audio_for_channel and
                            apply_audio_frequency_filters):
                            try:
                                filters = self._frequency_filters.get(channel_id, [])
                                if filters:
                                    filtered_audio = apply_audio_frequency_filters(
                                        audio_chunk, filters, sample_rate=48000
                                    )
                                else:
                                    filtered_audio = audio_chunk
                                
                                detected_tone = process_audio_for_channel(
                                    channel_id, filtered_audio
                                )
                                if detected_tone:
                                    print(f"\n[UDP] *** TONE SEQUENCE DETECTED ON CHANNEL {channel_id} ***")
                                    print(f"[UDP] Tone ID: {detected_tone.get('tone_id', 'unknown')}\n")
                            except Exception as e:
                                if send_count <= 10:
                                    print(f"[AUDIO TX] WARNING: Tone detection "
                                          f"failed: {e}")
                        
                        passthrough_active = False
                        if HAS_PASSTHROUGH and global_passthrough_manager:
                            try:
                                global_passthrough_manager.cleanup_expired_sessions()
                                if global_passthrough_manager.is_active(channel_id):
                                    passthrough_active = True
                                    try:
                                        global_passthrough_manager.route_audio(channel_id, audio_chunk)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        
                        if HAS_RECORDING and global_recording_manager:
                            try:
                                global_recording_manager.cleanup_expired_sessions()
                                if global_recording_manager.is_active(channel_id):
                                    try:
                                        global_recording_manager.route_audio(channel_id, audio_chunk)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        
                        if send_count == 0 and passthrough_active:
                            print(f"[AUDIO TX] Passthrough active for channel {channel_id}, continuing transmission...")
                        
                        pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(
                            np.int16
                        )
                        pcm_bytes = pcm.tobytes()
                        
                        # Encode with Opus (like C code: opus_encode(encoder, pcm, 1920, opus_data, sizeof(opus_data)))
                        try:
                            opus_data = encoder.encode(pcm_bytes, 1920)  # type: ignore[attr-defined]
                            
                            if opus_data and len(opus_data) > 0:
                                # Encrypt with AES-256-GCM
                                if self._aes_key and HAS_AES:
                                    try:
                                        # Generate random IV (12 bytes for GCM)
                                        iv = os.urandom(12)
                                        aesgcm_enc = AESGCM(self._aes_key)
                                        encrypted = aesgcm_enc.encrypt(iv, opus_data, None)
                                        
                                        # Combine IV + ciphertext + tag (like C code)
                                        # C code: IV (12) + CT + tag (16)
                                        encrypted_with_iv = iv + encrypted
                                        
                                        # Base64 encode
                                        b64_data = base64.b64encode(encrypted_with_iv).decode('utf-8')
                                        
                                        # Send via UDP (like C code)
                                        # IMPORTANT: Continue broadcasting to UDP server even during passthrough
                                        if self._sock and self._server_addr:
                                            msg = json.dumps({
                                                "channel_id": channel_id,
                                                "type": "audio",
                                                "data": b64_data
                                            })
                                            self._sock.sendto(msg.encode('utf-8'), self._server_addr)  # nosec
                                            
                                            send_count += 1
                                            if send_count <= 5 or send_count % 500 == 0:
                                                print(f"[AUDIO TX] Channel {channel_id}: Sent audio packet #{send_count} "
                                                      f"({len(msg)} bytes) to {self._server_addr}, passthrough={'active' if passthrough_active else 'inactive'}")
                                            elif passthrough_active and (send_count % 200 == 0):
                                                print(f"[AUDIO TX] Channel {channel_id}: Broadcasting to UDP server (packet #{send_count}, passthrough active)")
                                        else:
                                            if send_count <= 10:
                                                print(f"[AUDIO TX ERROR] Channel {channel_id}: Cannot send - sock={self._sock is not None}, addr={self._server_addr}")
                                    except Exception as e:
                                        if send_count <= 10:
                                            print(f"[AUDIO TX ERROR] Channel {channel_id}: Encryption/send failed: {e}")
                                            import traceback
                                            traceback.print_exc()
                                else:
                                    if send_count <= 10:
                                        print(f"[AUDIO TX ERROR] Channel {channel_id}: No AES key available")
                        except Exception as e:
                            if send_count <= 10:
                                print(f"[AUDIO TX ERROR] Channel {channel_id}: Opus encode failed: {e}")
                        
                        # Reset buffer position
                        bundle[buffer_pos_key] = 0
            except Exception as e:
                if not self._running.is_set():
                    break
                if send_count % 100 == 0:
                    print(f"[AUDIO TX] Input error for {channel_id}: {e}")
                time.sleep(0.1)
        
        print(f"[AUDIO TX] Input worker stopped for channel {channel_id}")

    def start_transmission_for_channel(self, channel_index: int) -> bool:
        """Start audio transmission for a channel (like C code's start_transmission_for_channel)"""
        if channel_index < 0 or channel_index >= len(self._channel_ids):
            print(f"[AUDIO TX] ERROR: Invalid channel index {channel_index}")
            return False
        
        channel_id = self._channel_ids[channel_index]
        
        with self._input_lock:
            if channel_index in self._input_streams:
                print(f"[AUDIO TX] Channel {channel_id} already transmitting")
                self._transmitting[channel_index] = True
                return True
            
            # Select input device
            device_index = select_input_device_for_channel(channel_index)
            if device_index is None:
                print(f"[AUDIO TX] ERROR: No input device available for channel {channel_index}")
                return False
            
            # Open input stream
            pa, stream = open_input_stream(device_index, frames_per_buffer=1024)
            if pa is None or stream is None:
                print(f"[AUDIO TX] ERROR: Failed to open input stream for channel {channel_index}")
                return False
            
            # Create Opus encoder (like C code)
            encoder = None
            if HAS_OPUS:
                try:
                    encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_VOIP)  # type: ignore[attr-defined]
                    encoder.bitrate = 64000  # type: ignore[attr-defined]
                    encoder.vbr = True  # type: ignore[attr-defined]
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
                stream.start_stream()  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[AUDIO TX] ERROR: Failed to start input stream: {e}")
                close_stream(pa, stream)
                if encoder:
                    try:
                        encoder.destroy()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return False
            
            # Create input buffer (1920 samples like C code)
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
                daemon=True
            )
            bundle["thread"] = thread
            self._transmitting[channel_index] = True
            thread.start()
            
            print(f"[AUDIO TX] Started transmission for channel {channel_id} (index {channel_index}, device {device_index})")
            return True

    def stop_transmission_for_channel(self, channel_index: int) -> None:
        """Stop audio transmission for a channel"""
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


