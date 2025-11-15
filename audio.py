"""
Audio handling - PortAudio, Opus encoding/decoding, and audio stream management
"""
import pyaudio
import opuslib
import numpy as np
import threading
import time
from typing import Optional, List
from echostream import (
    MAX_CHANNELS, CHANNEL_ID_LEN, JITTER_BUFFER_SIZE, 
    SAMPLES_PER_FRAME, SAMPLE_RATE, global_channel_ids, global_channel_count
)
import crypto
import udp
import config
import tone_detect

# Audio structures
class AudioFrame:
    def __init__(self):
        self.samples = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        self.sample_count = 0
        self.valid = False

class JitterBuffer:
    def __init__(self):
        self.frames = [AudioFrame() for _ in range(JITTER_BUFFER_SIZE)]
        self.write_index = 0
        self.read_index = 0
        self.frame_count = 0
        self.mutex = threading.Lock()

class AudioStream:
    def __init__(self):
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        self.encoder: Optional[opuslib.Encoder] = None
        self.decoder: Optional[opuslib.Decoder] = None
        self.key = [0] * 32
        self.transmitting = False
        self.gpio_active = False
        self.input_buffer: np.ndarray = np.zeros(4800, dtype=np.float32)
        self.output_jitter = JitterBuffer()
        self.buffer_size = 4800
        self.input_buffer_pos = 0
        self.current_output_frame_pos = 0
        self.device_index = -1
        self.channel_id = ""

class ChannelContext:
    def __init__(self):
        self.audio = AudioStream()
        self.thread: Optional[threading.Thread] = None
        self.active = False

# Global audio state
channels = [ChannelContext() for _ in range(MAX_CHANNELS)]
usb_devices = [-1] * MAX_CHANNELS
device_assigned = False

# Global shared audio buffer
class SharedAudioBuffer:
    def __init__(self):
        self.samples = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        self.sample_count = 0
        self.valid = False
        self.mutex = threading.Lock()
        self.data_ready = threading.Condition(self.mutex)

global_shared_buffer = SharedAudioBuffer()

# Global tone detection control
class ToneDetectControl:
    def __init__(self):
        self.enabled = True
        self.card1_input_enabled = True
        self.passthrough_mode = False
        self.mutex = threading.Lock()

global_tone_detect = ToneDetectControl()

# PyAudio instance
pa_instance: Optional[pyaudio.PyAudio] = None

def initialize_portaudio() -> bool:
    """Initialize PortAudio"""
    global pa_instance
    if pa_instance is not None:
        return True
    
    try:
        pa_instance = pyaudio.PyAudio()
        return True
    except Exception as e:
        print(f"PortAudio initialization failed: {e}")
        return False

def setup_audio_for_channel(audio_stream: AudioStream) -> bool:
    """Setup audio encoder/decoder for a channel"""
    try:
        # Setup Opus encoder
        audio_stream.encoder = opuslib.Encoder(SAMPLE_RATE, 1, opuslib.APPLICATION_VOIP)
        try:
            audio_stream.encoder.bitrate = 64000
            audio_stream.encoder.vbr = True
        except AttributeError:
            # Some opuslib versions use different API
            pass
        
        # Setup Opus decoder
        audio_stream.decoder = opuslib.Decoder(SAMPLE_RATE, 1)
        
        # Initialize buffers
        audio_stream.input_buffer = np.zeros(4800, dtype=np.float32)
        audio_stream.input_buffer_pos = 0
        audio_stream.current_output_frame_pos = 0
        audio_stream.gpio_active = False
        
        return True
    except Exception as e:
        print(f"Audio setup failed: {e}")
        return False

def auto_assign_usb_devices():
    """Auto-assign USB audio devices to channels"""
    global device_assigned, usb_devices
    
    if device_assigned or pa_instance is None:
        return
    
    print("Scanning for USB audio devices...")
    
    num_devices = pa_instance.get_device_count()
    usb_count = 0
    
    for i in range(num_devices):
        if usb_count >= MAX_CHANNELS:
            break
        
        try:
            device_info = pa_instance.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                name = device_info['name'].lower()
                if 'usb' in name or 'audio device' in name or 'headset' in name:
                    usb_devices[usb_count] = i
                    print(f"USB Device {i} assigned to slot {usb_count}: {device_info['name']}")
                    usb_count += 1
        except Exception:
            continue
    
    if usb_count == 0:
        print("No USB audio devices found, using default input device for all channels")
        default_device = pa_instance.get_default_input_device_info()['index']
        for i in range(MAX_CHANNELS):
            usb_devices[i] = default_device
    elif usb_count < MAX_CHANNELS:
        print(f"Only {usb_count} USB device(s) found, some channels will share devices")
        for i in range(usb_count, MAX_CHANNELS):
            usb_devices[i] = usb_devices[i % usb_count]
    
    print("Channel assignments:")
    for i in range(global_channel_count):
        print(f"Channel {global_channel_ids[i]} -> Device {usb_devices[i]}")
    
    device_assigned = True

def get_device_for_channel(channel: str) -> int:
    """Get device index for a channel"""
    auto_assign_usb_devices()
    
    # Find channel index
    channel_index = -1
    for i in range(global_channel_count):
        if channel == global_channel_ids[i]:
            channel_index = i
            break
    
    if 0 <= channel_index < MAX_CHANNELS:
        return usb_devices[channel_index]
    
    return usb_devices[0] if usb_devices[0] >= 0 else 0

def audio_input_callback(in_data, frame_count, time_info, status):
    """Audio input callback"""
    if status:
        print(f"Input callback status: {status}")
    
    # Get audio stream from user_data (passed via stream)
    # Note: PyAudio doesn't support user_data directly, so we'll handle this differently
    return (None, pyaudio.paContinue)

def audio_output_callback(in_data, frame_count, time_info, status):
    """Audio output callback"""
    if status:
        print(f"Output callback status: {status}")
    
    # Generate silence for now (will be filled by jitter buffer)
    output = np.zeros(frame_count, dtype=np.float32)
    return (output.tobytes(), pyaudio.paContinue)

def get_passthrough_target_channel_index() -> int:
    """Get the index of the passthrough target channel"""
    import config
    
    # Find channel with tone detection enabled
    for i in range(MAX_CHANNELS):
        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            tone_cfg = channel_config.tone_config
            if tone_cfg.tone_passthrough:
                # Map passthrough_channel string to channel index
                passthrough_channel = tone_cfg.passthrough_channel
                if passthrough_channel == "channel_four":
                    return 2  # Last channel (index 2 for 3 channels)
                elif passthrough_channel == "channel_three":
                    return 2
                elif passthrough_channel == "channel_two":
                    return 1
                elif passthrough_channel == "channel_one":
                    return 0
    return -1

def is_configured_passthrough_channel_id(channel_id: str) -> bool:
    """Check if channel_id matches the configured passthrough target"""
    import config
    from echostream import global_channel_ids, global_channel_count
    
    # Find channel with tone detection enabled
    for i in range(MAX_CHANNELS):
        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            tone_cfg = channel_config.tone_config
            if tone_cfg.tone_passthrough:
                # Map passthrough_channel to index
                passthrough_channel = tone_cfg.passthrough_channel
                idx = -1
                if passthrough_channel == "channel_four":
                    idx = 2 if global_channel_count > 2 else -1
                elif passthrough_channel == "channel_three":
                    idx = 2 if global_channel_count > 2 else -1
                elif passthrough_channel == "channel_two":
                    idx = 1 if global_channel_count > 1 else -1
                elif passthrough_channel == "channel_one":
                    idx = 0
                
                if idx >= 0 and idx < global_channel_count:
                    return channel_id == global_channel_ids[idx]
    return False

def audio_input_worker(audio_stream: AudioStream):
    """Audio input worker thread - captures audio and sends it"""
    from echostream import global_interrupted
    import udp
    import tone_detect
    import config
    
    print(f"[AUDIO] Input worker started for channel {audio_stream.channel_id}")
    
    # Check if this channel has tone detection enabled
    channel_has_tone_detect = False
    for i in range(MAX_CHANNELS):
        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            if channel_config.channel_id == audio_stream.channel_id:
                channel_has_tone_detect = True
                break
    
    read_count = 0
    skip_count = 0
    
    while not global_interrupted.is_set() and audio_stream.transmitting:
        if not audio_stream.gpio_active:
            skip_count += 1
            if skip_count % 100 == 0:  # Log every 10 seconds
                print(f"[AUDIO DEBUG] Channel {audio_stream.channel_id}: GPIO not active (gpio_active={audio_stream.gpio_active})")
            time.sleep(0.1)
            continue
        
        if audio_stream.input_stream is None:
            skip_count += 1
            if skip_count % 100 == 0:  # Log every 10 seconds
                print(f"[AUDIO DEBUG] Channel {audio_stream.channel_id}: Input stream is None")
            time.sleep(0.1)
            continue
        
        try:
            # Read audio data
            data = audio_stream.input_stream.read(1024, exception_on_overflow=False)
            if len(data) == 0:
                skip_count += 1
                if skip_count % 100 == 0:
                    print(f"[AUDIO DEBUG] Channel {audio_stream.channel_id}: Read 0 bytes from input stream")
                time.sleep(0.01)
                continue
            
            read_count += 1
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            if read_count % 1000 == 0:  # Log every ~10 seconds at 48kHz
                print(f"[AUDIO DEBUG] Channel {audio_stream.channel_id}: Read {len(audio_data)} samples (read_count={read_count}, skip_count={skip_count})")
            
            # Update shared buffer for tone detection (if this channel has tone detection)
            if channel_has_tone_detect and tone_detect.is_tone_detect_enabled():
                with global_shared_buffer.mutex:
                    sample_count = min(len(audio_data), SAMPLES_PER_FRAME)
                    global_shared_buffer.samples[:sample_count] = audio_data[:sample_count]
                    global_shared_buffer.sample_count = sample_count
                    global_shared_buffer.valid = True
                    global_shared_buffer.data_ready.notify_all()
                
                # Debug: Log occasionally that we're writing to shared buffer
                static_buffer_write_count = getattr(audio_input_worker, '_buffer_write_count', {})
                if audio_stream.channel_id not in static_buffer_write_count:
                    static_buffer_write_count[audio_stream.channel_id] = 0
                static_buffer_write_count[audio_stream.channel_id] += 1
                audio_input_worker._buffer_write_count = static_buffer_write_count
                if static_buffer_write_count[audio_stream.channel_id] % 1000 == 0:
                    print(f"[AUDIO DEBUG] Writing audio to shared buffer for tone detection (channel {audio_stream.channel_id}, count={static_buffer_write_count[audio_stream.channel_id]})")
            
            # Accumulate samples for EchoStream transmission
            for sample in audio_data:
                audio_stream.input_buffer[audio_stream.input_buffer_pos] = sample
                audio_stream.input_buffer_pos += 1
                
                if audio_stream.input_buffer_pos >= 1920:
                    # Convert to int16 PCM
                    pcm = (audio_stream.input_buffer[:1920] * 32767.0).astype(np.int16)
                    
                    # Encode with Opus
                    if audio_stream.encoder:
                        # Opus encoder expects bytes of int16 PCM data
                        pcm_bytes = pcm.tobytes()
                        opus_data = audio_stream.encoder.encode(pcm_bytes, 1920)
                        
                        if opus_data and len(opus_data) > 0:
                            # Encrypt
                            encrypted = crypto.encrypt_data(opus_data, bytes(audio_stream.key))
                            if encrypted:
                                # Base64 encode
                                b64_data = crypto.encode_base64(encrypted)
                                
                                # Send via UDP
                                if udp.global_udp_socket and udp.global_server_addr:
                                    msg = f'{{"channel_id":"{audio_stream.channel_id}","type":"audio","data":"{b64_data}"}}'
                                    try:
                                        udp.global_udp_socket.sendto(
                                            msg.encode('utf-8'),
                                            udp.global_server_addr
                                        )
                                    except Exception as e:
                                        pass  # Silently handle UDP errors
                                
                                del encrypted
                    
                    audio_stream.input_buffer_pos = 0
        except Exception as e:
            if not global_interrupted.is_set():
                print(f"[AUDIO] Input error for {audio_stream.channel_id}: {e}")
            time.sleep(0.1)
    
    print(f"[AUDIO] Input worker stopped for channel {audio_stream.channel_id}")

def audio_output_worker(audio_stream: AudioStream):
    """Audio output worker thread - plays audio from jitter buffer or passthrough"""
    from echostream import global_interrupted
    import tone_detect
    
    # Initialize last packet time tracking (if not already initialized)
    if not hasattr(audio_output_worker, '_last_packet_time'):
        audio_output_worker._last_packet_time = {}
    if not hasattr(audio_output_worker, '_underrun_count'):
        audio_output_worker._underrun_count = {}
    
    print(f"[AUDIO] Output worker started for channel {audio_stream.channel_id}")
    
    is_passthrough_target = is_configured_passthrough_channel_id(audio_stream.channel_id)
    
    output_count = 0
    silence_count = 0
    
    # Minimum buffer threshold - wait until we have at least 2 frames before starting playback
    # This prevents choppy audio from buffer underruns
    MIN_BUFFER_THRESHOLD = 2
    buffer_wait_count = 0
    
    # Buffer size: At 48kHz, 1024 samples take ~21.3ms to play
    # We need to rate-limit to prevent consuming frames faster than they arrive
    FRAMES_PER_BUFFER = 1024
    SAMPLE_RATE = 48000  # Explicitly set to match echostream.py
    TIME_PER_BUFFER = FRAMES_PER_BUFFER / SAMPLE_RATE  # ~0.0213 seconds per buffer
    
    # Rate limiting: track last write time to ensure we don't write too fast
    import time
    last_write_time = time.time()
    write_count = 0
    
    while not global_interrupted.is_set() and audio_stream.transmitting:
        if audio_stream.output_stream is None:
            silence_count += 1
            if silence_count % 100 == 0:  # Log every 10 seconds
                print(f"[AUDIO DEBUG] Channel {audio_stream.channel_id}: Output stream is None")
            time.sleep(0.1)
            continue
        
        try:
            samples_to_play = None
            
            # Check if this is passthrough target and passthrough is active
            if is_passthrough_target and tone_detect.global_tone_detection.passthrough_active:
                # Read from shared buffer (input audio from source channel)
                with global_shared_buffer.mutex:
                    if global_shared_buffer.valid and global_shared_buffer.sample_count > 0:
                        # Get samples from shared buffer
                        take = min(1024, global_shared_buffer.sample_count)
                        samples_to_play = global_shared_buffer.samples[:take].copy()
                        # Shift remaining samples
                        if global_shared_buffer.sample_count > take:
                            remaining = global_shared_buffer.sample_count - take
                            global_shared_buffer.samples[:remaining] = global_shared_buffer.samples[take:take+remaining]
                            global_shared_buffer.sample_count = remaining
                        else:
                            global_shared_buffer.valid = False
                            global_shared_buffer.sample_count = 0
                    else:
                        # No passthrough audio available - output silence
                        samples_to_play = None
            
            # If not passthrough or no passthrough data, get from jitter buffer
            # Match C implementation: accumulate samples from multiple frames to fill requested buffer
            if samples_to_play is None:
                frames_to_fill = 1024  # Target buffer size (match C: frames parameter)
                frames_filled = 0
                output_buffer = np.zeros(frames_to_fill, dtype=np.float32)
                
                # Lock jitter buffer and fill output buffer (match C: while loop)
                with audio_stream.output_jitter.mutex:
                    # Check buffer status for logging
                    jitter_frames = audio_stream.output_jitter.frame_count
                    
                    # Check if buffer is ready (has minimum threshold of frames)
                    if jitter_frames >= MIN_BUFFER_THRESHOLD:
                        buffer_ready = True
                        if buffer_wait_count > 0:
                            print(f"[JITTER READY] Channel {audio_stream.channel_id}: Buffer ready! frames={jitter_frames} (waited {buffer_wait_count} cycles)")
                            buffer_wait_count = 0
                    elif jitter_frames > 0:
                        # Buffer has some frames but not enough - still use them, but log wait
                        buffer_ready = True  # Allow playback even with fewer frames
                        buffer_wait_count += 1
                        if buffer_wait_count % 100 == 0:
                            print(f"[JITTER WAIT] Channel {audio_stream.channel_id}: Buffer low (frames={jitter_frames}/{MIN_BUFFER_THRESHOLD}, wait_count={buffer_wait_count})")
                    else:
                        # Buffer empty
                        buffer_ready = False
                        static_underrun_count = getattr(audio_output_worker, '_underrun_count', {})
                        if audio_stream.channel_id not in static_underrun_count:
                            static_underrun_count[audio_stream.channel_id] = 0
                        static_underrun_count[audio_stream.channel_id] += 1
                        audio_output_worker._underrun_count = static_underrun_count
                        
                        if static_underrun_count[audio_stream.channel_id] == 1:
                            print(f"[JITTER UNDERRUN] Channel {audio_stream.channel_id}: Buffer empty! (read_idx={audio_stream.output_jitter.read_index}, write_idx={audio_stream.output_jitter.write_index})")
                        elif static_underrun_count[audio_stream.channel_id] % 500 == 0:
                            print(f"[JITTER UNDERRUN] Channel {audio_stream.channel_id}: Buffer still empty (underrun_count={static_underrun_count[audio_stream.channel_id]})")
                    
                    # Fill buffer from jitter buffer (match C: while loop to fill ALL frames)
                    # IMPORTANT: Always fill entire buffer, even if we run out of jitter frames
                    while frames_filled < frames_to_fill:
                        if audio_stream.output_jitter.frame_count > 0:
                            frame = audio_stream.output_jitter.frames[audio_stream.output_jitter.read_index]
                            
                            if frame.valid:
                                # Calculate how many samples we can copy from current frame
                                remaining_in_frame = frame.sample_count - audio_stream.current_output_frame_pos
                                frames_to_copy = frames_to_fill - frames_filled
                                
                                if frames_to_copy > remaining_in_frame:
                                    frames_to_copy = remaining_in_frame
                                
                                # Copy samples from current frame with gain boost (match C: 1.5x gain)
                                output_gain = 1.5
                                for i in range(frames_to_copy):
                                    sample = frame.samples[audio_stream.current_output_frame_pos + i] * output_gain
                                    # Clamp to prevent distortion
                                    if sample > 1.0:
                                        sample = 1.0
                                    elif sample < -1.0:
                                        sample = -1.0
                                    output_buffer[frames_filled + i] = sample
                                
                                frames_filled += frames_to_copy
                                audio_stream.current_output_frame_pos += frames_to_copy
                                
                                # Check if we finished this frame
                                if audio_stream.current_output_frame_pos >= frame.sample_count:
                                    # Mark frame as consumed
                                    frame.valid = False
                                    old_read_idx = audio_stream.output_jitter.read_index
                                    audio_stream.output_jitter.read_index = (
                                        audio_stream.output_jitter.read_index + 1
                                    ) % JITTER_BUFFER_SIZE
                                    audio_stream.output_jitter.frame_count -= 1
                                    audio_stream.current_output_frame_pos = 0
                                    
                                    # Log frame consumption occasionally
                                    if output_count % 500 == 0:
                                        print(f"[JITTER READ] Channel {audio_stream.channel_id}: Consumed frame at idx {old_read_idx}, "
                                              f"remaining_frames={audio_stream.output_jitter.frame_count}")
                            else:
                                # Frame is invalid, skip it
                                audio_stream.output_jitter.read_index = (
                                    audio_stream.output_jitter.read_index + 1
                                ) % JITTER_BUFFER_SIZE
                                audio_stream.output_jitter.frame_count -= 1
                                audio_stream.current_output_frame_pos = 0
                                
                                static_invalid_count = getattr(audio_output_worker, '_invalid_frame_count', {})
                                if audio_stream.channel_id not in static_invalid_count:
                                    static_invalid_count[audio_stream.channel_id] = 0
                                static_invalid_count[audio_stream.channel_id] += 1
                                audio_output_worker._invalid_frame_count = static_invalid_count
                                
                                if static_invalid_count[audio_stream.channel_id] % 100 == 0:
                                    print(f"[JITTER WARNING] Channel {audio_stream.channel_id}: Invalid frame at read_idx {audio_stream.output_jitter.read_index} "
                                          f"(frame_count={audio_stream.output_jitter.frame_count}) - advancing read index")
                        else:
                            # No frames available, fill remainder with silence (match C implementation)
                            # This ensures we always fill the entire buffer
                            for i in range(frames_filled, frames_to_fill):
                                output_buffer[i] = 0.0
                            frames_filled = frames_to_fill
                            break
                
                # Always use the filled buffer (will contain audio + silence if buffer ran out)
                samples_to_play = output_buffer
            
            # Play audio (gain already applied during buffer fill)
            # Rate-limit output to match playback rate (48kHz: 1024 samples = ~21.3ms)
            current_time = time.time()
            elapsed = current_time - last_write_time
            
            # Ensure we don't write faster than playback rate
            # At 48kHz, 1024 samples should take ~21.3ms to play
            if elapsed < TIME_PER_BUFFER:
                # Wait for the remaining time to maintain correct playback rate
                sleep_time = TIME_PER_BUFFER - elapsed
                time.sleep(sleep_time)
                current_time = time.time()
            
            # Always write the full buffer (1024 samples) - this matches C implementation
            if samples_to_play is not None and len(samples_to_play) > 0:
                # Ensure samples are in correct format (gain already applied)
                samples_to_play = np.clip(samples_to_play, -1.0, 1.0)
                
                # Always write exactly FRAMES_PER_BUFFER samples (match C: always fill entire buffer)
                if len(samples_to_play) < FRAMES_PER_BUFFER:
                    # Pad with silence if needed (shouldn't happen, but be safe)
                    padded = np.zeros(FRAMES_PER_BUFFER, dtype=np.float32)
                    padded[:len(samples_to_play)] = samples_to_play
                    samples_to_play = padded
                elif len(samples_to_play) > FRAMES_PER_BUFFER:
                    # Truncate if somehow too large (shouldn't happen)
                    samples_to_play = samples_to_play[:FRAMES_PER_BUFFER]
                
                audio_bytes = samples_to_play.astype(np.float32).tobytes()
                
                # Write to output stream (this should block if buffer is full)
                try:
                    audio_stream.output_stream.write(audio_bytes, exception_on_underflow=False)
                    last_write_time = time.time()
                    write_count += 1
                except Exception as e:
                    print(f"[AUDIO ERROR] Channel {audio_stream.channel_id}: Write error: {e}")
                    time.sleep(0.01)  # Small delay on error
                    continue
                
                output_count += 1
                
                # Calculate RMS to detect if we're playing audio or silence
                rms = np.sqrt(np.mean(samples_to_play**2))
                is_silence = rms < 0.001  # Threshold for silence detection
                
                # Log buffer status more frequently to diagnose the issue
                if output_count % 500 == 0:  # Log every ~10 seconds (500 * 21.3ms = ~10.6s)
                    with audio_stream.output_jitter.mutex:
                        jitter_frames = audio_stream.output_jitter.frame_count
                        read_idx = audio_stream.output_jitter.read_index
                        write_idx = audio_stream.output_jitter.write_index
                    
                    elapsed_ms = elapsed * 1000
                    if is_silence:
                        print(f"[AUDIO OUT] Channel {audio_stream.channel_id}: Playing SILENCE "
                              f"(RMS={rms:.6f}, jitter_frames={jitter_frames}, elapsed={elapsed_ms:.2f}ms, "
                              f"read_idx={read_idx}, write_idx={write_idx})")
                    else:
                        print(f"[AUDIO OUT] Channel {audio_stream.channel_id}: Playing AUDIO "
                              f"(RMS={rms:.4f}, jitter_frames={jitter_frames}, elapsed={elapsed_ms:.2f}ms, "
                              f"read_idx={read_idx}, write_idx={write_idx})")
                    
                    # Warn if buffer is consistently low or empty
                    if jitter_frames == 0:
                        print(f"[AUDIO WARNING] Channel {audio_stream.channel_id}: Buffer EMPTY during playback!")
                    elif jitter_frames < 2:
                        print(f"[AUDIO WARNING] Channel {audio_stream.channel_id}: Buffer LOW (frames={jitter_frames}) - "
                              f"packets may not be arriving fast enough")
            else:
                # Fallback: should never happen since we always fill buffer above
                silence = np.zeros(FRAMES_PER_BUFFER, dtype=np.float32)
                try:
                    audio_stream.output_stream.write(silence.tobytes(), exception_on_underflow=False)
                    last_write_time = time.time()
                except Exception as e:
                    print(f"[AUDIO ERROR] Channel {audio_stream.channel_id}: Silence write error: {e}")
                    time.sleep(0.01)
                    continue
                
                silence_count += 1
                if silence_count % 1000 == 0:  # Log every ~10 seconds
                    with audio_stream.output_jitter.mutex:
                        jitter_frames = audio_stream.output_jitter.frame_count
                    print(f"[AUDIO OUT] Channel {audio_stream.channel_id}: Playing silence (fallback) "
                          f"(jitter_frames={jitter_frames})")
            
        except Exception as e:
            if not global_interrupted.is_set():
                print(f"[AUDIO] Output error for {audio_stream.channel_id}: {e}")
            time.sleep(0.1)
    
    print(f"[AUDIO] Output worker stopped for channel {audio_stream.channel_id}")

def start_transmission_for_channel(audio_stream: AudioStream) -> bool:
    """Start audio transmission for a channel"""
    if pa_instance is None:
        return False
    
    audio_stream.device_index = get_device_for_channel(audio_stream.channel_id)
    
    try:
        device_info = pa_instance.get_device_info_by_index(audio_stream.device_index)
        
        # Setup input stream
        audio_stream.input_stream = pa_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=audio_stream.device_index,
            frames_per_buffer=1024,
            stream_callback=None  # We'll use blocking I/O
        )
        audio_stream.input_stream.start_stream()
        
        # Setup output stream
        try:
            audio_stream.output_stream = pa_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                output_device_index=audio_stream.device_index,
                frames_per_buffer=1024,
                stream_callback=None
            )
            audio_stream.output_stream.start_stream()
        except Exception as e:
            print(f"Output stream failed, using input-only mode: {e}")
            audio_stream.output_stream = None
        
        audio_stream.transmitting = True
        
        # Start audio worker threads
        input_thread = threading.Thread(
            target=audio_input_worker,
            args=(audio_stream,),
            daemon=True
        )
        input_thread.start()
        
        if audio_stream.output_stream:
            output_thread = threading.Thread(
                target=audio_output_worker,
                args=(audio_stream,),
                daemon=True
            )
            output_thread.start()
        
        print(f"Audio transmission started for channel {audio_stream.channel_id}")
        return True
    except Exception as e:
        print(f"Failed to start transmission: {e}")
        return False

def setup_channel(ctx: ChannelContext, channel_id: str) -> bool:
    """Setup a channel"""
    ctx.audio.channel_id = channel_id
    
    if not setup_audio_for_channel(ctx.audio):
        print(f"[ERROR] Audio setup failed for channel {channel_id}")
        return False
    
    ctx.active = True
    print(f"[INFO] Channel {channel_id} setup completed successfully")
    return True

def process_received_audio(audio_stream: AudioStream, opus_data: bytes, channel_id: str, target_index: int):
    """Process received audio data and add to jitter buffer"""
    if audio_stream.decoder is None:
        return
    
    # Debug: Track audio reception
    static_receive_count = getattr(process_received_audio, '_receive_count', {})
    if channel_id not in static_receive_count:
        static_receive_count[channel_id] = 0
    static_receive_count[channel_id] += 1
    process_received_audio._receive_count = static_receive_count
    
    # Log first few packets and then occasionally
    if static_receive_count[channel_id] <= 5 or static_receive_count[channel_id] % 500 == 0:
        print(f"[AUDIO RX] Channel {channel_id}: Received audio packet #{static_receive_count[channel_id]} "
              f"({len(opus_data)} bytes)")
    
    try:
        # Decode Opus audio
        pcm_data = audio_stream.decoder.decode(opus_data, SAMPLES_PER_FRAME)
        
        if len(pcm_data) > 0:
            # Convert to float32 and normalize
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            float_samples = pcm_array.astype(np.float32) / 32767.0
            
            # Apply gain boost (match C implementation: 20x gain)
            float_samples *= 20.0
            float_samples = np.clip(float_samples, -1.0, 1.0)
            
            # Add to jitter buffer
            with audio_stream.output_jitter.mutex:
                buffer_before = audio_stream.output_jitter.frame_count
                write_idx_before = audio_stream.output_jitter.write_index
                
                if audio_stream.output_jitter.frame_count < JITTER_BUFFER_SIZE:
                    frame = audio_stream.output_jitter.frames[audio_stream.output_jitter.write_index]
                    frame.samples[:len(float_samples)] = float_samples
                    frame.sample_count = len(float_samples)
                    frame.valid = True
                    
                    audio_stream.output_jitter.write_index = (
                        audio_stream.output_jitter.write_index + 1
                    ) % JITTER_BUFFER_SIZE
                    audio_stream.output_jitter.frame_count += 1
                    
                    # Track last packet reception time for underrun diagnostics
                    static_last_packet_time = getattr(process_received_audio, '_last_packet_time', {})
                    static_last_packet_time[channel_id] = time.time()
                    process_received_audio._last_packet_time = static_last_packet_time
                    
                    # Update output worker's last packet time
                    if hasattr(audio_output_worker, '_last_packet_time'):
                        audio_output_worker._last_packet_time[channel_id] = time.time()
                    
                    # Log when buffer fills up initially
                    if static_receive_count[channel_id] <= 5 or static_receive_count[channel_id] % 500 == 0:
                        print(f"[JITTER WRITE] Channel {channel_id}: Added frame at idx {write_idx_before}, "
                              f"frames={audio_stream.output_jitter.frame_count}/{JITTER_BUFFER_SIZE}, "
                              f"read_idx={audio_stream.output_jitter.read_index}")
                else:
                    # Buffer full, drop oldest
                    old_read_idx = audio_stream.output_jitter.read_index
                    audio_stream.output_jitter.read_index = (
                        audio_stream.output_jitter.read_index + 1
                    ) % JITTER_BUFFER_SIZE
                    audio_stream.output_jitter.frame_count -= 1
                    
                    frame = audio_stream.output_jitter.frames[audio_stream.output_jitter.write_index]
                    frame.samples[:len(float_samples)] = float_samples
                    frame.sample_count = len(float_samples)
                    frame.valid = True
                    
                    audio_stream.output_jitter.write_index = (
                        audio_stream.output_jitter.write_index + 1
                    ) % JITTER_BUFFER_SIZE
                    
                    # Log when buffer overflows (drops frames)
                    if static_receive_count[channel_id] % 100 == 0:
                        print(f"[JITTER OVERFLOW] Channel {channel_id}: Buffer full, dropped frame at idx {old_read_idx}, "
                              f"frames={audio_stream.output_jitter.frame_count}/{JITTER_BUFFER_SIZE}")
                
                # Log buffer status when it transitions from empty to having data
                if buffer_before == 0 and audio_stream.output_jitter.frame_count > 0:
                    print(f"[JITTER RECOVERY] Channel {channel_id}: Buffer refilled! frames={audio_stream.output_jitter.frame_count}, "
                          f"write_idx={audio_stream.output_jitter.write_index}, read_idx={audio_stream.output_jitter.read_index}")
                
                # Log jitter buffer status with packet reception rate
                if static_receive_count[channel_id] % 500 == 0:
                    # Calculate packet reception rate (packets per second)
                    # If we've received packets, estimate rate based on count
                    packet_rate = "unknown"
                    if static_receive_count[channel_id] > 5:
                        # Rough estimate: assume packets arrive at ~50 packets/second (1920 samples * 50 = 96000 samples/sec = 48kHz)
                        # This is approximate, but helps diagnose if packets are arriving
                        packet_rate = "~50 pps (estimated)"
                    
                    print(f"[JITTER STATUS] Channel {channel_id}: frames={audio_stream.output_jitter.frame_count}/"
                          f"{JITTER_BUFFER_SIZE}, write_idx={audio_stream.output_jitter.write_index}, "
                          f"read_idx={audio_stream.output_jitter.read_index}, packets_received={static_receive_count[channel_id]}, "
                          f"rate={packet_rate}")
                    
                    # Warn if buffer is consistently low (might indicate packet loss)
                    if audio_stream.output_jitter.frame_count < 3:
                        print(f"[JITTER WARNING] Channel {channel_id}: Buffer consistently low (frames={audio_stream.output_jitter.frame_count}) - "
                              f"check if UDP packets are arriving (packets_received={static_receive_count[channel_id]})")
    except Exception as e:
        print(f"Error processing received audio: {e}")

def initialize_audio_devices() -> bool:
    """Initialize audio devices"""
    print("[AUDIO INIT] Starting comprehensive audio device initialization...")
    # In Python, we don't need to kill processes like in C
    # PyAudio handles device access
    return True

def cleanup_audio_devices() -> bool:
    """Cleanup audio devices"""
    print("[AUDIO CLEANUP] Restoring audio devices to normal state...")
    return True

def init_shared_audio_buffer() -> bool:
    """Initialize shared audio buffer"""
    global global_shared_buffer
    global_shared_buffer = SharedAudioBuffer()
    print("[INFO] Shared audio buffer initialized")
    return True

def init_audio_passthrough() -> bool:
    """Initialize audio passthrough"""
    print("[INFO] Audio passthrough initialized")
    return True

def start_audio_passthrough() -> bool:
    """Start audio passthrough"""
    print("[INFO] Audio passthrough enabled")
    return True

def stop_audio_passthrough():
    """Stop audio passthrough"""
    print("[INFO] Audio passthrough stopped")

def init_tone_detect_control() -> bool:
    """Initialize tone detection control"""
    global global_tone_detect
    global_tone_detect = ToneDetectControl()
    global_tone_detect.enabled = True
    global_tone_detect.card1_input_enabled = True
    print("[INFO] Tone detection control initialized")
    return True

def enable_tone_detection() -> bool:
    """Enable tone detection"""
    with global_tone_detect.mutex:
        global_tone_detect.enabled = True
        global_tone_detect.card1_input_enabled = True
    print("[INFO] Tone detection ENABLED")
    return True

def disable_tone_detection() -> bool:
    """Disable tone detection"""
    with global_tone_detect.mutex:
        global_tone_detect.enabled = False
        global_tone_detect.card1_input_enabled = False
    print("[INFO] Tone detection DISABLED")
    return True

def is_tone_detect_enabled() -> bool:
    """Check if tone detection is enabled"""
    with global_tone_detect.mutex:
        return global_tone_detect.enabled

def is_card1_input_enabled() -> bool:
    """Check if Card 1 input is enabled"""
    with global_tone_detect.mutex:
        return global_tone_detect.card1_input_enabled

def is_passthrough_mode() -> bool:
    """Check if passthrough mode is enabled"""
    with global_tone_detect.mutex:
        return global_tone_detect.passthrough_mode

def set_passthrough_output_mode(passthrough_mode: bool) -> bool:
    """Set passthrough output mode"""
    with global_tone_detect.mutex:
        global_tone_detect.passthrough_mode = passthrough_mode
    print(f"[INFO] Passthrough output mode set to {'PASSTHROUGH' if passthrough_mode else 'ECHOSTREAM'}")
    return True

