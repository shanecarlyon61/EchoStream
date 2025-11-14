"""
Tone detection - FFT-based tone detection and alert generation
"""
import numpy as np
import threading
import time
from typing import Optional, List
from echostream import (
    MAX_CHANNELS, MAX_TONE_DEFINITIONS, MAX_FILTERS, FFT_SIZE, 
    SAMPLE_RATE, FREQ_BINS, SAMPLES_PER_FRAME
)
from scipy import signal
from numpy.fft import rfft
import config
import mqtt

# Tone definition structure
class ToneDefinition:
    def __init__(self):
        self.tone_id = ""
        self.tone_a_freq = 0.0
        self.tone_b_freq = 0.0
        self.tone_a_length_ms = 0
        self.tone_b_length_ms = 0
        self.tone_a_range_hz = 0
        self.tone_b_range_hz = 0
        self.record_length_ms = 0
        self.detection_tone_alert = ""
        self.valid = False

# Filter definition
class FrequencyFilter:
    def __init__(self):
        self.filter_id = ""
        self.frequency = 0.0
        self.filter_range_hz = 0
        self.filter_type = ""  # "above", "below", "center"
        self.valid = False

# Tone detection state
class ToneDetectionState:
    def __init__(self):
        self.tone_definitions = [ToneDefinition() for _ in range(MAX_TONE_DEFINITIONS)]
        self.filters = [FrequencyFilter() for _ in range(MAX_FILTERS)]
        self.current_tone_a_detected = False
        self.current_tone_b_detected = False
        self.tone_sequence_active = False
        self.recording_active = False
        self.recording_start_time = 0
        self.recording_duration_ms = 0
        self.tone_a_start_time = 0
        self.tone_b_start_time = 0
        self.active = False
        self.mutex = threading.Lock()
        
        # Tone sequence tracking
        self.tone_a_confirmed = False
        self.tone_b_confirmed = False
        self.tone_a_tracking = False
        self.tone_b_tracking = False
        self.tone_a_tracking_start = 0
        self.tone_b_tracking_start = 0
        self.active_tone_def = None  # Currently active tone definition
        self.passthrough_active = False
        
        # Audio buffer for sliding window analysis (like ToneDetect project)
        self.audio_buffer = []  # List of float32 samples
        self.max_buffer_samples = int(SAMPLE_RATE * 10)  # 10 seconds max buffer
        self.last_detect_time = 0  # Prevent duplicate detections

global_tone_detection = ToneDetectionState()

def init_tone_detection() -> bool:
    """Initialize tone detection system"""
    global global_tone_detection
    # Don't recreate if already initialized (tone definitions may have been loaded)
    if global_tone_detection is None:
        global_tone_detection = ToneDetectionState()
    # Just ensure audio_buffer is initialized
    if not hasattr(global_tone_detection, 'audio_buffer'):
        global_tone_detection.audio_buffer = []
        global_tone_detection.max_buffer_samples = int(SAMPLE_RATE * 10)
        global_tone_detection.last_detect_time = 0
    print("[INFO] Tone detection system initialized")
    return True

def start_tone_detection() -> bool:
    """Start tone detection"""
    with global_tone_detection.mutex:
        global_tone_detection.active = True
    print("[INFO] Tone detection started")
    return True

def stop_tone_detection():
    """Stop tone detection"""
    with global_tone_detection.mutex:
        global_tone_detection.active = False
    print("[INFO] Tone detection stopped")

def add_tone_definition(tone_id: str, tone_a_freq: float, tone_b_freq: float,
                       tone_a_length: int, tone_b_length: int,
                       tone_a_range: int, tone_b_range: int,
                       record_length: int, detection_tone_alert: Optional[str]) -> bool:
    """Add a tone definition"""
    for tone_def in global_tone_detection.tone_definitions:
        if not tone_def.valid:
            tone_def.tone_id = tone_id
            tone_def.tone_a_freq = tone_a_freq
            tone_def.tone_b_freq = tone_b_freq
            tone_def.tone_a_length_ms = tone_a_length
            tone_def.tone_b_length_ms = tone_b_length
            tone_def.tone_a_range_hz = tone_a_range
            tone_def.tone_b_range_hz = tone_b_range
            tone_def.record_length_ms = record_length
            tone_def.detection_tone_alert = detection_tone_alert or ""
            tone_def.valid = True
            print(f"Added tone definition: {tone_id}")
            return True
    return False

def add_frequency_filter(filter_id: str, frequency: float, filter_range: int, filter_type: str) -> bool:
    """Add a frequency filter"""
    for filt in global_tone_detection.filters:
        if not filt.valid:
            filt.filter_id = filter_id
            filt.frequency = frequency
            filt.filter_range_hz = filter_range
            filt.filter_type = filter_type
            filt.valid = True
            print(f"Added frequency filter: {filter_id}")
            return True
    return False

def set_tone_config(threshold: float, gain: float, db_threshold: int,
                   detect_new_tones: bool, new_tone_length: int, new_tone_range: int) -> bool:
    """Set tone detection configuration"""
    # Configuration is stored in config module
    print(f"Tone config set: threshold={threshold}, gain={gain}, db={db_threshold}")
    return True

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    """
    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
    return xv, yv

def freq_from_fft(sig, fs=SAMPLE_RATE):
    """
    Estimate frequency from peak of FFT using parabolic interpolation
    (from ToneDetect project)
    """
    if len(sig) < 2:
        return 0.0
    
    # Compute Fourier transform of windowed signal
    windowed = sig * np.hanning(len(sig))
    f = rfft(windowed)
    
    # Find the peak and interpolate to get a more accurate peak
    magnitudes = np.abs(f)
    i = np.argmax(magnitudes)  # Just use this for less-accurate, naive version
    
    if i == 0 or i >= len(magnitudes) - 1:
        # Can't interpolate at edges
        return fs * i / len(windowed)
    
    # Use parabolic interpolation for more accuracy
    try:
        true_i = parabolic(np.log(magnitudes + 1e-10), i)[0]  # Add small value to avoid log(0)
    except:
        true_i = float(i)
    
    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def is_frequency_in_range(detected_freq: float, target_freq: float, range_hz: int) -> bool:
    """Check if detected frequency is within range of target"""
    return abs(detected_freq - target_freq) <= range_hz

def check_tone_duration(is_tone_b: bool, current_time_ms: int, tone_def: ToneDefinition) -> bool:
    """Check if tone has been present for required duration"""
    if is_tone_b:
        if not global_tone_detection.tone_b_tracking:
            return False
        elapsed = current_time_ms - global_tone_detection.tone_b_tracking_start
        return elapsed >= tone_def.tone_b_length_ms
    else:
        if not global_tone_detection.tone_a_tracking:
            return False
        elapsed = current_time_ms - global_tone_detection.tone_a_tracking_start
        return elapsed >= tone_def.tone_a_length_ms

def reset_tone_tracking():
    """Reset tone tracking state"""
    with global_tone_detection.mutex:
        global_tone_detection.tone_a_confirmed = False
        global_tone_detection.tone_b_confirmed = False
        global_tone_detection.tone_a_tracking = False
        global_tone_detection.tone_b_tracking = False
        global_tone_detection.tone_a_tracking_start = 0
        global_tone_detection.tone_b_tracking_start = 0
        global_tone_detection.active_tone_def = None

def trigger_tone_passthrough(tone_def: ToneDefinition):
    """Trigger passthrough routing when tone sequence is confirmed"""
    import audio
    import config
    
    # Find which channel has tone detection enabled
    tone_config = None
    source_channel_idx = -1
    
    for i in range(MAX_CHANNELS):
        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            tone_config = channel_config.tone_config
            source_channel_idx = i
            break
    
    if not tone_config or not tone_config.tone_passthrough:
        print("[TONE PASSTHROUGH] Passthrough not enabled in config")
        return
    
    print("=" * 60)
    print("[🔊 PASSTHROUGH START]")
    print(f"  Tone ID: {tone_def.tone_id}")
    print(f"  Source Channel: {source_channel_idx + 1}")
    print(f"  Target Channel: {tone_config.passthrough_channel}")
    print(f"  Duration: {tone_def.record_length_ms} ms ({tone_def.record_length_ms / 1000:.1f} seconds)")
    print(f"  Alert Type: {tone_def.detection_tone_alert if tone_def.detection_tone_alert else 'N/A'}")
    print("=" * 60)
    
    # Enable passthrough mode
    with global_tone_detection.mutex:
        global_tone_detection.passthrough_active = True
        global_tone_detection.active_tone_def = tone_def
    
    # Start recording timer for passthrough duration
    if tone_def.record_length_ms > 0:
        start_recording_timer(tone_def.record_length_ms)
        print(f"[PASSTHROUGH] Timer started: {tone_def.record_length_ms} ms")
    
    # Enable passthrough mode in audio module
    audio.set_passthrough_output_mode(True)
    print(f"[PASSTHROUGH] Audio routing enabled to {tone_config.passthrough_channel}")

def process_audio_python_approach(samples: np.ndarray, sample_count: int) -> bool:
    """Process audio using sliding window approach (like ToneDetect project)"""
    if not global_tone_detection.active:
        return False
    
    current_time_ms = int(time.time() * 1000)
    
    # Add samples to sliding window buffer
    with global_tone_detection.mutex:
        global_tone_detection.audio_buffer.extend(samples[:sample_count])
        # Keep only last max_buffer_samples
        if len(global_tone_detection.audio_buffer) > global_tone_detection.max_buffer_samples:
            global_tone_detection.audio_buffer = global_tone_detection.audio_buffer[-global_tone_detection.max_buffer_samples:]
        buffer_len = len(global_tone_detection.audio_buffer)
    
    # Log start of tone detection (first time only)
    static_start_logged = getattr(process_audio_python_approach, '_start_logged', False)
    if not static_start_logged:
        valid_tones = sum(1 for td in global_tone_detection.tone_definitions if td.valid)
        print(f"[TONE DETECTION START] Tone detection active: {global_tone_detection.active}")
        print(f"[TONE DETECTION START] Loaded {valid_tones} tone definition(s)")
        for i, td in enumerate(global_tone_detection.tone_definitions):
            if td.valid:
                print(f"[TONE DETECTION START] Tone {i+1}: ID={td.tone_id}, A={td.tone_a_freq}Hz±{td.tone_a_range_hz} ({td.tone_a_length_ms}ms), B={td.tone_b_freq}Hz±{td.tone_b_range_hz} ({td.tone_b_length_ms}ms)")
        process_audio_python_approach._start_logged = True
    
    # Debug: Log when processing audio (occasionally)
    static_process_count = getattr(process_audio_python_approach, '_process_count', 0)
    process_audio_python_approach._process_count = static_process_count + 1
    if static_process_count % 1000 == 0:  # Log every 1000th call (~every 10 seconds at 48kHz)
        valid_tones = sum(1 for td in global_tone_detection.tone_definitions if td.valid)
        buffer_seconds = buffer_len / SAMPLE_RATE
        print(f"[TONE DEBUG] Processing audio: {valid_tones} tone(s), buffer={buffer_seconds:.2f}s ({buffer_len} samples)")
    
    # Check volume threshold first (like ToneDetect)
    # Calculate RMS volume
    if len(global_tone_detection.audio_buffer) < SAMPLE_RATE:  # Need at least 1 second
        return False
    
    # Get threshold from config
    threshold_db = -20  # Default
    try:
        import config
        for i in range(MAX_CHANNELS):
            channel_config = config.get_channel_config(i)
            if channel_config and channel_config.valid and channel_config.tone_detect:
                threshold_db = channel_config.tone_config.db_threshold
                break
    except:
        pass
    
    # Calculate volume from recent samples
    recent_samples = np.array(global_tone_detection.audio_buffer[-SAMPLE_RATE:])  # Last 1 second
    rms = np.sqrt(np.mean(recent_samples**2))
    volume_db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
    
    if volume_db < threshold_db:
        # Volume too low, skip processing
        return False
    
    # Get unique length groups from tone definitions (like ToneDetect)
    lengths = []
    for tone_def in global_tone_detection.tone_definitions:
        if tone_def.valid:
            l_a = tone_def.tone_a_length_ms / 1000.0  # Convert to seconds
            l_b = tone_def.tone_b_length_ms / 1000.0
            lengths.append((l_a, l_b))
    
    # Remove duplicates and sort by total length (longest first)
    unique_lengths = sorted(list(set(lengths)), key=lambda x: x[0] + x[1], reverse=True)
    
    if not unique_lengths:
        return False
    
    # Process each unique length group
    for l_a, l_b in unique_lengths:
        # Need at least l_a + l_b seconds of audio
        required_samples = int((l_a + l_b) * SAMPLE_RATE)
        if len(global_tone_detection.audio_buffer) < required_samples:
            continue
        
        # Extract tone A and tone B segments (like ToneDetect)
        buf_array = np.array(global_tone_detection.audio_buffer)
        # Convert to integer indices for array slicing
        start_idx = int((l_a + l_b) * SAMPLE_RATE)
        end_idx = int(l_b * SAMPLE_RATE)
        
        # Safety check: ensure indices are valid
        if start_idx <= 0 or end_idx <= 0 or start_idx <= end_idx:
            continue
        if len(buf_array) < start_idx:
            continue
        
        # Extract segments
        tone_a_segment = buf_array[-start_idx:-end_idx] if end_idx > 0 else buf_array[-start_idx:]
        tone_b_segment = buf_array[-end_idx:]
        
        if len(tone_a_segment) < int(SAMPLE_RATE * 0.1) or len(tone_b_segment) < int(SAMPLE_RATE * 0.1):
            continue
        
        # Detect frequencies using parabolic interpolation (like ToneDetect)
        try:
            a_tone_freq = freq_from_fft(tone_a_segment, SAMPLE_RATE)
            b_tone_freq = freq_from_fft(tone_b_segment, SAMPLE_RATE)
        except Exception as e:
            if static_process_count % 100 == 0:
                print(f"[TONE DEBUG] FFT error: {e}")
            continue
        
        # Check against tone definitions with matching lengths
        tolerance = 10  # Default tolerance in Hz
        detected = False
        
        for tone_def in global_tone_detection.tone_definitions:
            if not tone_def.valid:
                continue
            
            # Check if lengths match
            if abs(tone_def.tone_a_length_ms / 1000.0 - l_a) > 0.1 or \
               abs(tone_def.tone_b_length_ms / 1000.0 - l_b) > 0.1:
                continue
            
            # Check if frequencies match (use tone_a_range and tone_b_range as tolerance)
            a_match = abs(tone_def.tone_a_freq - a_tone_freq) <= max(tone_def.tone_a_range_hz, tolerance)
            b_match = abs(tone_def.tone_b_freq - b_tone_freq) <= max(tone_def.tone_b_range_hz, tolerance)
            
            # Prevent duplicate detections (like ToneDetect)
            time_since_last = current_time_ms - global_tone_detection.last_detect_time
            max_tone_len_ms = max(tone_def.tone_a_length_ms, tone_def.tone_b_length_ms)
            
            if a_match and b_match and time_since_last > max_tone_len_ms:
                print("=" * 60)
                print("[🎵 TONE SEQUENCE DETECTED! 🎵]")
                print(f"  Tone ID: {tone_def.tone_id}")
                print(f"  Tone A: {a_tone_freq:.1f} Hz (target: {tone_def.tone_a_freq} Hz ±{tone_def.tone_a_range_hz} Hz)")
                print(f"  Tone B: {b_tone_freq:.1f} Hz (target: {tone_def.tone_b_freq} Hz ±{tone_def.tone_b_range_hz} Hz)")
                print(f"  Tone A Length: {tone_def.tone_a_length_ms} ms")
                print(f"  Tone B Length: {tone_def.tone_b_length_ms} ms")
                print(f"  Record Length: {tone_def.record_length_ms} ms")
                if tone_def.detection_tone_alert:
                    print(f"  Alert Type: {tone_def.detection_tone_alert}")
                print("=" * 60)
                
                global_tone_detection.last_detect_time = current_time_ms
                
                # Trigger passthrough immediately (both tones detected at once)
                trigger_tone_passthrough(tone_def)
                break
        
        # Debug: Log detected frequencies (occasionally, with tone match status)
        if static_process_count % 500 == 0:  # Log every ~5 seconds
            # Check if any tone matches
            any_match = False
            for tone_def in global_tone_detection.tone_definitions:
                if not tone_def.valid:
                    continue
                if abs(tone_def.tone_a_length_ms / 1000.0 - l_a) > 0.1 or \
                   abs(tone_def.tone_b_length_ms / 1000.0 - l_b) > 0.1:
                    continue
                tolerance = max(tone_def.tone_a_range_hz, tone_def.tone_b_range_hz, 10)
                a_match = abs(tone_def.tone_a_freq - a_tone_freq) <= tolerance
                b_match = abs(tone_def.tone_b_freq - b_tone_freq) <= tolerance
                if a_match or b_match:
                    status = "MATCH" if (a_match and b_match) else ("A_ONLY" if a_match else "B_ONLY")
                    print(f"[TONE SCAN] A: {a_tone_freq:.1f}Hz, B: {b_tone_freq:.1f}Hz | "
                          f"Target: A={tone_def.tone_a_freq}Hz, B={tone_def.tone_b_freq}Hz | Status: {status}")
                    break
    
    # Check if recording timer expired
    if global_tone_detection.recording_active:
        remaining = get_recording_time_remaining_ms()
        if remaining <= 0:
            # Recording timer expired - stop passthrough
            print("=" * 60)
            print("[🔊 PASSTHROUGH STOP]")
            if global_tone_detection.active_tone_def:
                print(f"  Tone ID: {global_tone_detection.active_tone_def.tone_id}")
                print(f"  Duration completed: {global_tone_detection.active_tone_def.record_length_ms} ms")
            print("=" * 60)
            with global_tone_detection.mutex:
                global_tone_detection.passthrough_active = False
                global_tone_detection.recording_active = False
                global_tone_detection.active_tone_def = None
            import audio
            audio.set_passthrough_output_mode(False)
            reset_tone_tracking()
            print("[PASSTHROUGH] Audio routing disabled")
    
    return global_tone_detection.passthrough_active

def is_tone_detect_enabled() -> bool:
    """Check if tone detection is enabled"""
    with global_tone_detection.mutex:
        return global_tone_detection.active

def is_recording_active() -> bool:
    """Check if recording is active"""
    with global_tone_detection.mutex:
        return global_tone_detection.recording_active

def get_recording_time_remaining_ms() -> int:
    """Get remaining recording time in milliseconds"""
    with global_tone_detection.mutex:
        if not global_tone_detection.recording_active:
            return 0
        
        elapsed = int((time.time() * 1000) - global_tone_detection.recording_start_time)
        remaining = global_tone_detection.recording_duration_ms - elapsed
        return max(0, remaining)

def start_recording_timer(record_length_ms: int) -> bool:
    """Start recording timer"""
    with global_tone_detection.mutex:
        global_tone_detection.recording_active = True
        global_tone_detection.recording_start_time = int(time.time() * 1000)
        global_tone_detection.recording_duration_ms = record_length_ms
    print(f"[INFO] Recording timer started: {record_length_ms} ms")
    return True

def stop_recording_timer():
    """Stop recording timer"""
    with global_tone_detection.mutex:
        global_tone_detection.recording_active = False
    print("[INFO] Recording timer stopped")

def should_play_alert_on_channel(channel_index: int) -> bool:
    """Check if alert should play on channel"""
    # Simplified - always return False for now
    return False

def get_alert_audio_samples(output_buffer: np.ndarray, max_samples: int) -> int:
    """Get alert audio samples"""
    # Simplified - return 0 (no alert)
    return 0

