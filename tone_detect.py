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

global_tone_detection = ToneDetectionState()

def init_tone_detection() -> bool:
    """Initialize tone detection system"""
    global global_tone_detection
    global_tone_detection = ToneDetectionState()
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
    
    print(f"[TONE PASSTHROUGH] Complete alert pair confirmed: Tone A={tone_def.tone_a_freq} Hz, Tone B={tone_def.tone_b_freq} Hz (ID: {tone_def.tone_id})")
    print(f"[TONE PASSTHROUGH] Routing input audio to passthrough target: {tone_config.passthrough_channel}")
    
    # Enable passthrough mode
    with global_tone_detection.mutex:
        global_tone_detection.passthrough_active = True
        global_tone_detection.active_tone_def = tone_def
    
    # Start recording timer for passthrough duration
    if tone_def.record_length_ms > 0:
        start_recording_timer(tone_def.record_length_ms)
        print(f"[TONE PASSTHROUGH] Passthrough active for {tone_def.record_length_ms} ms")
    
    # Enable passthrough mode in audio module
    audio.set_passthrough_output_mode(True)

def process_audio_python_approach(samples: np.ndarray, sample_count: int) -> bool:
    """Process audio using Python approach with full tone sequence detection"""
    if not global_tone_detection.active:
        return False
    
    # Simple FFT-based frequency detection
    if sample_count < FFT_SIZE:
        return False
    
    # Debug: Log when processing audio (occasionally)
    static_process_count = getattr(process_audio_python_approach, '_process_count', 0)
    process_audio_python_approach._process_count = static_process_count + 1
    if static_process_count % 1000 == 0:  # Log every 1000th call (~every 10 seconds at 48kHz)
        valid_tones = sum(1 for td in global_tone_detection.tone_definitions if td.valid)
        if valid_tones == 0:
            print("[TONE DEBUG] Processing audio but no tone definitions loaded!")
        else:
            print(f"[TONE DEBUG] Processing audio, {valid_tones} tone definition(s) loaded")
    
    current_time_ms = int(time.time() * 1000)
    
    # Take FFT
    fft_result = np.fft.rfft(samples[:FFT_SIZE])
    magnitudes = np.abs(fft_result)
    
    # Find peak frequencies (use higher threshold to reduce false positives)
    peak_threshold = np.max(magnitudes) * 0.2
    peak_indices = signal.find_peaks(magnitudes, height=peak_threshold)[0]
    
    if len(peak_indices) == 0:
        # No peaks found, check for timeout
        if global_tone_detection.tone_sequence_active:
            time_since_tone_a = current_time_ms - global_tone_detection.tone_a_start_time
            if time_since_tone_a > 5000:  # 5 second timeout
                print("[TONE] Sequence reset due to timeout")
                reset_tone_tracking()
                with global_tone_detection.mutex:
                    global_tone_detection.tone_sequence_active = False
        return False
    
    # Convert bin indices to frequencies
    frequencies = peak_indices * SAMPLE_RATE / FFT_SIZE
    
    # Constants for hit/miss tracking
    HIT_REQUIRED = 3
    MISS_REQUIRED = 5
    GRACE_MS = 200
    
    # Check each tone definition
    for tone_def in global_tone_detection.tone_definitions:
        if not tone_def.valid:
            continue
        
        # Check for Tone A first
        if not global_tone_detection.tone_a_confirmed:
            # Check if Tone B appears before Tone A is confirmed - invalid sequence
            tone_b_detected = False
            for freq in frequencies:
                if is_frequency_in_range(freq, tone_def.tone_b_freq, tone_def.tone_b_range_hz):
                    tone_b_detected = True
                    break
            
            if tone_b_detected:
                print(f"[TONE] INVALID SEQUENCE: Tone B ({tone_def.tone_b_freq} Hz) detected before Tone A confirmed - resetting")
                reset_tone_tracking()
                continue
            
            # Check for Tone A
            tone_a_detected = False
            for freq in frequencies:
                if is_frequency_in_range(freq, tone_def.tone_a_freq, tone_def.tone_a_range_hz):
                    tone_a_detected = True
                    break
            
            if tone_a_detected:
                # Start or continue tracking Tone A
                if not global_tone_detection.tone_a_tracking:
                    global_tone_detection.tone_a_tracking = True
                    global_tone_detection.tone_a_tracking_start = current_time_ms
                    print(f"[TONE] Tone A tracking started: {tone_def.tone_a_freq} Hz (ID: {tone_def.tone_id})")
                else:
                    # Check if duration requirement is met
                    if check_tone_duration(False, current_time_ms, tone_def):
                        with global_tone_detection.mutex:
                            global_tone_detection.tone_a_confirmed = True
                            global_tone_detection.current_tone_a_detected = True
                            global_tone_detection.tone_a_start_time = current_time_ms
                            global_tone_detection.tone_sequence_active = True
                        print(f"[TONE CONFIRMED] Tone A confirmed! ({tone_def.tone_a_freq} Hz ±{tone_def.tone_a_range_hz} Hz, {tone_def.tone_a_length_ms} ms duration)")
            else:
                # Tone A lost - reset if grace period expired
                if global_tone_detection.tone_a_tracking:
                    elapsed = current_time_ms - global_tone_detection.tone_a_tracking_start
                    if elapsed > GRACE_MS:
                        print("[TONE] Tone A tracking reset - frequency lost")
                        with global_tone_detection.mutex:
                            global_tone_detection.tone_a_tracking = False
                            global_tone_detection.tone_a_tracking_start = 0
        
        # Check for Tone B (only if Tone A is confirmed)
        elif not global_tone_detection.tone_b_confirmed:
            tone_b_detected = False
            for freq in frequencies:
                if is_frequency_in_range(freq, tone_def.tone_b_freq, tone_def.tone_b_range_hz):
                    tone_b_detected = True
                    break
            
            if tone_b_detected:
                # Start or continue tracking Tone B
                if not global_tone_detection.tone_b_tracking:
                    global_tone_detection.tone_b_tracking = True
                    global_tone_detection.tone_b_tracking_start = current_time_ms
                    print(f"[TONE] Tone B tracking started: {tone_def.tone_b_freq} Hz (ID: {tone_def.tone_id})")
                else:
                    # Check if duration requirement is met
                    if check_tone_duration(True, current_time_ms, tone_def):
                        with global_tone_detection.mutex:
                            global_tone_detection.tone_b_confirmed = True
                            global_tone_detection.current_tone_b_detected = True
                            global_tone_detection.tone_b_start_time = current_time_ms
                        
                        print(f"[TONE CONFIRMED] Tone B confirmed! ({tone_def.tone_b_freq} Hz ±{tone_def.tone_b_range_hz} Hz, {tone_def.tone_b_length_ms} ms duration)")
                        print(f"[TONE ALERT] Complete alert pair confirmed: Tone A={tone_def.tone_a_freq} Hz, Tone B={tone_def.tone_b_freq} Hz")
                        
                        # Both tones confirmed - trigger passthrough
                        trigger_tone_passthrough(tone_def)
            else:
                # Tone B lost - reset if grace period expired
                if global_tone_detection.tone_b_tracking:
                    elapsed = current_time_ms - global_tone_detection.tone_b_tracking_start
                    if elapsed > GRACE_MS:
                        print("[TONE] Tone B tracking reset - frequency lost")
                        with global_tone_detection.mutex:
                            global_tone_detection.tone_b_tracking = False
                            global_tone_detection.tone_b_tracking_start = 0
    
    # Check if recording timer expired
    if global_tone_detection.recording_active:
        remaining = get_recording_time_remaining_ms()
        if remaining <= 0:
            # Recording timer expired - stop passthrough
            print("[TONE PASSTHROUGH] Recording timer expired - stopping passthrough")
            with global_tone_detection.mutex:
                global_tone_detection.passthrough_active = False
                global_tone_detection.recording_active = False
            import audio
            audio.set_passthrough_output_mode(False)
            reset_tone_tracking()
    
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

