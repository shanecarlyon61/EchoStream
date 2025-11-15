"""
Tone Detection Engine - FFT-based tone detection and alert generation

This module implements DTMF-style tone detection using FFT analysis,
parabolic interpolation for accurate frequency detection, and tone sequence matching.
"""
import numpy as np
import threading
import time
from typing import Optional, List, Tuple
from scipy import signal
from numpy.fft import rfft
from echostream import (
    SAMPLE_RATE, FFT_SIZE, FREQ_BINS, SAMPLES_PER_FRAME,
    MAX_TONE_DEFINITIONS, MAX_FILTERS
)
from config import ToneDefinition


# ============================================================================
# Tone Detection State
# ============================================================================

class ToneDetectionState:
    """Global tone detection state."""
    def __init__(self):
        self.tone_definitions: List[ToneDefinition] = [ToneDefinition() for _ in range(MAX_TONE_DEFINITIONS)]
        self.filters: List = []  # Frequency filters (not used yet)
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
        self.active_tone_def: Optional[ToneDefinition] = None
        self.passthrough_active = False
        
        # Audio buffer for sliding window analysis
        self.audio_buffer: List[float] = []  # List of float32 samples
        self.max_buffer_samples = int(SAMPLE_RATE * 10)  # 10 seconds max buffer
        self.last_detect_time = 0  # Prevent duplicate detections


# Global tone detection state
global_tone_detection = ToneDetectionState()


# ============================================================================
# Initialization
# ============================================================================

def init_tone_detection() -> bool:
    """
    Initialize tone detection system.
    
    Returns:
        True if initialization successful, False otherwise
    """
    global global_tone_detection
    
    if global_tone_detection is None:
        global_tone_detection = ToneDetectionState()
    
    # Ensure audio buffer is initialized
    if not hasattr(global_tone_detection, 'audio_buffer'):
        global_tone_detection.audio_buffer = []
        global_tone_detection.max_buffer_samples = int(SAMPLE_RATE * 10)
        global_tone_detection.last_detect_time = 0
    
    print("[TONE_DETECT] Tone detection system initialized")
    return True


def start_tone_detection() -> bool:
    """
    Start tone detection.
    
    Returns:
        True if started successfully
    """
    with global_tone_detection.mutex:
        global_tone_detection.active = True
    print("[TONE_DETECT] Tone detection started")
    return True


def stop_tone_detection():
    """Stop tone detection."""
    with global_tone_detection.mutex:
        global_tone_detection.active = False
    print("[TONE_DETECT] Tone detection stopped")


def is_tone_detect_enabled() -> bool:
    """
    Check if tone detection is enabled.
    
    Returns:
        True if tone detection is active
    """
    with global_tone_detection.mutex:
        return global_tone_detection.active


# ============================================================================
# Tone Definition Management
# ============================================================================

def add_tone_definition(
    tone_id: str,
    tone_a_freq: float,
    tone_b_freq: float,
    tone_a_length: int,
    tone_b_length: int,
    tone_a_range: int,
    tone_b_range: int,
    record_length: int,
    detection_tone_alert: Optional[str] = None
) -> bool:
    """
    Add a tone definition.
    
    Args:
        tone_id: Tone identifier string
        tone_a_freq: Frequency of tone A in Hz
        tone_b_freq: Frequency of tone B in Hz
        tone_a_length: Required duration of tone A in ms
        tone_b_length: Required duration of tone B in ms
        tone_a_range: Frequency tolerance for tone A in Hz
        tone_b_range: Frequency tolerance for tone B in Hz
        record_length: Recording duration after detection in ms
        detection_tone_alert: Alert type identifier (optional)
        
    Returns:
        True if tone definition added successfully, False otherwise
    """
    with global_tone_detection.mutex:
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
                print(f"[TONE_DETECT] Added tone definition: {tone_id}")
                return True
    
    print("[TONE_DETECT] ERROR: No free slot for tone definition")
    return False


def reset_detection_state():
    """Reset tone detection state."""
    with global_tone_detection.mutex:
        global_tone_detection.tone_a_confirmed = False
        global_tone_detection.tone_b_confirmed = False
        global_tone_detection.tone_a_tracking = False
        global_tone_detection.tone_b_tracking = False
        global_tone_detection.tone_a_tracking_start = 0
        global_tone_detection.tone_b_tracking_start = 0
        global_tone_detection.active_tone_def = None


def get_active_tone() -> Optional[ToneDefinition]:
    """
    Get currently detected tone.
    
    Returns:
        Active tone definition or None
    """
    with global_tone_detection.mutex:
        return global_tone_detection.active_tone_def


# ============================================================================
# Frequency Detection Utilities
# ============================================================================

def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    
    Args:
        f: Vector of values
        x: Index for that vector
        
    Returns:
        Tuple of (vx, vy), the coordinates of the vertex of a parabola
        that goes through point x and its two neighbors.
    """
    if x == 0 or x >= len(f) - 1:
        return float(x), float(f[x])
    
    try:
        numerator = f[x - 1] - f[x + 1]
        denominator = f[x - 1] - 2 * f[x] + f[x + 1]
        
        if abs(denominator) < 1e-10:
            return float(x), float(f[x])
        
        xv = 0.5 * numerator / denominator + x
        yv = f[x] - 0.25 * numerator * (xv - x)
        return float(xv), float(yv)
    except Exception:
        return float(x), float(f[x])


def freq_from_fft(sig: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """
    Estimate frequency from peak of FFT using parabolic interpolation.
    
    Args:
        sig: Audio signal samples (numpy array)
        fs: Sample rate in Hz (default: SAMPLE_RATE)
        
    Returns:
        Estimated frequency in Hz
    """
    if len(sig) < 2:
        return 0.0
    
    try:
        # Compute Fourier transform of windowed signal
        windowed = sig * np.hanning(len(sig))
        f = rfft(windowed)
        
        # Find the peak
        magnitudes = np.abs(f)
        i = int(np.argmax(magnitudes))
        
        if i == 0 or i >= len(magnitudes) - 1:
            # Can't interpolate at edges
            return float(fs * i / len(windowed))
        
        # Use parabolic interpolation for more accuracy
        try:
            # Add small value to avoid log(0)
            log_magnitudes = np.log(magnitudes + 1e-10)
            true_i, _ = parabolic(log_magnitudes, i)
            true_i = float(true_i)
        except Exception:
            true_i = float(i)
        
        # Convert to frequency
        freq = fs * true_i / len(windowed)
        return float(freq)
        
    except Exception as e:
        print(f"[TONE_DETECT] ERROR: freq_from_fft failed: {e}")
        return 0.0


def is_frequency_in_range(detected_freq: float, target_freq: float, range_hz: int) -> bool:
    """
    Check if detected frequency is within range of target.
    
    Args:
        detected_freq: Detected frequency in Hz
        target_freq: Target frequency in Hz
        range_hz: Tolerance range in Hz
        
    Returns:
        True if frequency is within range
    """
    return abs(detected_freq - target_freq) <= range_hz


# ============================================================================
# Tone Detection Processing
# ============================================================================

def process_audio_samples(samples: np.ndarray, sample_count: int) -> bool:
    """
    Process audio samples for tone detection.
    
    This is the main entry point for tone detection. It processes audio samples
    using a sliding window approach similar to the ToneDetect project.
    
    Args:
        samples: Audio samples array
        sample_count: Number of samples to process
        
    Returns:
        True if a tone was detected, False otherwise
    """
    if not global_tone_detection.active:
        return False
    
    current_time_ms = int(time.time() * 1000)
    
    # Add samples to sliding window buffer
    with global_tone_detection.mutex:
        # Extend buffer with new samples
        global_tone_detection.audio_buffer.extend(samples[:sample_count].tolist())
        
        # Keep only last max_buffer_samples
        if len(global_tone_detection.audio_buffer) > global_tone_detection.max_buffer_samples:
            excess = len(global_tone_detection.audio_buffer) - global_tone_detection.max_buffer_samples
            global_tone_detection.audio_buffer = global_tone_detection.audio_buffer[excess:]
        
        buffer_len = len(global_tone_detection.audio_buffer)
    
    # Log start of tone detection (first time only)
    static_start_logged = getattr(process_audio_samples, '_start_logged', False)
    if not static_start_logged:
        valid_tones = sum(1 for td in global_tone_detection.tone_definitions if td.valid)
        print(f"[TONE_DETECT] Tone detection active: {global_tone_detection.active}")
        print(f"[TONE_DETECT] Loaded {valid_tones} tone definition(s)")
        for i, td in enumerate(global_tone_detection.tone_definitions):
            if td.valid:
                print(f"[TONE_DETECT] Tone {i + 1}: ID={td.tone_id}, A={td.tone_a_freq}Hz±{td.tone_a_range_hz} ({td.tone_a_length_ms}ms), B={td.tone_b_freq}Hz±{td.tone_b_range_hz} ({td.tone_b_length_ms}ms)")
        process_audio_samples._start_logged = True
    
    # Check volume threshold first (like ToneDetect)
    if buffer_len < SAMPLE_RATE:  # Need at least 1 second
        return False
    
    # Get threshold from config
    threshold_db = -20  # Default
    try:
        import config
        for i in range(4):
            channel_config = config.get_channel_config(i)
            if channel_config and channel_config.valid and channel_config.tone_detect:
                threshold_db = channel_config.tone_config.db_threshold
                break
    except Exception:
        pass
    
    # Calculate volume from recent samples
    recent_samples = np.array(global_tone_detection.audio_buffer[-SAMPLE_RATE:])
    rms = np.sqrt(np.mean(recent_samples**2))
    volume_db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
    
    if volume_db < threshold_db:
        # Volume too low, skip processing
        return False
    
    # Get unique length groups from tone definitions
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
    static_process_count = getattr(process_audio_samples, '_process_count', 0)
    process_audio_samples._process_count = static_process_count + 1
    
    for l_a, l_b in unique_lengths:
        # Ensure l_a and l_b are valid numbers BEFORE any calculations
        try:
            # Check for NaN or Inf values first
            if not np.isfinite(l_a) or not np.isfinite(l_b) or l_a <= 0 or l_b <= 0:
                continue
            
            # Need at least l_a + l_b seconds of audio
            total_length = l_a + l_b
            if not np.isfinite(total_length) or total_length <= 0:
                continue
            
            required_samples = int(np.round(total_length * SAMPLE_RATE))
            if required_samples <= 0:
                continue
                
            if buffer_len < required_samples:
                continue
            
            # Extract tone A and tone B segments
            buf_array = np.array(global_tone_detection.audio_buffer, dtype=np.float32)
            
            # Calculate indices (ensure they are valid numbers)
            start_idx_val = (l_a + l_b) * SAMPLE_RATE
            end_idx_val = l_b * SAMPLE_RATE
            
            # Check if calculations resulted in valid numbers
            if not np.isfinite(start_idx_val) or not np.isfinite(end_idx_val):
                continue
            
            # Convert to integers, ensuring they're valid pure Python ints (not numpy types)
            start_idx = int(np.round(float(start_idx_val)))
            end_idx = int(np.round(float(end_idx_val)))
            
            # Ensure indices are Python ints for slicing (not numpy int64)
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            
            # Safety check: ensure indices are valid positive integers
            if start_idx <= 0 or end_idx <= 0 or start_idx <= end_idx:
                continue
            if len(buf_array) < start_idx:
                continue
            
            # Extract segments - ensure indices are integers for slicing
            tone_a_segment = buf_array[-start_idx:-end_idx] if end_idx > 0 else buf_array[-start_idx:]
            tone_b_segment = buf_array[-end_idx:]
            
            # Validate segment lengths
            if len(tone_a_segment) < int(SAMPLE_RATE * 0.1) or len(tone_b_segment) < int(SAMPLE_RATE * 0.1):
                continue
            
            # Detect frequencies using parabolic interpolation
            try:
                a_tone_freq = freq_from_fft(tone_a_segment, SAMPLE_RATE)
                b_tone_freq = freq_from_fft(tone_b_segment, SAMPLE_RATE)
            except Exception as e:
                if static_process_count % 100 == 0:
                    print(f"[TONE_DETECT] FFT error: {e}")
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
                
                # Prevent duplicate detections
                time_since_last = current_time_ms - global_tone_detection.last_detect_time
                max_tone_len_ms = max(tone_def.tone_a_length_ms, tone_def.tone_b_length_ms)
                
                if a_match and b_match and time_since_last > max_tone_len_ms:
                    print("=" * 60)
                    print("[TONE_DETECT] TONE SEQUENCE DETECTED!")
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
                    
                    # Trigger passthrough (will be handled by tone_passthrough module)
                    try:
                        import tone_passthrough
                        tone_passthrough.enable_passthrough(tone_def)
                    except ImportError:
                        print("[TONE_DETECT] WARNING: tone_passthrough module not available")
                    
                    detected = True
                    break
            
            # Debug: Log detected frequencies (occasionally)
            if static_process_count % 500 == 0:  # Log every ~5 seconds
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
                        print(f"[TONE_DETECT] A: {a_tone_freq:.1f}Hz, B: {b_tone_freq:.1f}Hz | "
                              f"Target: A={tone_def.tone_a_freq}Hz, B={tone_def.tone_b_freq}Hz | Status: {status}")
                        break
            
            if detected:
                return True
                
        except (ValueError, OverflowError, TypeError) as e:
            # Skip this iteration if index calculation fails
            if static_process_count % 100 == 0:
                print(f"[TONE_DETECT] Index calculation error: {e}, l_a={l_a}, l_b={l_b}")
            continue
        except Exception as e:
            # Catch any other unexpected errors during processing
            if static_process_count % 100 == 0:
                print(f"[TONE_DETECT] Unexpected error in tone detection: {e}")
            continue
    
    # Check if recording timer expired
    if global_tone_detection.recording_active:
        remaining = get_recording_time_remaining_ms()
        if remaining <= 0:
            # Recording timer expired - stop passthrough
            try:
                import tone_passthrough
                tone_passthrough.disable_passthrough()
            except ImportError:
                with global_tone_detection.mutex:
                    global_tone_detection.passthrough_active = False
                    global_tone_detection.recording_active = False
                    global_tone_detection.active_tone_def = None
            reset_detection_state()
    
    return global_tone_detection.passthrough_active


def detect_tones(samples: np.ndarray) -> Optional[ToneDefinition]:
    """
    Detect tones in audio samples (alias for process_audio_samples).
    
    Args:
        samples: Audio samples array
        
    Returns:
        Detected tone definition or None
    """
    if process_audio_samples(samples, len(samples)):
        return get_active_tone()
    return None


# ============================================================================
# Recording Timer Management
# ============================================================================

def is_recording_active() -> bool:
    """
    Check if recording is active.
    
    Returns:
        True if recording is active
    """
    with global_tone_detection.mutex:
        return global_tone_detection.recording_active


def get_recording_time_remaining_ms() -> int:
    """
    Get remaining recording time in milliseconds.
    
    Returns:
        Remaining time in ms, or 0 if not recording
    """
    with global_tone_detection.mutex:
        if not global_tone_detection.recording_active:
            return 0
        
        elapsed = int((time.time() * 1000) - global_tone_detection.recording_start_time)
        remaining = global_tone_detection.recording_duration_ms - elapsed
        return max(0, remaining)


def start_recording_timer(record_length_ms: int) -> bool:
    """
    Start recording timer.
    
    Args:
        record_length_ms: Recording duration in milliseconds
        
    Returns:
        True if timer started successfully
    """
    with global_tone_detection.mutex:
        global_tone_detection.recording_active = True
        global_tone_detection.recording_start_time = int(time.time() * 1000)
        global_tone_detection.recording_duration_ms = record_length_ms
    print(f"[TONE_DETECT] Recording timer started: {record_length_ms} ms")
    return True


def stop_recording_timer():
    """Stop recording timer."""
    with global_tone_detection.mutex:
        global_tone_detection.recording_active = False
    print("[TONE_DETECT] Recording timer stopped")


# Legacy function name for compatibility
def process_audio_python_approach(samples: np.ndarray, sample_count: int) -> bool:
    """
    Process audio using Python approach (legacy function name).
    
    Args:
        samples: Audio samples array
        sample_count: Number of samples
        
    Returns:
        True if tone detected
    """
    return process_audio_samples(samples, sample_count)

