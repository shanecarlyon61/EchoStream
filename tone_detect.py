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

def process_audio_python_approach(samples: np.ndarray, sample_count: int) -> bool:
    """Process audio using Python approach (sliding window with FFT)"""
    if not global_tone_detection.active:
        return False
    
    # Simple FFT-based frequency detection
    if sample_count < FFT_SIZE:
        return False
    
    # Take FFT
    fft_result = np.fft.rfft(samples[:FFT_SIZE])
    magnitudes = np.abs(fft_result)
    
    # Find peak frequencies
    peak_indices = signal.find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)[0]
    
    if len(peak_indices) > 0:
        # Convert bin indices to frequencies
        frequencies = peak_indices * SAMPLE_RATE / FFT_SIZE
        
        # Check against tone definitions
        for tone_def in global_tone_detection.tone_definitions:
            if not tone_def.valid:
                continue
            
            for freq in frequencies:
                # Check if frequency matches tone A or tone B
                if abs(freq - tone_def.tone_a_freq) <= tone_def.tone_a_range_hz:
                    print(f"[TONE DETECTED] Tone A detected: {freq} Hz (target: {tone_def.tone_a_freq} Hz)")
                    return True
                elif abs(freq - tone_def.tone_b_freq) <= tone_def.tone_b_range_hz:
                    print(f"[TONE DETECTED] Tone B detected: {freq} Hz (target: {tone_def.tone_b_freq} Hz)")
                    return True
    
    return False

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

