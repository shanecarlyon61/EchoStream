"""
Tone detection module - detects tone sequences (Tone A followed by Tone B)
from filtered audio on channels with tone_detect enabled.
"""
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from numpy.fft import rfft

SAMPLE_RATE = 48000
FFT_SIZE = 1024
FREQ_BINS = FFT_SIZE // 2
HIT_REQUIRED = 1      # require K hits (reduced from 2 to 1 for faster confirmation)
MISS_REQUIRED = 3     # require K misses (increased from 2 to 3 for stability)
GRACE_MS = 500        # allow brief gaps without resetting (increased from 250ms)


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    """
    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    try:
        xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
        yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
        return xv, yv
    except (ZeroDivisionError, IndexError):
        return float(x), float(f[x])


def frequency_to_bin(frequency: float) -> float:
    """Convert frequency (Hz) to FFT bin index."""
    return (frequency * FFT_SIZE) / SAMPLE_RATE


def is_frequency_in_range(detected_freq: float, target_freq: float, range_hz: int) -> bool:
    """Check if detected frequency is within range of target"""
    return abs(detected_freq - target_freq) <= range_hz


class ChannelToneDetector:
    """Tone detector for a single channel"""
    
    def __init__(self, channel_id: str, tone_definitions: List[Dict[str, Any]]):
        self.channel_id = channel_id
        self.tone_definitions = tone_definitions
        self.audio_buffer: List[float] = []
        self.max_buffer_samples = int(SAMPLE_RATE * 10)
        self.mutex = threading.Lock()
        
        self.tone_a_tracking: Dict[str, bool] = {}
        self.tone_b_tracking: Dict[str, bool] = {}
        self.tone_a_confirmed: Dict[str, bool] = {}
        self.tone_b_confirmed: Dict[str, bool] = {}
        self.tone_a_tracking_start: Dict[str, int] = {}
        self.tone_b_tracking_start: Dict[str, int] = {}
        self.tone_a_hit_streak: Dict[str, int] = {}
        self.tone_b_hit_streak: Dict[str, int] = {}
        self.tone_a_miss_streak: Dict[str, int] = {}
        self.tone_b_miss_streak: Dict[str, int] = {}
        self.tone_a_last_seen: Dict[str, int] = {}
        self.tone_b_last_seen: Dict[str, int] = {}
        
        for tone_def in tone_definitions:
            tone_id = tone_def["tone_id"]
            self.tone_a_tracking[tone_id] = False
            self.tone_b_tracking[tone_id] = False
            self.tone_a_confirmed[tone_id] = False
            self.tone_b_confirmed[tone_id] = False
            self.tone_a_tracking_start[tone_id] = 0
            self.tone_b_tracking_start[tone_id] = 0
            self.tone_a_hit_streak[tone_id] = 0
            self.tone_b_hit_streak[tone_id] = 0
            self.tone_a_miss_streak[tone_id] = 0
            self.tone_b_miss_streak[tone_id] = 0
            self.tone_a_last_seen[tone_id] = 0
            self.tone_b_last_seen[tone_id] = 0
        
    def add_audio_samples(self, samples: np.ndarray):
        """Add audio samples to the buffer"""
        with self.mutex:
            self.audio_buffer.extend(samples.tolist())
            if len(self.audio_buffer) > self.max_buffer_samples:
                self.audio_buffer = self.audio_buffer[-self.max_buffer_samples:]
    
    def _find_peak_frequencies(self, audio_samples: np.ndarray) -> List[float]:
        """Perform FFT and find peak frequencies using parabolic interpolation"""
        if len(audio_samples) < FFT_SIZE:
            return []
        
        windowed = audio_samples[:FFT_SIZE] * np.hanning(FFT_SIZE)
        fft_output = rfft(windowed)
        
        magnitudes = np.abs(fft_output)
        max_magnitude = np.max(magnitudes)
        
        if max_magnitude == 0:
            return []
        
        db_threshold = -45
        absolute_db_threshold = 10.0 ** (db_threshold / 20.0)
        relative_threshold = max_magnitude * 0.1
        magnitude_threshold = max(absolute_db_threshold, relative_threshold)
        
        peaks = []
        for i in range(1, len(magnitudes) - 1):
            if (magnitudes[i] > magnitudes[i-1] and 
                magnitudes[i] > magnitudes[i+1] and 
                magnitudes[i] > magnitude_threshold):
                try:
                    true_i, _ = parabolic(np.log(magnitudes + 1e-10), i)
                    freq = SAMPLE_RATE * true_i / FFT_SIZE
                    peaks.append(freq)
                except Exception:
                    freq = SAMPLE_RATE * i / FFT_SIZE
                    peaks.append(freq)
        
        return sorted(peaks)
    
    def _check_tone_frequency(self, peak_freqs: List[float], target_freq: float, range_hz: int) -> bool:
        """Check if any peak frequency matches the target frequency within range"""
        for peak_freq in peak_freqs:
            if is_frequency_in_range(peak_freq, target_freq, range_hz):
                return True
        return False
    
    def process_audio(self) -> Optional[Dict[str, Any]]:
        """
        Process audio buffer and detect tone sequences.
        Returns tone definition dict if sequence detected, None otherwise.
        """
        with self.mutex:
            if len(self.audio_buffer) < FFT_SIZE:
                return None
            
            buffer_array = np.array(self.audio_buffer[-FFT_SIZE:], dtype=np.float32)
            current_time_ms = int(time.time() * 1000)
            
            peak_freqs = self._find_peak_frequencies(buffer_array)
            
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            self._debug_count += 1
            if self._debug_count % 200 == 0 and peak_freqs:
                print(f"[TONE DETECTION] Channel {self.channel_id}: "
                      f"Found {len(peak_freqs)} peak frequency(ies): "
                      f"{', '.join(f'{f:.1f} Hz' for f in peak_freqs[:5])}")
            
            for tone_def in self.tone_definitions:
                tone_id = tone_def["tone_id"]
                
                if not self.tone_a_confirmed.get(tone_id, False):
                    tone_b_detected = self._check_tone_frequency(
                        peak_freqs, tone_def["tone_b"], tone_def["tone_b_range"]
                    )
                    if tone_b_detected:
                        self.tone_a_hit_streak[tone_id] = 0
                        self.tone_a_tracking[tone_id] = False
                        self.tone_a_tracking_start[tone_id] = 0
                        continue
                    
                    tone_a_detected = self._check_tone_frequency(
                        peak_freqs, tone_def["tone_a"], tone_def["tone_a_range"]
                    )
                    
                    if tone_a_detected:
                        self.tone_a_hit_streak[tone_id] = self.tone_a_hit_streak.get(tone_id, 0) + 1
                        self.tone_a_miss_streak[tone_id] = 0
                        self.tone_a_last_seen[tone_id] = current_time_ms
                        if self.tone_a_tracking.get(tone_id, False):
                            self.tone_a_miss_streak[tone_id] = 0
                        
                        matching_freq = next(
                            (f for f in peak_freqs 
                             if is_frequency_in_range(f, tone_def["tone_a"], 
                                                     tone_def["tone_a_range"])),
                            None
                        )
                        
                        if self.tone_a_hit_streak[tone_id] >= HIT_REQUIRED:
                            if not self.tone_a_tracking.get(tone_id, False):
                                self.tone_a_tracking[tone_id] = True
                                self.tone_a_tracking_start[tone_id] = current_time_ms
                                self.tone_a_miss_streak[tone_id] = 0
                                print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                      f"Tone A detected! Starting duration tracking for "
                                      f"{tone_def['tone_a']:.1f} Hz ±{tone_def['tone_a_range']} Hz "
                                      f"(need {tone_def['tone_a_length_ms']} ms)")
                        elif matching_freq:
                            print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                  f"Tone A frequency detected: {matching_freq:.1f} Hz "
                                  f"(target: {tone_def['tone_a']:.1f} Hz ±{tone_def['tone_a_range']} Hz) "
                                  f"[Hit streak: {self.tone_a_hit_streak[tone_id]}/{HIT_REQUIRED}]")
                    else:
                        is_tracking = self.tone_a_tracking.get(tone_id, False)
                        if is_tracking:
                            tracking_duration = current_time_ms - self.tone_a_tracking_start.get(tone_id, 0)
                            last_seen = self.tone_a_last_seen.get(tone_id, 0)
                            time_since_last_seen = current_time_ms - last_seen
                            if time_since_last_seen > GRACE_MS * 2:
                                self.tone_a_miss_streak[tone_id] = self.tone_a_miss_streak.get(tone_id, 0) + 1
                                if (self.tone_a_miss_streak[tone_id] >= MISS_REQUIRED and
                                    tracking_duration < tone_def["tone_a_length_ms"]):
                                    self.tone_a_tracking[tone_id] = False
                                    self.tone_a_tracking_start[tone_id] = 0
                                    self.tone_a_hit_streak[tone_id] = 0
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"Tone A tracking reset - frequency lost "
                                          f"(after {tracking_duration} ms, needed {tone_def['tone_a_length_ms']} ms, "
                                          f"last seen {time_since_last_seen} ms ago)")
                        else:
                            self.tone_a_miss_streak[tone_id] = self.tone_a_miss_streak.get(tone_id, 0) + 1
                            last_seen = self.tone_a_last_seen.get(tone_id, 0)
                            time_since_last_seen = current_time_ms - last_seen
                            if (time_since_last_seen > GRACE_MS and 
                                self.tone_a_miss_streak[tone_id] >= MISS_REQUIRED):
                                old_hit_streak = self.tone_a_hit_streak.get(tone_id, 0)
                                self.tone_a_hit_streak[tone_id] = 0
                                if old_hit_streak > 0:
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"Tone A hit streak reset (miss streak reached {MISS_REQUIRED})")
                    
                    if (self.tone_a_tracking.get(tone_id, False) and 
                        self.tone_a_tracking_start[tone_id] > 0):
                        duration = current_time_ms - self.tone_a_tracking_start[tone_id]
                        if duration >= tone_def["tone_a_length_ms"]:
                            self.tone_a_confirmed[tone_id] = True
                            matching_freq = next(
                                (f for f in peak_freqs 
                                 if is_frequency_in_range(f, tone_def["tone_a"], 
                                                         tone_def["tone_a_range"])),
                                None
                            )
                            print(f"[TONE CONFIRMED] Channel {self.channel_id}: "
                                  f"Tone A confirmed! "
                                  f"Frequency: {matching_freq:.1f} Hz "
                                  f"(target: {tone_def['tone_a']:.1f} Hz ±{tone_def['tone_a_range']} Hz), "
                                  f"Duration: {duration} ms "
                                  f"(required: {tone_def['tone_a_length_ms']} ms), "
                                  f"Tone ID: {tone_id}")
                
                elif not self.tone_b_confirmed.get(tone_id, False):
                    tone_b_detected = self._check_tone_frequency(
                        peak_freqs, tone_def["tone_b"], tone_def["tone_b_range"]
                    )
                    
                    if tone_b_detected:
                        self.tone_b_hit_streak[tone_id] = self.tone_b_hit_streak.get(tone_id, 0) + 1
                        self.tone_b_miss_streak[tone_id] = 0
                        self.tone_b_last_seen[tone_id] = current_time_ms
                        if self.tone_b_tracking.get(tone_id, False):
                            self.tone_b_miss_streak[tone_id] = 0
                        
                        matching_freq = next(
                            (f for f in peak_freqs 
                             if is_frequency_in_range(f, tone_def["tone_b"], 
                                                     tone_def["tone_b_range"])),
                            None
                        )
                        
                        if self.tone_b_hit_streak[tone_id] >= HIT_REQUIRED:
                            if not self.tone_b_tracking.get(tone_id, False):
                                self.tone_b_tracking[tone_id] = True
                                self.tone_b_tracking_start[tone_id] = current_time_ms
                                self.tone_b_miss_streak[tone_id] = 0
                                print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                      f"Tone B detected! Starting duration tracking for "
                                      f"{tone_def['tone_b']:.1f} Hz ±{tone_def['tone_b_range']} Hz "
                                      f"(need {tone_def['tone_b_length_ms']} ms)")
                        elif matching_freq:
                            print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                  f"Tone B frequency detected: {matching_freq:.1f} Hz "
                                  f"(target: {tone_def['tone_b']:.1f} Hz ±{tone_def['tone_b_range']} Hz) "
                                  f"[Hit streak: {self.tone_b_hit_streak[tone_id]}/{HIT_REQUIRED}]")
                    else:
                        is_tracking = self.tone_b_tracking.get(tone_id, False)
                        if is_tracking:
                            tracking_duration = current_time_ms - self.tone_b_tracking_start.get(tone_id, 0)
                            last_seen = self.tone_b_last_seen.get(tone_id, 0)
                            time_since_last_seen = current_time_ms - last_seen
                            if time_since_last_seen > GRACE_MS * 2:
                                self.tone_b_miss_streak[tone_id] = self.tone_b_miss_streak.get(tone_id, 0) + 1
                                if (self.tone_b_miss_streak[tone_id] >= MISS_REQUIRED and
                                    tracking_duration < tone_def["tone_b_length_ms"]):
                                    self.tone_b_tracking[tone_id] = False
                                    self.tone_b_tracking_start[tone_id] = 0
                                    self.tone_b_hit_streak[tone_id] = 0
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"Tone B tracking reset - frequency lost "
                                          f"(after {tracking_duration} ms, needed {tone_def['tone_b_length_ms']} ms, "
                                          f"last seen {time_since_last_seen} ms ago)")
                        else:
                            self.tone_b_miss_streak[tone_id] = self.tone_b_miss_streak.get(tone_id, 0) + 1
                            last_seen = self.tone_b_last_seen.get(tone_id, 0)
                            time_since_last_seen = current_time_ms - last_seen
                            if (time_since_last_seen > GRACE_MS and 
                                self.tone_b_miss_streak[tone_id] >= MISS_REQUIRED):
                                old_hit_streak = self.tone_b_hit_streak.get(tone_id, 0)
                                self.tone_b_hit_streak[tone_id] = 0
                                if old_hit_streak > 0:
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"Tone B hit streak reset (miss streak reached {MISS_REQUIRED})")
                    
                    if (self.tone_b_tracking.get(tone_id, False) and 
                        self.tone_b_tracking_start[tone_id] > 0):
                        duration = current_time_ms - self.tone_b_tracking_start[tone_id]
                        if duration >= tone_def["tone_b_length_ms"]:
                            self.tone_b_confirmed[tone_id] = True
                            matching_freq_b = next(
                                (f for f in peak_freqs 
                                 if is_frequency_in_range(f, tone_def["tone_b"], 
                                                         tone_def["tone_b_range"])),
                                None
                            )
                            print("\n" + "=" * 80)
                            print(" " * 20 + "*** TONE SEQUENCE DETECTED! ***")
                            print("=" * 80)
                            print(f"  Channel ID:     {self.channel_id}")
                            print(f"  Tone ID:        {tone_def['tone_id']}")
                            print("  ")
                            print("  Tone A Details:")
                            print(f"    Frequency:    {tone_def['tone_a']:.1f} Hz "
                                  f"±{tone_def['tone_a_range']} Hz")
                            print(f"    Duration:     {tone_def['tone_a_length_ms']} ms (required)")
                            print("  ")
                            print("  Tone B Details:")
                            print(f"    Detected:     {matching_freq_b:.1f} Hz")
                            print(f"    Target:       {tone_def['tone_b']:.1f} Hz "
                                  f"±{tone_def['tone_b_range']} Hz")
                            print(f"    Duration:     {duration} ms "
                                  f"(required: {tone_def['tone_b_length_ms']} ms)")
                            print("  ")
                            print(f"  Record Length:  {tone_def['record_length_ms']} ms")
                            if tone_def.get("detection_tone_alert"):
                                print(f"  Alert Type:     {tone_def['detection_tone_alert']}")
                            print("=" * 80 + "\n")
                            
                            self.tone_a_confirmed[tone_id] = False
                            self.tone_b_confirmed[tone_id] = False
                            self.tone_a_tracking[tone_id] = False
                            self.tone_b_tracking[tone_id] = False
                            self.tone_a_tracking_start[tone_id] = 0
                            self.tone_b_tracking_start[tone_id] = 0
                            self.tone_a_hit_streak[tone_id] = 0
                            self.tone_b_hit_streak[tone_id] = 0
                            
                            return tone_def
        
        return None


_channel_detectors: Dict[str, ChannelToneDetector] = {}
_detectors_mutex = threading.Lock()


def init_channel_detector(channel_id: str, tone_definitions: List[Dict[str, Any]]) -> None:
    """Initialize tone detector for a channel"""
    with _detectors_mutex:
        if tone_definitions:
            _channel_detectors[channel_id] = ChannelToneDetector(channel_id, tone_definitions)
            print(f"[TONE DETECTION] Initialized detector for channel {channel_id} "
                  f"with {len(tone_definitions)} tone definition(s)")
            for i, tone_def in enumerate(tone_definitions, 1):
                print(f"[TONE DETECTION]   Definition {i}:")
                print(f"    Tone ID: {tone_def.get('tone_id', 'N/A')}")
                print(f"    Tone A: {tone_def.get('tone_a', 0):.1f} Hz "
                      f"±{tone_def.get('tone_a_range', 0)} Hz "
                      f"({tone_def.get('tone_a_length_ms', 0)} ms)")
                print(f"    Tone B: {tone_def.get('tone_b', 0):.1f} Hz "
                      f"±{tone_def.get('tone_b_range', 0)} Hz "
                      f"({tone_def.get('tone_b_length_ms', 0)} ms)")
                print(f"    Record Length: {tone_def.get('record_length_ms', 0)} ms")
        else:
            _channel_detectors.pop(channel_id, None)
            print(f"[TONE DETECTION] No tone definitions found for channel {channel_id}")


def process_audio_for_channel(channel_id: str, filtered_audio: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Process filtered audio samples for a specific channel.
    Returns tone definition dict if sequence detected, None otherwise.
    """
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if not detector:
            return None
    
    detector.add_audio_samples(filtered_audio)
    return detector.process_audio()

