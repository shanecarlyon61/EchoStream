import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from numpy.fft import rfft

try:
    from communication.mqtt import publish_known_tone_detection, publish_new_tone_pair
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    def publish_known_tone_detection(*args, **kwargs):
        return False
    def publish_new_tone_pair(*args, **kwargs):
        return False


SAMPLE_RATE = 48000
FFT_SIZE = 1024
FREQ_BINS = FFT_SIZE // 2
HIT_REQUIRED = 1
MISS_REQUIRED = 3
GRACE_MS = 500


def parabolic(f, x):
    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    try:
        xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
        yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
        return xv, yv
    except (ZeroDivisionError, IndexError):
        return float(x), float(f[x])


def frequency_to_bin(frequency: float) -> float:
    return (frequency * FFT_SIZE) / SAMPLE_RATE


def is_frequency_in_range(detected_freq: float, target_freq: float, range_hz: int) -> bool:
    return abs(detected_freq - target_freq) <= range_hz


class ChannelToneDetector:
    def __init__(self, channel_id: str, tone_definitions: List[Dict[str, Any]], new_tone_config: Optional[Dict[str, Any]] = None, passthrough_config: Optional[Dict[str, Any]] = None):
        self.channel_id = channel_id
        self.tone_definitions = tone_definitions
        self.audio_buffer: List[float] = []
        self.max_buffer_samples = int(SAMPLE_RATE * 10)
        self.mutex = threading.Lock()
        
        if new_tone_config is None:
            new_tone_config = {"detect_new_tones": False, "new_tone_length_ms": 1000, "new_tone_range_hz": 3}
        self.detect_new_tones = new_tone_config.get("detect_new_tones", False)
        self.new_tone_length_ms = new_tone_config.get("new_tone_length_ms", 1000)
        self.new_tone_range_hz = new_tone_config.get("new_tone_range_hz", 3)
        
        if passthrough_config is None:
            passthrough_config = {"tone_passthrough": False, "passthrough_channel": ""}
        self.passthrough_config = passthrough_config
        
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
        self.tone_a_confirmed_end: Dict[str, int] = {}  # Track when Tone A ended (for sequence validation)
        self.intervening_tones_detected: Dict[str, bool] = {}  # Track if other tones appeared between A and B
        
        self.new_tone_tracking: List[Dict[str, Any]] = []
        for i in range(10):
            self.new_tone_tracking.append({
                "is_tracking": False,
                "frequency": 0.0,
                "tracking_start": 0,
                "hit_streak": 0,
                "miss_streak": 0,
                "last_seen": 0
            })
        self.detected_frequencies: List[float] = []
        
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
            self.tone_a_confirmed_end[tone_id] = 0
            self.intervening_tones_detected[tone_id] = False
        
    def add_audio_samples(self, samples: np.ndarray):
        with self.mutex:
            self.audio_buffer.extend(samples.tolist())
            if len(self.audio_buffer) > self.max_buffer_samples:
                self.audio_buffer = self.audio_buffer[-self.max_buffer_samples:]
    
    def _find_peak_frequencies(self, audio_samples: np.ndarray) -> List[float]:
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
        for peak_freq in peak_freqs:
            if is_frequency_in_range(peak_freq, target_freq, range_hz):
                return True
        return False
    
    def _check_for_intervening_tones(self, peak_freqs: List[float], tone_def: Dict[str, Any], tone_id: str) -> bool:
        """Check if any other tones (from other definitions) appear in current peak frequencies.
        This is used to invalidate sequences where other tones appear between Tone A and Tone B."""
        for other_tone_def in self.tone_definitions:
            if other_tone_def["tone_id"] == tone_id:
                continue
            
            # Check if Tone A or Tone B from other definitions appear
            other_tone_a_detected = self._check_tone_frequency(
                peak_freqs, other_tone_def["tone_a"], other_tone_def["tone_a_range"]
            )
            other_tone_b_detected = self._check_tone_frequency(
                peak_freqs, other_tone_def["tone_b"], other_tone_def["tone_b_range"]
            )
            
            if other_tone_a_detected or other_tone_b_detected:
                return True
        
        return False
    
    def process_audio(self) -> Optional[Dict[str, Any]]:
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
                        # If Tone B appears before Tone A is confirmed, reset Tone A tracking
                        self.tone_a_hit_streak[tone_id] = 0
                        self.tone_a_tracking[tone_id] = False
                        self.tone_a_tracking_start[tone_id] = 0
                        self.tone_a_confirmed_end[tone_id] = 0
                        self.intervening_tones_detected[tone_id] = False
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
                            required_duration = tone_def["tone_a_length_ms"]
                            max_gap_allowed = max(GRACE_MS * 4, required_duration // 2)
                            if time_since_last_seen > max_gap_allowed and tracking_duration < required_duration:
                                self.tone_a_miss_streak[tone_id] = self.tone_a_miss_streak.get(tone_id, 0) + 1
                                if self.tone_a_miss_streak[tone_id] >= MISS_REQUIRED:
                                    self.tone_a_tracking[tone_id] = False
                                    self.tone_a_tracking_start[tone_id] = 0
                                    self.tone_a_hit_streak[tone_id] = 0
                                    self.tone_a_confirmed_end[tone_id] = 0
                                    self.intervening_tones_detected[tone_id] = False
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"Tone A tracking reset - frequency lost "
                                          f"(after {tracking_duration} ms, needed {required_duration} ms, "
                                          f"last seen {time_since_last_seen} ms ago, "
                                          f"max gap: {max_gap_allowed} ms)")
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
                            self.tone_a_confirmed_end[tone_id] = current_time_ms  # Mark when Tone A ended
                            self.intervening_tones_detected[tone_id] = False  # Reset intervening tones flag
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
                    # Check for intervening tones (other tones between A and B)
                    if self.tone_a_confirmed.get(tone_id, False):
                        has_intervening_tones = self._check_for_intervening_tones(peak_freqs, tone_def, tone_id)
                        if has_intervening_tones:
                            self.intervening_tones_detected[tone_id] = True
                            print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                  f"Intervening tone detected between Tone A and Tone B for {tone_id}! "
                                  f"Sequence will be invalidated.")
                    
                    tone_b_detected = self._check_tone_frequency(
                        peak_freqs, tone_def["tone_b"], tone_def["tone_b_range"]
                    )
                    
                    if tone_b_detected:
                        self.tone_b_hit_streak[tone_id] = self.tone_b_hit_streak.get(tone_id, 0) + 1
                        self.tone_b_miss_streak[tone_id] = 0
                        self.tone_b_last_seen[tone_id] = current_time_ms
                        if self.tone_b_tracking.get(tone_id, False):
                            self.tone_b_miss_streak[tone_id] = 0
                            
                            duration = current_time_ms - self.tone_b_tracking_start.get(tone_id, 0)
                            if duration >= tone_def["tone_b_length_ms"]:
                                # Check if intervening tones were detected - if so, invalidate sequence
                                if self.intervening_tones_detected.get(tone_id, False):
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"INVALID SEQUENCE for {tone_id}! "
                                          f"Other tones detected between Tone A and Tone B. "
                                          f"Sequence rejected.")
                                    # Reset all state for this tone
                                    self.tone_a_confirmed[tone_id] = False
                                    self.tone_b_confirmed[tone_id] = False
                                    self.tone_a_tracking[tone_id] = False
                                    self.tone_b_tracking[tone_id] = False
                                    self.tone_a_tracking_start[tone_id] = 0
                                    self.tone_b_tracking_start[tone_id] = 0
                                    self.tone_a_hit_streak[tone_id] = 0
                                    self.tone_b_hit_streak[tone_id] = 0
                                    self.tone_a_miss_streak[tone_id] = 0
                                    self.tone_b_miss_streak[tone_id] = 0
                                    self.tone_a_last_seen[tone_id] = 0
                                    self.tone_b_last_seen[tone_id] = 0
                                    self.tone_a_confirmed_end[tone_id] = 0
                                    self.intervening_tones_detected[tone_id] = False
                                    continue
                                
                                self.tone_b_confirmed[tone_id] = True
                                matching_freq_b = next(
                                    (f for f in peak_freqs 
                                     if is_frequency_in_range(f, tone_def["tone_b"], 
                                                             tone_def["tone_b_range"])),
                                    None
                                )
                                alert_type_line = ""
                                if tone_def.get("detection_tone_alert"):
                                    alert_type_line = f"  Alert Type:     {tone_def['detection_tone_alert']}\n"
                                
                                confirmation_log = (
                                    "\n" + "=" * 80 + "\n" +
                                    " " * 20 + "*** TONE SEQUENCE DETECTED! ***\n" +
                                    "=" * 80 + "\n" +
                                    f"  Channel ID:     {self.channel_id}\n" +
                                    f"  Tone ID:        {tone_def['tone_id']}\n" +
                                    "  \n" +
                                    "  Tone A Details:\n" +
                                    f"    Frequency:    {tone_def['tone_a']:.1f} Hz "
                                    f"±{tone_def['tone_a_range']} Hz\n" +
                                    f"    Duration:     {tone_def['tone_a_length_ms']} ms (required)\n" +
                                    "  \n" +
                                    "  Tone B Details:\n" +
                                    f"    Detected:     {matching_freq_b:.1f} Hz\n" +
                                    f"    Target:       {tone_def['tone_b']:.1f} Hz "
                                    f"±{tone_def['tone_b_range']} Hz\n" +
                                    f"    Duration:     {duration} ms "
                                    f"(required: {tone_def['tone_b_length_ms']} ms)\n" +
                                    "  \n" +
                                    f"  Record Length:  {tone_def['record_length_ms']} ms\n" +
                                    alert_type_line +
                                    "=" * 80 + "\n"
                                )
                                print(confirmation_log, flush=True)
                                
                                if MQTT_AVAILABLE:
                                    publish_known_tone_detection(
                                        tone_id=tone_def["tone_id"],
                                        tone_a_hz=tone_def["tone_a"],
                                        tone_b_hz=tone_def["tone_b"],
                                        tone_a_duration_ms=tone_def["tone_a_length_ms"],
                                        tone_b_duration_ms=duration,
                                        tone_a_range_hz=tone_def["tone_a_range"],
                                        tone_b_range_hz=tone_def["tone_b_range"],
                                        channel_id=self.channel_id,
                                        record_length_ms=tone_def.get("record_length_ms", 0),
                                        detection_tone_alert=tone_def.get("detection_tone_alert")
                                    )
                                
                                try:
                                    from processing.passthrough import global_passthrough_manager
                                    if hasattr(self, 'passthrough_config') and self.passthrough_config.get("tone_passthrough", False):
                                        target_channel = self.passthrough_config.get("passthrough_channel", "")
                                        record_length_ms = tone_def.get("record_length_ms", 0)
                                        if target_channel and record_length_ms > 0:
                                            global_passthrough_manager.start_passthrough(
                                                self.channel_id, target_channel, record_length_ms
                                            )
                                            print(f"[PASSTHROUGH] Triggered: {self.channel_id} -> {target_channel}, duration={record_length_ms} ms")
                                except Exception as e:
                                    print(f"[PASSTHROUGH] ERROR: Failed to trigger passthrough: {e}")
                                
                                try:
                                    from processing.recording import global_recording_manager
                                    record_length_ms = tone_def.get("record_length_ms", 0)
                                    if record_length_ms > 0:
                                        global_recording_manager.start_recording(
                                            self.channel_id, "defined",
                                            tone_def["tone_a"], tone_def["tone_b"], record_length_ms
                                        )
                                except Exception as e:
                                    print(f"[RECORDING] ERROR: Failed to start recording: {e}")
                                
                                self.tone_a_confirmed[tone_id] = False
                                self.tone_b_confirmed[tone_id] = False
                                self.tone_a_tracking[tone_id] = False
                                self.tone_b_tracking[tone_id] = False
                                self.tone_a_tracking_start[tone_id] = 0
                                self.tone_b_tracking_start[tone_id] = 0
                                self.tone_a_hit_streak[tone_id] = 0
                                self.tone_b_hit_streak[tone_id] = 0
                                self.tone_a_miss_streak[tone_id] = 0
                                self.tone_b_miss_streak[tone_id] = 0
                                self.tone_a_last_seen[tone_id] = 0
                                self.tone_b_last_seen[tone_id] = 0
                                self.tone_a_confirmed_end[tone_id] = 0
                                self.intervening_tones_detected[tone_id] = False
                                
                                return tone_def
                        
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
                            required_duration = tone_def["tone_b_length_ms"]
                            max_gap_allowed = max(GRACE_MS * 4, required_duration // 2)
                            if time_since_last_seen > max_gap_allowed and tracking_duration < required_duration:
                                self.tone_b_miss_streak[tone_id] = self.tone_b_miss_streak.get(tone_id, 0) + 1
                                if self.tone_b_miss_streak[tone_id] >= MISS_REQUIRED:
                                    self.tone_b_tracking[tone_id] = False
                                    self.tone_b_tracking_start[tone_id] = 0
                                    self.tone_b_hit_streak[tone_id] = 0
                                    # If Tone B tracking is lost, also reset Tone A confirmed state to restart sequence
                                    if self.tone_a_confirmed.get(tone_id, False):
                                        self.tone_a_confirmed[tone_id] = False
                                        self.tone_a_confirmed_end[tone_id] = 0
                                        self.intervening_tones_detected[tone_id] = False
                                        print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                              f"Tone B tracking lost - resetting Tone A confirmed state")
                                    print(f"[TONE DETECTION] Channel {self.channel_id}: "
                                          f"Tone B tracking reset - frequency lost "
                                          f"(after {tracking_duration} ms, needed {required_duration} ms, "
                                          f"last seen {time_since_last_seen} ms ago, "
                                          f"max gap: {max_gap_allowed} ms)")
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
            
            if self.detect_new_tones:
                self._detect_new_tones(peak_freqs, current_time_ms)
        
        return None
    
    def _detect_new_tones(self, peak_freqs: List[float], current_time_ms: int):
        if not self.detect_new_tones:
            return
        
        confirmed_indices = []
        confirmed_freqs = []
        
        for i, freq in enumerate(peak_freqs):
            is_known_tone = False
            
            for tone_def in self.tone_definitions:
                if (is_frequency_in_range(freq, tone_def["tone_a"], tone_def["tone_a_range"]) or
                    is_frequency_in_range(freq, tone_def["tone_b"], tone_def["tone_b_range"])):
                    is_known_tone = True
                    break
            
            if is_known_tone:
                continue
            
            already_confirmed = False
            for detected_freq in self.detected_frequencies:
                if abs(detected_freq - freq) < self.new_tone_range_hz:
                    already_confirmed = True
                    break
            
            if already_confirmed:
                continue
            
            tracking_idx = -1
            for t in range(10):
                if self.new_tone_tracking[t]["is_tracking"]:
                    if abs(self.new_tone_tracking[t]["frequency"] - freq) < self.new_tone_range_hz:
                        tracking_idx = t
                        break
                elif tracking_idx == -1:
                    tracking_idx = t
            
            if tracking_idx >= 0:
                if not self.new_tone_tracking[tracking_idx]["is_tracking"]:
                    self.new_tone_tracking[tracking_idx]["frequency"] = freq
                    self.new_tone_tracking[tracking_idx]["is_tracking"] = True
                    self.new_tone_tracking[tracking_idx]["tracking_start"] = current_time_ms
                    self.new_tone_tracking[tracking_idx]["hit_streak"] = 1
                    self.new_tone_tracking[tracking_idx]["miss_streak"] = 0
                    self.new_tone_tracking[tracking_idx]["last_seen"] = current_time_ms
                else:
                    self.new_tone_tracking[tracking_idx]["hit_streak"] += 1
                    self.new_tone_tracking[tracking_idx]["miss_streak"] = 0
                    self.new_tone_tracking[tracking_idx]["last_seen"] = current_time_ms
                    
                    avg_freq = (self.new_tone_tracking[tracking_idx]["frequency"] + freq) / 2.0
                    if abs(avg_freq - self.new_tone_tracking[tracking_idx]["frequency"]) < self.new_tone_range_hz:
                        self.new_tone_tracking[tracking_idx]["frequency"] = avg_freq
                    
                    elapsed = current_time_ms - self.new_tone_tracking[tracking_idx]["tracking_start"]
                    if elapsed >= self.new_tone_length_ms:
                        if len(confirmed_indices) < 10:
                            confirmed_indices.append(tracking_idx)
                            confirmed_freqs.append(self.new_tone_tracking[tracking_idx]["frequency"])
            
            if i == len(peak_freqs) - 1:
                if len(confirmed_indices) >= 2:
                    idx_a = confirmed_indices[0]
                    idx_b = confirmed_indices[1]
                    tone_a = confirmed_freqs[0]
                    tone_b = confirmed_freqs[1]
                    
                    if self.new_tone_tracking[idx_b]["tracking_start"] < self.new_tone_tracking[idx_a]["tracking_start"]:
                        idx_a, idx_b = idx_b, idx_a
                        tone_a, tone_b = tone_b, tone_a
                    
                    # Check if there are other tones (confirmed or tracking) between tone_a and tone_b
                    # A valid sequence requires tone B to immediately follow tone A with no other tones in between
                    tone_a_end_time = self.new_tone_tracking[idx_a]["tracking_start"] + self.new_tone_length_ms
                    tone_b_start_time = self.new_tone_tracking[idx_b]["tracking_start"]
                    
                    has_intervening_tones = False
                    # Check all tracking tones (confirmed or not) for intervening tones
                    # A valid sequence requires tone B to start immediately after tone A ends, with no other tones in between
                    for t in range(10):
                        if not self.new_tone_tracking[t]["is_tracking"]:
                            continue
                        if t == idx_a or t == idx_b:
                            continue
                        
                        other_start_time = self.new_tone_tracking[t]["tracking_start"]
                        # If another tone starts between tone_a ending and tone_b starting, it's intervening
                        if tone_a_end_time <= other_start_time <= tone_b_start_time:
                            has_intervening_tones = True
                            break
                    
                    if has_intervening_tones:
                        print(f"[NEW TONE PAIR] Channel {self.channel_id}: "
                              f"INVALID SEQUENCE! Other tones detected between "
                              f"A={tone_a:.1f} Hz and B={tone_b:.1f} Hz. "
                              f"Sequence rejected.")
                        # Reset these two tones
                        self.new_tone_tracking[idx_a]["is_tracking"] = False
                        self.new_tone_tracking[idx_a]["tracking_start"] = 0
                        self.new_tone_tracking[idx_a]["hit_streak"] = 0
                        self.new_tone_tracking[idx_a]["miss_streak"] = 0
                        self.new_tone_tracking[idx_b]["is_tracking"] = False
                        self.new_tone_tracking[idx_b]["tracking_start"] = 0
                        self.new_tone_tracking[idx_b]["hit_streak"] = 0
                        self.new_tone_tracking[idx_b]["miss_streak"] = 0
                    else:
                        print(f"[NEW TONE PAIR] Channel {self.channel_id}: "
                              f"A={tone_a:.1f} Hz, B={tone_b:.1f} Hz "
                              f"(each ≥ {self.new_tone_length_ms} ms, ±{self.new_tone_range_hz} Hz stable)")
                        
                        if len(self.detected_frequencies) < 100:
                            self.detected_frequencies.append(tone_a)
                        if len(self.detected_frequencies) < 100:
                            self.detected_frequencies.append(tone_b)
                        
                        if MQTT_AVAILABLE:
                            publish_new_tone_pair(tone_a, tone_b)
                        
                        try:
                            from recording import global_recording_manager
                            if hasattr(self, 'new_tone_config') and self.new_tone_config:
                                new_tone_length_ms = self.new_tone_config.get("new_tone_length_ms", 0)
                                if new_tone_length_ms > 0:
                                    global_recording_manager.start_recording(
                                        self.channel_id, "new", tone_a, tone_b, new_tone_length_ms
                                    )
                        except Exception as e:
                            print(f"[RECORDING] ERROR: Failed to start new tone recording: {e}")
                        
                        self.new_tone_tracking[idx_a]["is_tracking"] = False
                        self.new_tone_tracking[idx_a]["tracking_start"] = 0
                        self.new_tone_tracking[idx_a]["hit_streak"] = 0
                        self.new_tone_tracking[idx_a]["miss_streak"] = 0
                        self.new_tone_tracking[idx_b]["is_tracking"] = False
                        self.new_tone_tracking[idx_b]["tracking_start"] = 0
                        self.new_tone_tracking[idx_b]["hit_streak"] = 0
                        self.new_tone_tracking[idx_b]["miss_streak"] = 0
        
        for t in range(10):
            if self.new_tone_tracking[t]["is_tracking"]:
                found = False
                for peak_freq in peak_freqs:
                    if abs(self.new_tone_tracking[t]["frequency"] - peak_freq) < self.new_tone_range_hz:
                        found = True
                        break
                
                if not found:
                    self.new_tone_tracking[t]["miss_streak"] += 1
                    time_since_last_seen = current_time_ms - self.new_tone_tracking[t]["last_seen"]
                    if time_since_last_seen > GRACE_MS and self.new_tone_tracking[t]["miss_streak"] >= MISS_REQUIRED:
                        self.new_tone_tracking[t]["is_tracking"] = False
                        self.new_tone_tracking[t]["tracking_start"] = 0
                        self.new_tone_tracking[t]["hit_streak"] = 0
                        self.new_tone_tracking[t]["miss_streak"] = 0


_channel_detectors: Dict[str, ChannelToneDetector] = {}
_detectors_mutex = threading.Lock()


def init_channel_detector(channel_id: str, tone_definitions: List[Dict[str, Any]], new_tone_config: Optional[Dict[str, Any]] = None, passthrough_config: Optional[Dict[str, Any]] = None) -> None:
    with _detectors_mutex:
        if tone_definitions or (new_tone_config and new_tone_config.get("detect_new_tones", False)):
            _channel_detectors[channel_id] = ChannelToneDetector(channel_id, tone_definitions, new_tone_config, passthrough_config)
            print(f"[TONE DETECTION] Initialized detector for channel {channel_id} "
                  f"with {len(tone_definitions)} tone definition(s)")
            if new_tone_config and new_tone_config.get("detect_new_tones", False):
                print(f"[TONE DETECTION] New tone detection enabled: "
                      f"length={new_tone_config.get('new_tone_length_ms', 1000)} ms, "
                      f"range=±{new_tone_config.get('new_tone_range_hz', 3)} Hz")
            if passthrough_config and passthrough_config.get("tone_passthrough", False):
                print(f"[TONE DETECTION] Passthrough enabled: target={passthrough_config.get('passthrough_channel', 'N/A')}")
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
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if not detector:
            return None
    
    detector.add_audio_samples(filtered_audio)
    return detector.process_audio()

