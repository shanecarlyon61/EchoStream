import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from numpy.fft import rfft, rfftfreq

try:
    from scipy.signal import hanning
except ImportError:
    from numpy import hanning

try:
    from mqtt_client import publish_known_tone_detection, publish_new_tone_pair

    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

    def publish_known_tone_detection(*args, **kwargs):
        return False

    def publish_new_tone_pair(*args, **kwargs):
        return False

SAMPLE_RATE = 48000

MAX_BUFFER_SECONDS = 10
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)

MIN_DETECTION_INTERVAL_SECONDS = 2.0

DURATION_TOLERANCE_MS = 80

def parabolic(f, x):
    
    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    try:
        xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
        yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
        return xv, yv
    except (ZeroDivisionError, IndexError):
        return float(x), float(f[x])

def calculate_peak_prominence(
    magnitudes: np.ndarray, peak_idx: int, window_size: int = 5
) -> float:
    
    peak_mag = magnitudes[peak_idx]

    left_start = max(0, peak_idx - window_size)
    left_valley = (
        np.min(magnitudes[left_start:peak_idx]) if peak_idx > 0 else peak_mag
    )

    right_end = min(len(magnitudes), peak_idx + window_size + 1)
    right_valley = (
        np.min(magnitudes[peak_idx + 1 : right_end])
        if peak_idx < len(magnitudes) - 1
        else peak_mag
    )

    prominence = peak_mag - max(left_valley, right_valley)

    return prominence

def calculate_peak_width(
    magnitudes: np.ndarray, peak_idx: int, prominence_fraction: float = 0.5
) -> int:
    
    peak_mag = magnitudes[peak_idx]
    prominence = calculate_peak_prominence(magnitudes, peak_idx)
    threshold = peak_mag - (prominence * prominence_fraction)

    left_edge = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if magnitudes[i] < threshold:
            left_edge = i + 1
            break
    else:
        left_edge = 0

    right_edge = peak_idx
    for i in range(peak_idx + 1, len(magnitudes)):
        if magnitudes[i] < threshold:
            right_edge = i - 1
            break
    else:
        right_edge = len(magnitudes) - 1

    return right_edge - left_edge + 1

def freq_from_fft(
    sig: np.ndarray,
    fs: int = SAMPLE_RATE,
    magnitude_threshold: float = 300,
    min_prominence_ratio: float = 0.3,
    max_width_bins: int = 20,
    min_separation_hz: float = 50.0,  # Increased from 20.0 to better filter harmonics
    min_freq: float = 50.0,  # Minimum frequency to consider (filters out low-frequency noise)
    max_freq: float = 4000.0,  # Maximum frequency to consider
) -> List[float]:
    
    if len(sig) == 0:
        return []

    windowed = sig * hanning(len(sig))
    f = rfft(windowed)

    magnitudes = np.abs(f)
    max_magnitude = np.max(magnitudes)
    if max_magnitude == 0:
        return []
    
    # Additional check: if the signal is too quiet, return empty
    # This prevents detecting tones from noise/empty audio
    signal_rms = np.sqrt(np.mean(sig**2))
    if signal_rms < 1e-6:  # Very quiet signal threshold
        return []

    candidate_peaks = []
    for i in range(1, len(magnitudes) - 1):
        if (
            magnitudes[i] > magnitude_threshold
            and magnitudes[i] > magnitudes[i - 1]
            and magnitudes[i] > magnitudes[i + 1]
        ):
            candidate_peaks.append(i)

    if len(candidate_peaks) == 0:
        return []
    
    peak_data = []
    for peak_idx in candidate_peaks:
        true_i = parabolic(np.log(magnitudes + 1e-10), peak_idx)[0]
        peak_freq = fs * true_i / len(windowed)
        
        # Apply frequency range filter to exclude low-frequency noise and very high frequencies
        if peak_freq < min_freq or peak_freq > max_freq:
            continue
        
        prominence = calculate_peak_prominence(magnitudes, peak_idx)
        width = calculate_peak_width(magnitudes, peak_idx)
        peak_mag = magnitudes[peak_idx]

        # Apply prominence filter to remove weak peaks
        min_prominence = peak_mag * min_prominence_ratio
        if prominence < min_prominence:
            continue

        # Apply width filter to remove overly broad peaks
        if width > max_width_bins:
            continue

        peak_data.append(
            {
                "index": peak_idx,
                "freq": peak_freq,
                "magnitude": peak_mag,
                "prominence": prominence,
                "width": width,
            }
        )

    if len(peak_data) == 0:
        return []

    peak_data.sort(key=lambda x: x["magnitude"], reverse=True)

    filtered_peaks = []
    min_separation_bins = int((min_separation_hz * len(windowed)) / fs)

    for peak in peak_data:

        too_close = False
        for accepted_peak in filtered_peaks:
            bin_separation = abs(peak["index"] - accepted_peak["index"])
            if bin_separation < min_separation_bins:
                too_close = True
                break

        if not too_close:
            filtered_peaks.append(peak)
            print(f"magnitude: {peak['magnitude']}")
    
    if len(filtered_peaks) > 0:
        filtered_peaks = [filtered_peaks[0]]  # Keep only the strongest peak
    
    peaks = [p["freq"] for p in filtered_peaks]

    return peaks

def is_frequency_in_range(
    detected_freq: float, target_freq: float, range_hz: int
) -> bool:
    
    return abs(detected_freq - target_freq) <= range_hz

def calculate_rms_volume(samples: np.ndarray) -> float:
    
    if len(samples) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(samples**2))
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms)

def count_significant_peaks(
    sig: np.ndarray,
    fs: int = SAMPLE_RATE,
    peak_threshold: float = 0.3,
    min_freq: float = 50.0,
    max_freq: float = 4000.0,
    min_peak_separation_hz: float = 20.0,
) -> int:
    
    if len(sig) == 0:
        return 0

    windowed = sig * hanning(len(sig))
    f = rfft(windowed)
    magnitudes = np.abs(f)

    if np.max(magnitudes) == 0:
        return 0

    magnitudes_norm = magnitudes / np.max(magnitudes)

    freqs = np.fft.rfftfreq(len(windowed), 1.0 / fs)

    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    magnitudes_filtered = magnitudes_norm[freq_mask]
    freqs_filtered = freqs[freq_mask]

    if len(magnitudes_filtered) < 3:
        return 0

    peaks = []
    for i in range(1, len(magnitudes_filtered) - 1):

        if (
            magnitudes_filtered[i] > magnitudes_filtered[i - 1]
            and magnitudes_filtered[i] > magnitudes_filtered[i + 1]
            and magnitudes_filtered[i] >= peak_threshold
        ):
            peaks.append((freqs_filtered[i], magnitudes_filtered[i]))

    if len(peaks) > 1:
        peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
        filtered_peaks = [peaks_sorted[0]]

        for freq, mag in peaks_sorted[1:]:

            min_distance = min(abs(freq - fp[0]) for fp in filtered_peaks)
            if min_distance > min_peak_separation_hz:
                filtered_peaks.append((freq, mag))

        return len(filtered_peaks)

    return len(peaks)

class ChannelToneDetector:
    def __init__(
        self,
        channel_id: str,
        tone_definitions: List[Dict[str, Any]],
        new_tone_config: Optional[Dict[str, Any]] = None,
        passthrough_config: Optional[Dict[str, Any]] = None,
        frequency_filters: Optional[List[Dict[str, Any]]] = None,
    ):
        self.channel_id = channel_id
        self.tone_definitions = tone_definitions
        self.audio_buffer: List[float] = []
        self.mutex = threading.Lock()

        self.frequency_filters = frequency_filters or []

        if new_tone_config is None:
            new_tone_config = {
                "detect_new_tones": False,
                "new_tone_length_ms": 1000,
                "new_tone_range_hz": 3,
            }
        self.detect_new_tones = new_tone_config.get("detect_new_tones", False)
        self.new_tone_length_ms = new_tone_config.get("new_tone_length_ms", 1000)
        self.new_tone_length_seconds = self.new_tone_length_ms / 1000.0
        self.new_tone_range_hz = new_tone_config.get("new_tone_range_hz", 3)

        self.new_tone_length_tolerance_ms = new_tone_config.get(
            "new_tone_length_tolerance_ms", 50
        )
        self.new_tone_length_tolerance_seconds = (
            self.new_tone_length_tolerance_ms / 1000.0
        )

        self.new_tone_consecutive_required = 3
        self.new_tone_config = new_tone_config

        if passthrough_config is None:
            passthrough_config = {"tone_passthrough": False, "passthrough_channel": ""}
        self.passthrough_config = passthrough_config

        self.last_detection_time: Dict[str, float] = {}
        for tone_def in tone_definitions:
            tone_id = tone_def["tone_id"]
            self.last_detection_time[tone_id] = 0.0

        self.last_new_tone_detection_time = 0.0

        self.new_tone_pair_count = 0
        self.last_new_tone_pair: Optional[Dict[str, Any]] = None

        self.last_detected_new_pair: Optional[Dict[str, Any]] = None

        self.frequency_history: Dict[str, List[Dict[str, Any]]] = {}

        self.stability_required_count = 2
        self.stability_tolerance_hz = 10.0
        self.stability_max_age_seconds = 0.3

    def add_audio_samples(self, samples: np.ndarray):
        
        with self.mutex:

            if len(samples) < 10000:

                self.audio_buffer.extend(samples.tolist())
            else:

                chunk_size = 5000
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    self.audio_buffer.extend(chunk.tolist())
            if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
                self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

    def _get_buffer_array(self) -> np.ndarray:
        
        with self.mutex:
            buffer_copy = list(self.audio_buffer)

        return np.array(buffer_copy, dtype=np.float32)

    def _check_frequency_stability(
        self, frequency: float, window_id: str, current_time: float
    ) -> bool:
        if window_id not in self.frequency_history:
            self.frequency_history[window_id] = []

        history = self.frequency_history[window_id]

        history[:] = [
            entry
            for entry in history
            if (current_time - entry["time"]) <= self.stability_max_age_seconds
        ]

        history.append({"freq": frequency, "time": current_time})

        if len(history) < self.stability_required_count:
            return False

        recent_detections = history[-self.stability_required_count :]
        frequencies = [d["freq"] for d in recent_detections]

        mean_freq = np.mean(frequencies)
        max_deviation = max(abs(f - mean_freq) for f in frequencies)

        return max_deviation <= self.stability_tolerance_hz

    def _is_frequency_filtered(self, frequency: float) -> bool:
        
        if not self.frequency_filters:
            return False

        for filter_data in self.frequency_filters:
            filter_freq = filter_data.get("frequency", 0.0)
            filter_type = filter_data.get("type", "")

            if filter_type == "below":

                if frequency < filter_freq:
                    return True
            elif filter_type == "above":

                if frequency > filter_freq:
                    return True

        return False

    def _detect_defined_tone(self, tone_def: Dict[str, Any]) -> bool:
        
        import random

        tone_a_length_seconds = tone_def["tone_a_length_ms"] / 1000.0
        tone_b_length_seconds = tone_def["tone_b_length_ms"] / 1000.0

        tone_a_samples = int(tone_a_length_seconds * SAMPLE_RATE)
        tone_b_samples = int(tone_b_length_seconds * SAMPLE_RATE)
        total_samples = tone_a_samples + tone_b_samples

        with self.mutex:
            buffer_len = len(self.audio_buffer)

        if buffer_len < total_samples:
            # if random.randint(1, 100) == 1:
            #     print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_def['tone_id']}: Buffer too small - {buffer_len} samples (need {total_samples})")
            return False

        with self.mutex:
            recent_samples_list = self.audio_buffer[-total_samples:]

        recent_samples = np.array(recent_samples_list, dtype=np.float32)
        volume_db = calculate_rms_volume(recent_samples)

        # Skip tone detection if audio is too quiet (empty/silent audio)
        MIN_VOLUME_THRESHOLD_DB = -30.0
        if volume_db < MIN_VOLUME_THRESHOLD_DB:
            if random.randint(1, 100) == 1:
                print(f"[TONE DETECT] Channel {self.channel_id}: Volume too low ({volume_db:.1f} dB < {MIN_VOLUME_THRESHOLD_DB} dB), skipping tone detection")
            return False

        if random.randint(1, 50) == 1:
            print(f"[TONE DETECT] Channel {self.channel_id}: Volume={volume_db:.1f} dB (threshold={MIN_VOLUME_THRESHOLD_DB}dB)")

        tone_id = tone_def["tone_id"]
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time.get(tone_id, 0)
        if time_since_last < MIN_DETECTION_INTERVAL_SECONDS:
            # if random.randint(1, 50) == 1:
            #     print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Too soon since last detection - {time_since_last:.1f}s (min {MIN_DETECTION_INTERVAL_SECONDS}s)")
            return False

        try:

            buffer_end_samples = len(self.audio_buffer)
            tone_a_end_samples = buffer_end_samples - (total_samples - tone_a_samples)
            tone_b_end_samples = buffer_end_samples
            tone_a_position_seconds = tone_a_end_samples / SAMPLE_RATE
            tone_b_position_seconds = tone_b_end_samples / SAMPLE_RATE

            tone_a_window = recent_samples[0:tone_a_samples]
            tone_a_peaks = freq_from_fft(tone_a_window, SAMPLE_RATE, tone_a_position_seconds)

            if len(tone_a_peaks) != 1:
                return False

            tone_a_freq = tone_a_peaks[0]

            tone_b_window = recent_samples[tone_a_samples:]
            tone_b_peaks = freq_from_fft(tone_b_window, SAMPLE_RATE, tone_b_position_seconds)

            if len(tone_b_peaks) != 1:
                return False

            tone_b_freq = tone_b_peaks[0]

            if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                return False

            tone_a_match = is_frequency_in_range(
                tone_a_freq, tone_def["tone_a"], tone_def.get("tone_a_range", 10)
            )
            tone_b_match = is_frequency_in_range(
                tone_b_freq, tone_def["tone_b"], tone_def.get("tone_b_range", 10)
            )

            detected_tone_a_duration_ms = (tone_a_samples / SAMPLE_RATE) * 1000.0
            detected_tone_b_duration_ms = (tone_b_samples / SAMPLE_RATE) * 1000.0
            tone_a_duration_match = abs(detected_tone_a_duration_ms - tone_def["tone_a_length_ms"]) <= DURATION_TOLERANCE_MS
            tone_b_duration_match = abs(detected_tone_b_duration_ms - tone_def["tone_b_length_ms"]) <= DURATION_TOLERANCE_MS

            if not (tone_a_match and tone_b_match and tone_a_duration_match and tone_b_duration_match):

                mismatch_details = []
                if not tone_a_match:
                    diff_a = abs(tone_a_freq - tone_def["tone_a"])
                    mismatch_details.append(f"A: {tone_a_freq:.1f}Hz (target {tone_def['tone_a']:.1f}Hz±{tone_def.get('tone_a_range', 10)}, diff={diff_a:.1f}Hz)")
                if not tone_b_match:
                    diff_b = abs(tone_b_freq - tone_def["tone_b"])
                    mismatch_details.append(f"B: {tone_b_freq:.1f}Hz (target {tone_def['tone_b']:.1f}Hz±{tone_def.get('tone_b_range', 10)}, diff={diff_b:.1f}Hz)")
                if not tone_a_duration_match:
                    diff_dur_a = abs(detected_tone_a_duration_ms - tone_def["tone_a_length_ms"])
                    mismatch_details.append(f"A duration: {detected_tone_a_duration_ms:.1f}ms (target {tone_def['tone_a_length_ms']}ms±{DURATION_TOLERANCE_MS}ms, diff={diff_dur_a:.1f}ms)")
                if not tone_b_duration_match:
                    diff_dur_b = abs(detected_tone_b_duration_ms - tone_def["tone_b_length_ms"])
                    mismatch_details.append(f"B duration: {detected_tone_b_duration_ms:.1f}ms (target {tone_def['tone_b_length_ms']}ms±{DURATION_TOLERANCE_MS}ms, diff={diff_dur_b:.1f}ms)")
                # print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Stable frequencies but mismatch - {', '.join(mismatch_details)}")
                return False

            if tone_a_match and tone_b_match and tone_a_duration_match and tone_b_duration_match:

                self.last_detection_time[tone_id] = current_time

                window_id_a = f"{tone_id}_tone_a"
                window_id_b = f"{tone_id}_tone_b"
                self.frequency_history.pop(window_id_a, None)
                self.frequency_history.pop(window_id_b, None)

                confirmation_log = (
                    "\n"
                    + "=" * 80
                    + "\n"
                    + " " * 20
                    + "*** DEFINED TONE SEQUENCE DETECTED! ***\n"
                    + "=" * 80
                    + "\n"
                    + f"  Channel ID:     {self.channel_id}\n"
                    + f"  Tone ID:        {tone_def['tone_id']}\n"
                    + "  \n"
                    + "  Tone A Details:\n"
                    + f"    Detected:     {tone_a_freq:.1f} Hz\n"
                    + f"    Target:       {tone_def['tone_a']:.1f} Hz "
                    f"±{tone_def.get('tone_a_range', 10)} Hz\n"
                    + f"    Duration:     {detected_tone_a_duration_ms:.1f} ms "
                    f"(target: {tone_def['tone_a_length_ms']} ms, "
                    f"tolerance: ±{DURATION_TOLERANCE_MS} ms)\n"
                    + "  \n"
                    + "  Tone B Details:\n"
                    + f"    Detected:     {tone_b_freq:.1f} Hz\n"
                    + f"    Target:       {tone_def['tone_b']:.1f} Hz "
                    f"±{tone_def.get('tone_b_range', 10)} Hz\n"
                    + f"    Duration:     {detected_tone_b_duration_ms:.1f} ms "
                    f"(target: {tone_def['tone_b_length_ms']} ms, "
                    f"tolerance: ±{DURATION_TOLERANCE_MS} ms)\n"
                    + "  \n"
                    + "  Record Length:  "
                    f"{tone_def.get('record_length_ms', 0)} ms\n"
                )
                if tone_def.get("detection_tone_alert"):
                    confirmation_log += (
                        f"  Alert Type:     " f"{tone_def['detection_tone_alert']}\n"
                    )
                confirmation_log += "=" * 80 + "\n"
                print(confirmation_log, flush=True)

                if MQTT_AVAILABLE:
                    publish_known_tone_detection(
                        tone_id=tone_def["tone_id"],
                        tone_a_hz=tone_def["tone_a"],
                        tone_b_hz=tone_def["tone_b"],
                        tone_a_duration_ms=tone_def["tone_a_length_ms"],
                        tone_b_duration_ms=tone_def["tone_b_length_ms"],
                        tone_a_range_hz=tone_def.get("tone_a_range", 10),
                        tone_b_range_hz=tone_def.get("tone_b_range", 10),
                        channel_id=self.channel_id,
                        record_length_ms=tone_def.get("record_length_ms", 0),
                        detection_tone_alert=tone_def.get("detection_tone_alert"),
                    )

                try:
                    from passthrough import global_passthrough_manager

                    if self.passthrough_config.get("tone_passthrough", False):
                        target_channel = self.passthrough_config.get(
                            "passthrough_channel", ""
                        )
                        record_length_ms = tone_def.get("record_length_ms", 0)
                        if target_channel and record_length_ms > 0:
                            global_passthrough_manager.start_passthrough(
                                self.channel_id, target_channel, record_length_ms
                            )
                            print(
                                f"[PASSTHROUGH] Triggered: "
                                f"{self.channel_id} -> {target_channel}, "
                                f"duration={record_length_ms} ms"
                            )
                except Exception as e:
                    print(
                        f"[PASSTHROUGH] ERROR: Failed to trigger " f"passthrough: {e}"
                    )

                try:
                    from recording import global_recording_manager

                    record_length_ms = tone_def.get("record_length_ms", 0)
                    if record_length_ms > 0:
                        global_recording_manager.start_recording(
                            self.channel_id,
                            "defined",
                            tone_def["tone_a"],
                            tone_def["tone_b"],
                            record_length_ms,
                        )
                except Exception as e:
                    print(f"[RECORDING] ERROR: Failed to start " f"recording: {e}")

                return True

        except Exception as e:
            print(f"[TONE DETECTION] ERROR in defined tone detection: {e}")
            import traceback

            traceback.print_exc()

        return False

    def _detect_new_tone_pair(self) -> bool:
        
        if not self.detect_new_tones:
            return False

        max_tone_duration = (
            self.new_tone_length_seconds + self.new_tone_length_tolerance_seconds
        )
        max_tone_samples = int(max_tone_duration * SAMPLE_RATE)
        total_samples = 2 * max_tone_samples

        with self.mutex:
            buffer_len = len(self.audio_buffer)

        if buffer_len < total_samples:
            return False

        current_time = time.time()
        time_since_last = current_time - self.last_new_tone_detection_time
        if time_since_last < MIN_DETECTION_INTERVAL_SECONDS:
            return False

        with self.mutex:
            recent_samples_list = self.audio_buffer[-total_samples:]

        recent_samples = np.array(recent_samples_list, dtype=np.float32)
        
        # Skip tone detection if audio is too quiet (empty/silent audio)
        volume_db = calculate_rms_volume(recent_samples)
        MIN_VOLUME_THRESHOLD_DB = -30.0
        if volume_db < MIN_VOLUME_THRESHOLD_DB:
            return False

        try:
            target_tone_samples = int(self.new_tone_length_seconds * SAMPLE_RATE)
            window1 = recent_samples[0:target_tone_samples]
            
            peaks1 = freq_from_fft(window1, SAMPLE_RATE)
            
            if len(peaks1) != 1:
                return False

            tone_a_freq = peaks1[0]
            
            window2 = recent_samples[target_tone_samples:]
            peaks2 = freq_from_fft(window2, SAMPLE_RATE)

            if len(peaks2) != 1:
                return False

            tone_b_freq = peaks2[0]

            print(f"tone_a_freq: {tone_a_freq}, tone_b_freq: {tone_b_freq}")

            if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                return False

            if abs(tone_a_freq - tone_b_freq) <= 50:
                return False

            is_known_tone = False
            for tone_def in self.tone_definitions:
                if (
                    is_frequency_in_range(
                        tone_a_freq,
                        tone_def["tone_a"],
                        tone_def.get("tone_a_range", 10),
                    )
                    or is_frequency_in_range(
                        tone_a_freq,
                        tone_def["tone_b"],
                        tone_def.get("tone_b_range", 10),
                    )
                    or is_frequency_in_range(
                        tone_b_freq,
                        tone_def["tone_a"],
                        tone_def.get("tone_a_range", 10),
                    )
                    or is_frequency_in_range(
                        tone_b_freq,
                        tone_def["tone_b"],
                        tone_def.get("tone_b_range", 10),
                    )
                ):
                    is_known_tone = True
                    break

            if is_known_tone:
                return False

            current_pair = {
                "tone_a": int(tone_a_freq),
                "tone_b": int(tone_b_freq),
                "tone_a_length": self.new_tone_length_seconds,
                "tone_b_length": self.new_tone_length_seconds,
            }

            if self.last_new_tone_pair is None:
                self.new_tone_pair_count = 0
                self.last_new_tone_pair = current_pair
            elif self.last_new_tone_pair == current_pair:
                if self.new_tone_pair_count < self.new_tone_consecutive_required:
                    self.new_tone_pair_count += 1
                    return False
            else:

                self.new_tone_pair_count = 0
                self.last_new_tone_pair = current_pair
                return False

            self.last_new_tone_detection_time = current_time
            self.new_tone_pair_count = 0

            self.frequency_history.pop("new_tone_window1", None)
            self.frequency_history.pop("new_tone_window2", None)

            self.last_detected_new_pair = {
                "tone_a": tone_a_freq,
                "tone_b": tone_b_freq,
                "tone_a_length_ms": int(self.new_tone_length_seconds * 1000),
                "tone_b_length_ms": int(self.new_tone_length_seconds * 1000),
            }

            self.last_new_tone_pair = None

            print(
                f"[NEW TONE PAIR] Channel {self.channel_id}: "
                f"A={tone_a_freq:.1f} Hz, B={tone_b_freq:.1f} Hz "
                f"(each {self.new_tone_length_seconds:.2f} s, "
                f"±{self.new_tone_range_hz} Hz stable)"
            )

            if MQTT_AVAILABLE:
                publish_new_tone_pair(tone_a_freq, tone_b_freq)

            return True

        except Exception as e:
            print(f"[TONE DETECTION] ERROR in new tone detection: {e}")
            import traceback

            traceback.print_exc()

        return False

    def process_audio(self) -> Optional[Dict[str, Any]]:
        
        import random

        with self.mutex:
            buffer_len = len(self.audio_buffer)
            
            # Quick volume check - skip if buffer is empty or too small
            if buffer_len < 1000:  # Need at least some samples
                return None
            
            # Check volume of recent audio to skip empty/silent audio
            recent_samples_list = self.audio_buffer[-min(48000, buffer_len):]  # Last 1 second or all if less

        recent_samples = np.array(recent_samples_list, dtype=np.float32)
        volume_db = calculate_rms_volume(recent_samples)
        
        # Skip tone detection if audio is too quiet (empty/silent audio)
        MIN_VOLUME_THRESHOLD_DB = -30.0
        if volume_db < MIN_VOLUME_THRESHOLD_DB:
            return None

        if random.randint(1, 100) == 1:
            print(f"[TONE DETECT DEBUG] Channel {self.channel_id}: process_audio() called - buffer={buffer_len} samples, volume={volume_db:.1f} dB, {len(self.tone_definitions)} tone definition(s)")

        for tone_def in self.tone_definitions:
            if self._detect_defined_tone(tone_def):
                return tone_def

        if self.detect_new_tones:
            self._detect_new_tone_pair()

        return None

_channel_detectors: Dict[str, ChannelToneDetector] = {}
_detectors_mutex = threading.Lock()

def init_channel_detector(
    channel_id: str,
    tone_definitions: List[Dict[str, Any]],
    new_tone_config: Optional[Dict[str, Any]] = None,
    passthrough_config: Optional[Dict[str, Any]] = None,
    frequency_filters: Optional[List[Dict[str, Any]]] = None,
) -> None:
    
    with _detectors_mutex:
        if tone_definitions or (
            new_tone_config and new_tone_config.get("detect_new_tones", False)
        ):
            _channel_detectors[channel_id] = ChannelToneDetector(
                channel_id, tone_definitions, new_tone_config, passthrough_config, frequency_filters
            )
            print(
                f"[TONE DETECTION] Initialized detector for channel "
                f"{channel_id} with {len(tone_definitions)} tone "
                f"definition(s)"
            )
            if new_tone_config and new_tone_config.get("detect_new_tones", False):
                tolerance_ms = new_tone_config.get('new_tone_length_tolerance_ms', 50)
                print(
                    f"[TONE DETECTION] New tone detection enabled: "
                    f"length={new_tone_config.get('new_tone_length_ms', 1000)} ms "
                    f"(±{tolerance_ms} ms tolerance), "
                    f"range=±{new_tone_config.get('new_tone_range_hz', 3)} Hz, "
                    f"strikes=3"
                )
            if passthrough_config and passthrough_config.get("tone_passthrough", False):
                print(
                    f"[TONE DETECTION] Passthrough enabled: "
                    f"target={passthrough_config.get('passthrough_channel', 'N/A')}"
                )
            for i, tone_def in enumerate(tone_definitions, 1):
                print(f"[TONE DETECTION]   Definition {i}:")
                print(f"    Tone ID: {tone_def.get('tone_id', 'N/A')}")
                print(
                    f"    Tone A: {tone_def.get('tone_a', 0):.1f} Hz "
                    f"±{tone_def.get('tone_a_range', 10)} Hz "
                    f"({tone_def.get('tone_a_length_ms', 0)} ms)"
                )
                print(
                    f"    Tone B: {tone_def.get('tone_b', 0):.1f} Hz "
                    f"±{tone_def.get('tone_b_range', 10)} Hz "
                    f"({tone_def.get('tone_b_length_ms', 0)} ms)"
                )
                print(
                    f"    Record Length: " f"{tone_def.get('record_length_ms', 0)} ms"
                )
        else:
            _channel_detectors.pop(channel_id, None)
            print(
                f"[TONE DETECTION] No tone definitions found for "
                f"channel {channel_id}"
            )

def add_audio_samples_for_channel(channel_id: str, filtered_audio: np.ndarray) -> None:
    
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if detector:
            detector.add_audio_samples(filtered_audio)

def process_audio_for_channel(
    channel_id: str, filtered_audio: np.ndarray
) -> Optional[Dict[str, Any]]:
    
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if not detector:
            return None

    detector.add_audio_samples(filtered_audio)

    return detector.process_audio()
