"""
ToneDetection Module - Rebuilt from scratch using freq_from_fft approach.
Detects defined tone sequences and new tone pairs from filtered audio.
"""

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from numpy.fft import rfft

try:
    from scipy.signal import hanning
except ImportError:
    # Fallback to numpy if scipy not available
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
# Maximum buffer size: 10 seconds of audio
MAX_BUFFER_SECONDS = 10
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)
# Minimum time between detections (to avoid duplicates)
MIN_DETECTION_INTERVAL_SECONDS = 2.0
# Duration tolerance for tone detection (allows ±80ms variation)
DURATION_TOLERANCE_MS = 80


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of
    an inter-sample maximum when nearby samples are known.
    """
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
    """
    Calculate peak prominence - the height difference between the peak
    and the higher of the two surrounding valleys.

    Args:
        magnitudes: Magnitude array
        peak_idx: Index of the peak
        window_size: Size of window to search for valleys on each side

    Returns:
        Prominence value (magnitude difference)
    """
    peak_mag = magnitudes[peak_idx]

    # Find minimum on left side (valley)
    left_start = max(0, peak_idx - window_size)
    left_valley = (
        np.min(magnitudes[left_start:peak_idx]) if peak_idx > 0 else peak_mag
    )

    # Find minimum on right side (valley)
    right_end = min(len(magnitudes), peak_idx + window_size + 1)
    right_valley = (
        np.min(magnitudes[peak_idx + 1 : right_end])
        if peak_idx < len(magnitudes) - 1
        else peak_mag
    )

    # Prominence is the difference between peak and the higher valley
    prominence = peak_mag - max(left_valley, right_valley)

    return prominence


def calculate_peak_width(
    magnitudes: np.ndarray, peak_idx: int, prominence_fraction: float = 0.5
) -> int:
    """
    Calculate peak width at a fraction of prominence.

    Args:
        magnitudes: Magnitude array
        peak_idx: Index of the peak
        prominence_fraction: Fraction of prominence to measure width at
                            (default 0.5 = half-height)

    Returns:
        Width in bins
    """
    peak_mag = magnitudes[peak_idx]
    prominence = calculate_peak_prominence(magnitudes, peak_idx)
    threshold = peak_mag - (prominence * prominence_fraction)

    # Find left edge
    left_edge = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if magnitudes[i] < threshold:
            left_edge = i + 1
            break
    else:
        left_edge = 0

    # Find right edge
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
    magnitude_threshold: Optional[float] = None,
    min_prominence_ratio: float = 0.3,
    max_width_bins: int = 20,
    min_separation_hz: float = 20.0,
) -> List[float]:
    """
    Estimate frequency peaks from FFT with improved peak detection.
    Uses prominence, width validation, and minimum separation.

    Args:
        sig: Audio signal samples
        fs: Sample rate (default: 48000)
        magnitude_threshold: Minimum magnitude to consider. If None, uses adaptive
                            threshold (10% of max magnitude). If provided, uses that value.
                            (default: None = adaptive)
        min_prominence_ratio: Minimum prominence as ratio of peak magnitude
                            (default: 0.3 = 30%)
        max_width_bins: Maximum peak width in FFT bins (default: 20)
        min_separation_hz: Minimum frequency separation between peaks in Hz
                          (default: 20.0)

    Returns:
        List of detected frequency peaks in Hz (sorted by magnitude, descending)
    """
    if len(sig) == 0:
        return []

    # Compute Fourier transform of windowed signal
    windowed = sig * hanning(len(sig))
    f = rfft(windowed)

    # Find peaks with magnitude above threshold
    magnitudes = np.abs(f)
    max_magnitude = np.max(magnitudes)
    if max_magnitude == 0:
        return []

    # Use adaptive threshold if not provided
    # For low-volume audio, use a percentage of max magnitude instead of fixed value
    if magnitude_threshold is None:
        # Use 10% of max magnitude as threshold (works for both loud and quiet audio)
        magnitude_threshold = max_magnitude * 0.1
        # But ensure minimum threshold to avoid noise (at least 10.0 for very quiet signals)
        magnitude_threshold = max(magnitude_threshold, 10.0)
        # For very quiet signals, use even lower threshold (5% if max is very small)
        if max_magnitude < 100.0:
            magnitude_threshold = max_magnitude * 0.05
            magnitude_threshold = max(magnitude_threshold, 5.0)

    # Find all local maxima above threshold
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

    # Calculate prominence and width for each candidate peak
    peak_data = []
    for peak_idx in candidate_peaks:
        prominence = calculate_peak_prominence(magnitudes, peak_idx)
        width = calculate_peak_width(magnitudes, peak_idx)
        peak_mag = magnitudes[peak_idx]

        # Check prominence requirement (must be at least
        # min_prominence_ratio of peak magnitude)
        min_prominence = peak_mag * min_prominence_ratio
        if prominence < min_prominence:
            continue

        # Check width requirement (peaks shouldn't be too wide)
        if width > max_width_bins:
            continue

        # Interpolate to get more accurate peak position
        true_i = parabolic(np.log(magnitudes + 1e-10), peak_idx)[0]
        peak_freq = fs * true_i / len(windowed)

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

    # Sort by magnitude (strongest first)
    peak_data.sort(key=lambda x: x["magnitude"], reverse=True)

    # Apply minimum separation filter
    # Keep strongest peaks, but remove peaks that are too close to stronger ones
    filtered_peaks = []
    min_separation_bins = int((min_separation_hz * len(windowed)) / fs)

    for peak in peak_data:
        # Check if this peak is far enough from all already accepted peaks
        too_close = False
        for accepted_peak in filtered_peaks:
            bin_separation = abs(peak["index"] - accepted_peak["index"])
            if bin_separation < min_separation_bins:
                too_close = True
                break

        if not too_close:
            filtered_peaks.append(peak)

    # Extract frequencies
    peaks = [p["freq"] for p in filtered_peaks]

    return peaks


def is_frequency_in_range(
    detected_freq: float, target_freq: float, range_hz: int
) -> bool:
    """Check if detected frequency is within tolerance of target
    frequency."""
    return abs(detected_freq - target_freq) <= range_hz


def calculate_rms_volume(samples: np.ndarray) -> float:
    """Calculate RMS volume in dB."""
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
    """
    Count significant frequency peaks in FFT spectrum.

    Used to distinguish human voice (3+ peaks) from pure tones (≤2 peaks).

    Args:
        sig: Audio signal samples
        fs: Sample rate (default: 48000)
        peak_threshold: Minimum magnitude relative to max peak (0.0-1.0)
                       Default 0.3 = peak must be 30% of max magnitude
        min_freq: Minimum frequency to consider (Hz) - filters DC and very low
        max_freq: Maximum frequency to consider (Hz) - filters noise
        min_peak_separation_hz: Minimum Hz between peaks to count separately

    Returns:
        Number of significant peaks detected
        - Pure tone: 1-2 peaks
        - Voice: 3+ peaks
    """
    if len(sig) == 0:
        return 0

    # Compute FFT (same as freq_from_fft)
    windowed = sig * hanning(len(sig))
    f = rfft(windowed)
    magnitudes = np.abs(f)

    if np.max(magnitudes) == 0:
        return 0

    # Normalize magnitudes
    magnitudes_norm = magnitudes / np.max(magnitudes)

    # Convert frequency bins to Hz
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / fs)

    # Filter by frequency range
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    magnitudes_filtered = magnitudes_norm[freq_mask]
    freqs_filtered = freqs[freq_mask]

    if len(magnitudes_filtered) < 3:
        return 0

    # Find peaks using simple local maximum detection
    peaks = []
    for i in range(1, len(magnitudes_filtered) - 1):
        # Check if this is a local maximum above threshold
        if (
            magnitudes_filtered[i] > magnitudes_filtered[i - 1]
            and magnitudes_filtered[i] > magnitudes_filtered[i + 1]
            and magnitudes_filtered[i] >= peak_threshold
        ):
            peaks.append((freqs_filtered[i], magnitudes_filtered[i]))

    # Remove peaks that are too close together
    # Keep only the strongest peak in each cluster
    if len(peaks) > 1:
        peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
        filtered_peaks = [peaks_sorted[0]]

        for freq, mag in peaks_sorted[1:]:
            # Check if this peak is far enough from existing peaks
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

        # Store frequency filters for validation
        self.frequency_filters = frequency_filters or []

        # New tone detection configuration
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
        # Duration tolerance for new tone detection (allows ±50ms by default)
        # This handles cases where tone duration is 0.98s instead of 1.0s
        self.new_tone_length_tolerance_ms = new_tone_config.get(
            "new_tone_length_tolerance_ms", 50
        )
        self.new_tone_length_tolerance_seconds = (
            self.new_tone_length_tolerance_ms / 1000.0
        )
        # Hardcoded to 3 consecutive detections (strikes)
        self.new_tone_consecutive_required = 3
        self.new_tone_config = new_tone_config

        # Passthrough configuration
        if passthrough_config is None:
            passthrough_config = {"tone_passthrough": False, "passthrough_channel": ""}
        self.passthrough_config = passthrough_config

        # Tracking state for defined tones
        self.last_detection_time: Dict[str, float] = {}
        for tone_def in tone_definitions:
            tone_id = tone_def["tone_id"]
            self.last_detection_time[tone_id] = 0.0

        # Tracking state for new tones
        self.last_new_tone_detection_time = 0.0
        # For consistency check (like reference: 3 consecutive)
        self.new_tone_pair_count = 0
        self.last_new_tone_pair: Optional[Dict[str, Any]] = None
        # Store last detected pair for external access (doesn't get reset)
        self.last_detected_new_pair: Optional[Dict[str, Any]] = None

        # Frequency stability tracking
        # Track recent frequency detections for stability verification
        self.frequency_history: Dict[str, List[Dict[str, Any]]] = {}
        # Configuration for stability checking
        self.stability_required_count = 3  # Number of consistent detections required
        self.stability_tolerance_hz = 5.0  # Frequency tolerance for stability (Hz)
        self.stability_max_age_seconds = 1.0  # Maximum age of history entries (seconds)

    def add_audio_samples(self, samples: np.ndarray):
        """
        Add filtered audio samples to the buffer.
        These samples should already be filtered by
        apply_audio_frequency_filters() which includes -20dB threshold
        and frequency filtering.
        """
        with self.mutex:
            # Convert to list efficiently - use tolist() for small arrays,
            # or convert in chunks for very large arrays to avoid blocking
            if len(samples) < 10000:
                # Small arrays: convert directly
                self.audio_buffer.extend(samples.tolist())
            else:
                # Large arrays: convert in chunks to avoid blocking
                chunk_size = 5000
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    self.audio_buffer.extend(chunk.tolist())
            if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
                self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

    def _get_buffer_array(self) -> np.ndarray:
        """Get current audio buffer as numpy array."""
        # Minimize mutex hold time - just copy the list reference
        with self.mutex:
            buffer_copy = list(self.audio_buffer)  # Fast shallow copy
        # Convert to numpy array outside mutex to avoid blocking
        return np.array(buffer_copy, dtype=np.float32)

    def _check_frequency_stability(
        self, frequency: float, window_id: str, current_time: float
    ) -> bool:
        """
        Check if a frequency is stable across multiple windows.

        Args:
            frequency: Detected frequency in Hz
            window_id: Identifier for the window (e.g., "tone_a", "tone_b")
            current_time: Current timestamp

        Returns:
            True if frequency is stable (has required number of consistent detections)
        """
        # Initialize history for this window if needed
        if window_id not in self.frequency_history:
            self.frequency_history[window_id] = []

        history = self.frequency_history[window_id]

        # Remove old entries (older than max_age)
        history[:] = [
            entry
            for entry in history
            if (current_time - entry["time"]) <= self.stability_max_age_seconds
        ]

        # Add current detection
        history.append({"freq": frequency, "time": current_time})

        # If we don't have enough detections yet, return False
        if len(history) < self.stability_required_count:
            return False

        # Check if recent detections are consistent
        recent_detections = history[-self.stability_required_count :]
        frequencies = [d["freq"] for d in recent_detections]

        # Calculate mean and check if all are within tolerance
        mean_freq = np.mean(frequencies)
        max_deviation = max(abs(f - mean_freq) for f in frequencies)

        # Frequency is stable if all recent detections are within tolerance
        return max_deviation <= self.stability_tolerance_hz

    def _is_frequency_filtered(self, frequency: float) -> bool:
        """
        Check if a detected frequency falls within any configured filter range.

        Args:
            frequency: Detected frequency in Hz

        Returns:
            True if frequency is filtered out (should reject), False otherwise
        """
        if not self.frequency_filters:
            return False

        for filter_data in self.frequency_filters:
            filter_freq = filter_data.get("frequency", 0.0)
            filter_type = filter_data.get("type", "")

            if filter_type == "below":
                # Filter frequencies below the threshold
                if frequency < filter_freq:
                    return True
            elif filter_type == "above":
                # Filter frequencies above the threshold
                if frequency > filter_freq:
                    return True

        return False

    def _detect_defined_tone(self, tone_def: Dict[str, Any]) -> bool:
        """
        Detect defined tone sequence using time windows.
        Similar to reference implementation: analyzes specific time
        windows.

        Args:
            tone_def: Tone definition dictionary

        Returns:
            True if tone sequence detected, False otherwise
        """
        import random  # For debug logging
        
        # Get tone lengths in seconds
        tone_a_length_seconds = tone_def["tone_a_length_ms"] / 1000.0
        tone_b_length_seconds = tone_def["tone_b_length_ms"] / 1000.0

        # Calculate required samples
        tone_a_samples = int(tone_a_length_seconds * SAMPLE_RATE)
        tone_b_samples = int(tone_b_length_seconds * SAMPLE_RATE)
        total_samples = tone_a_samples + tone_b_samples

        # OPTIMIZATION 1: Quick buffer size check BEFORE expensive conversion
        with self.mutex:
            buffer_len = len(self.audio_buffer)
        
        if buffer_len < total_samples:
            if random.randint(1, 100) == 1:  # Log occasionally when buffer is too small
                print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_def['tone_id']}: Buffer too small - {buffer_len} samples (need {total_samples})")
            return False

        # OPTIMIZATION 2: Quick volume check on recent samples BEFORE expensive buffer conversion
        # Check volume on last total_samples only (much faster than converting entire 10s buffer)
        with self.mutex:
            recent_samples_list = self.audio_buffer[-total_samples:]
        
        # Convert only the recent samples we need (not entire 480k buffer)
        recent_samples = np.array(recent_samples_list, dtype=np.float32)
        volume_db = calculate_rms_volume(recent_samples)
        
        # Debug: Log volume periodically
        if random.randint(1, 50) == 1:
            print(f"[TONE DETECT] Channel {self.channel_id}: Volume={volume_db:.1f} dB (threshold=-20dB)")
        
        # if volume_db < -20:  # -20dB threshold (like reference project)
        #     return False

        # Check minimum time since last detection (avoid duplicates)
        tone_id = tone_def["tone_id"]
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time.get(tone_id, 0)
        if time_since_last < MIN_DETECTION_INTERVAL_SECONDS:
            if random.randint(1, 50) == 1:  # Log occasionally when too soon
                print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Too soon since last detection - {time_since_last:.1f}s (min {MIN_DETECTION_INTERVAL_SECONDS}s)")
            return False

        # OPTIMIZATION 3: Use already-converted recent_samples array (no duplicate conversion)
        try:
            # Analyze Tone A window: [0:tone_a_samples]
            # (recent_samples already contains the last total_samples)
            tone_a_window = recent_samples[0:tone_a_samples]
            # Use adaptive threshold based on signal strength for low-volume audio
            # Calculate a relative threshold (percentage of max magnitude)
            tone_a_peaks = freq_from_fft(tone_a_window, SAMPLE_RATE, magnitude_threshold=None)

            # If 2+ peaks detected, signal doesn't have a pair of tones - reject
            if len(tone_a_peaks) >= 2:
                if random.randint(1, 20) == 1:  # More frequent logging
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Tone A rejected - {len(tone_a_peaks)} peaks detected (expected 1): {[f'{p:.1f}Hz' for p in tone_a_peaks[:3]]}")
                return False

            # If no peaks found, reject
            if len(tone_a_peaks) == 0:
                if random.randint(1, 10) == 1:  # More frequent logging for debugging
                    # Calculate max magnitude for debugging
                    windowed = tone_a_window * hanning(len(tone_a_window))
                    f = rfft(windowed)
                    magnitudes = np.abs(f)
                    max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 0
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Tone A rejected - no peaks detected (max_magnitude={max_mag:.1f}, volume={volume_db:.1f}dB)")
                return False

            tone_a_freq = tone_a_peaks[0]

            # Analyze Tone B window: [tone_a_samples:]
            tone_b_window = recent_samples[tone_a_samples:]
            # Use adaptive threshold based on signal strength for low-volume audio
            tone_b_peaks = freq_from_fft(tone_b_window, SAMPLE_RATE, magnitude_threshold=None)

            # If 2+ peaks detected, signal doesn't have a pair of tones - reject
            if len(tone_b_peaks) >= 2:
                if random.randint(1, 20) == 1:  # More frequent logging
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Tone B rejected - {len(tone_b_peaks)} peaks detected (expected 1): {[f'{p:.1f}Hz' for p in tone_b_peaks[:3]]}")
                return False

            # If no peaks found, reject
            if len(tone_b_peaks) == 0:
                if random.randint(1, 20) == 1:  # More frequent logging
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Tone B rejected - no peaks detected")
                return False

            tone_b_freq = tone_b_peaks[0]

            # Check frequency stability across multiple windows
            window_id_a = f"{tone_id}_tone_a"
            window_id_b = f"{tone_id}_tone_b"
            tone_a_stable = self._check_frequency_stability(
                tone_a_freq, window_id_a, current_time
            )
            tone_b_stable = self._check_frequency_stability(
                tone_b_freq, window_id_b, current_time
            )

            # Require both frequencies to be stable before accepting
            if not (tone_a_stable and tone_b_stable):
                if random.randint(1, 10) == 1:  # More frequent logging for stability
                    stability_status = f"A: {'STABLE' if tone_a_stable else 'UNSTABLE'}, B: {'STABLE' if tone_b_stable else 'UNSTABLE'}"
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Frequencies not stable yet - {tone_a_freq:.1f}Hz, {tone_b_freq:.1f}Hz ({stability_status})")
                return False

            # Check if detected frequencies are within filter config ranges
            if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                if random.randint(1, 10) == 1:  # More frequent logging
                    filtered_a = self._is_frequency_filtered(tone_a_freq)
                    filtered_b = self._is_frequency_filtered(tone_b_freq)
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Frequencies filtered out - A={tone_a_freq:.1f}Hz ({'FILTERED' if filtered_a else 'OK'}), B={tone_b_freq:.1f}Hz ({'FILTERED' if filtered_b else 'OK'})")
                return False

            # Debug: Log detected frequencies more frequently
            if random.randint(1, 10) == 1:  # Changed from 1 in 50 to 1 in 10
                print(f"[TONE DETECT] Channel {self.channel_id}: Detected A={tone_a_freq:.1f}Hz, B={tone_b_freq:.1f}Hz | " +
                      f"Target A={tone_def['tone_a']:.1f}Hz±{tone_def.get('tone_a_range', 10)}, B={tone_def['tone_b']:.1f}Hz±{tone_def.get('tone_b_range', 10)}")

            # Check if frequencies match defined tone
            tone_a_match = is_frequency_in_range(
                tone_a_freq, tone_def["tone_a"], tone_def.get("tone_a_range", 10)
            )
            tone_b_match = is_frequency_in_range(
                tone_b_freq, tone_def["tone_b"], tone_def.get("tone_b_range", 10)
            )

            # Check if durations match within tolerance (50ms)
            detected_tone_a_duration_ms = (tone_a_samples / SAMPLE_RATE) * 1000.0
            detected_tone_b_duration_ms = (tone_b_samples / SAMPLE_RATE) * 1000.0
            tone_a_duration_match = abs(detected_tone_a_duration_ms - tone_def["tone_a_length_ms"]) <= DURATION_TOLERANCE_MS
            tone_b_duration_match = abs(detected_tone_b_duration_ms - tone_def["tone_b_length_ms"]) <= DURATION_TOLERANCE_MS

            # Log mismatch details if frequencies are stable but don't match
            if not (tone_a_match and tone_b_match and tone_a_duration_match and tone_b_duration_match):
                if random.randint(1, 5) == 1:  # Very frequent logging for mismatches
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
                    print(f"[TONE DETECT DEBUG] Channel {self.channel_id} Tone {tone_id}: Mismatch - {', '.join(mismatch_details)}")
                return False

            if tone_a_match and tone_b_match and tone_a_duration_match and tone_b_duration_match:
                # Valid tone sequence detected!
                self.last_detection_time[tone_id] = current_time

                # Clear frequency history for this tone to avoid stale data
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

                # Publish to MQTT
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

                # Trigger passthrough if configured
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

                # Trigger recording
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
        """
        Detect new tone pairs using new_tone_length windows with tolerance.
        Similar to reference implementation: analyzes pairs of windows.
        
        Uses duration tolerance to detect tones that are close to the
        expected duration (e.g., 0.98s instead of 1.0s).

        Returns:
            True if new tone pair detected, False otherwise
        """
        if not self.detect_new_tones:
            return False

        # Calculate required samples with tolerance
        # Use the maximum duration for buffer check to ensure we have enough samples
        max_tone_duration = (
            self.new_tone_length_seconds + self.new_tone_length_tolerance_seconds
        )
        max_tone_samples = int(max_tone_duration * SAMPLE_RATE)
        total_samples = 2 * max_tone_samples  # Need 2 tones worth

        # OPTIMIZATION 1: Quick buffer size check BEFORE expensive conversion
        with self.mutex:
            buffer_len = len(self.audio_buffer)
        
        if buffer_len < total_samples:
            return False

        # Check minimum time since last detection
        current_time = time.time()
        time_since_last = current_time - self.last_new_tone_detection_time
        if time_since_last < MIN_DETECTION_INTERVAL_SECONDS:
            return False

        # OPTIMIZATION 2: Quick volume check on recent samples BEFORE expensive buffer conversion
        # Check volume on last total_samples only (much faster than converting entire 10s buffer)
        with self.mutex:
            recent_samples_list = self.audio_buffer[-total_samples:]
        
        # Convert only the recent samples we need (not entire 480k buffer)
        recent_samples = np.array(recent_samples_list, dtype=np.float32)
        
        # Volume check removed - handled by frequency filters

        # OPTIMIZATION 3: Use already-converted recent_samples array (no duplicate conversion)
        try:
            # Try detecting with target duration (center of tolerance range)
            target_tone_samples = int(self.new_tone_length_seconds * SAMPLE_RATE)
            
            # Analyze Tone A window: [0:target_tone_samples]
            # (recent_samples already contains the last total_samples)
            tone_a_window = recent_samples[0:target_tone_samples]
            # Use adaptive threshold based on signal strength for low-volume audio
            tone_a_peaks = freq_from_fft(tone_a_window, SAMPLE_RATE, magnitude_threshold=None)

            # If 2+ peaks detected, signal doesn't have a pair of tones - reject
            if len(tone_a_peaks) >= 2:
                return False

            # If no peaks found, reject
            if len(tone_a_peaks) == 0:
                return False

            tone_a_freq = tone_a_peaks[0]

            # Analyze Tone B window: [target_tone_samples:]
            tone_b_window = recent_samples[target_tone_samples:]
            # Use adaptive threshold based on signal strength for low-volume audio
            tone_b_peaks = freq_from_fft(tone_b_window, SAMPLE_RATE, magnitude_threshold=None)

            # If 2+ peaks detected, signal doesn't have a pair of tones - reject
            if len(tone_b_peaks) >= 2:
                return False

            # If no peaks found, reject
            if len(tone_b_peaks) == 0:
                return False

            tone_b_freq = tone_b_peaks[0]

            # Check frequency stability across multiple windows
            window_id_a = "new_tone_a"
            window_id_b = "new_tone_b"
            tone_a_stable = self._check_frequency_stability(
                tone_a_freq, window_id_a, current_time
            )
            tone_b_stable = self._check_frequency_stability(
                tone_b_freq, window_id_b, current_time
            )

            # Require both frequencies to be stable before accepting
            if not (tone_a_stable and tone_b_stable):
                return False

            # Check if detected frequencies are within filter config ranges
            if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                return False

            # Voice rejection: Tones must be different
            # (like reference: >50 Hz difference)
            if abs(tone_a_freq - tone_b_freq) <= 50:
                return False
                
            # Voice rejection: Check if frequencies match any
            # defined tone
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


            # OPTIMIZATION 4: Removed expensive stability checks (2 extra FFTs)
            # Instead, rely on 3 consecutive detections (like reference project)
            # This eliminates 2 expensive FFT operations per detection cycle

            # Consistency check: Like reference, require consecutive
            # identical detections
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
                    return False  # Not enough consecutive detections yet
            else:
                # Different pair, reset
                self.new_tone_pair_count = 0
                self.last_new_tone_pair = current_pair
                return False

            # Valid new tone pair detected!
            self.last_new_tone_detection_time = current_time
            self.new_tone_pair_count = 0

            # Clear frequency history for new tones to avoid stale data
            self.frequency_history.pop("new_tone_a", None)
            self.frequency_history.pop("new_tone_b", None)
            
            # Store detected frequencies for external access (before reset)
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

            # Publish to MQTT
            if MQTT_AVAILABLE:
                publish_new_tone_pair(tone_a_freq, tone_b_freq)

            # Trigger recording
            # try:
            #     from recording import global_recording_manager

            #     new_tone_length_ms = self.new_tone_config.get("new_tone_length_ms", 0)
            #     if new_tone_length_ms > 0:
            #         global_recording_manager.start_recording(
            #             self.channel_id,
            #             "new",
            #             tone_a_freq,
            #             tone_b_freq,
            #             new_tone_length_ms,
            #         )
            # except Exception as e:
            #     print(f"[RECORDING] ERROR: Failed to start new tone " f"recording: {e}")

            return True

        except Exception as e:
            print(f"[TONE DETECTION] ERROR in new tone detection: {e}")
            import traceback

            traceback.print_exc()

        return False

    def process_audio(self) -> Optional[Dict[str, Any]]:
        """
        Process audio from buffer to detect defined and new tones.
        Note: The audio_buffer contains pre-filtered audio (with
        -20dB threshold and frequency filters applied) from
        apply_audio_frequency_filters().

        Returns:
            Tone definition dict if a defined tone sequence is
            detected, None otherwise.
        """
        import random
        
        # Log buffer status occasionally
        with self.mutex:
            buffer_len = len(self.audio_buffer)
        
        if random.randint(1, 100) == 1:  # Log occasionally
            print(f"[TONE DETECT DEBUG] Channel {self.channel_id}: process_audio() called - buffer={buffer_len} samples, {len(self.tone_definitions)} tone definition(s)")
        
        # Detect defined tones first
        for tone_def in self.tone_definitions:
            if self._detect_defined_tone(tone_def):
                return tone_def

        # Detect new tone pairs
        if self.detect_new_tones:
            self._detect_new_tone_pair()

        return None


# Global detector management
_channel_detectors: Dict[str, ChannelToneDetector] = {}
_detectors_mutex = threading.Lock()


def init_channel_detector(
    channel_id: str,
    tone_definitions: List[Dict[str, Any]],
    new_tone_config: Optional[Dict[str, Any]] = None,
    passthrough_config: Optional[Dict[str, Any]] = None,
    frequency_filters: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Initialize tone detector for a channel."""
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
                    f"length={new_tone_config.get('new_tone_length_ms', 1000)} ms "  # noqa: E501
                    f"(±{tolerance_ms} ms tolerance), "  # noqa: E501
                    f"range=±{new_tone_config.get('new_tone_range_hz', 3)} Hz, "  # noqa: E501
                    f"strikes=3"  # noqa: E501
                )
            if passthrough_config and passthrough_config.get("tone_passthrough", False):
                print(
                    f"[TONE DETECTION] Passthrough enabled: "
                    f"target={passthrough_config.get('passthrough_channel', 'N/A')}"  # noqa: E501
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
    """
    Add filtered audio samples to the buffer without processing.
    This is used to keep the buffer updated when processing is
    throttled.

    Args:
        channel_id: Channel identifier
        filtered_audio: Audio samples that have been filtered by
                       apply_audio_frequency_filters()
    """
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if detector:
            detector.add_audio_samples(filtered_audio)


def process_audio_for_channel(
    channel_id: str, filtered_audio: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Process filtered audio for tone detection (both defined and new
    tones).

    Args:
        channel_id: Channel identifier
        filtered_audio: Audio samples that have been filtered by
                       apply_audio_frequency_filters() (includes -20dB
                       threshold and frequency filters)

    Returns:
        Tone definition dict if a defined tone sequence is detected,
        None otherwise. New tone pairs are detected and published via
        MQTT but don't return a value.
    """
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if not detector:
            return None

    # Add filtered audio to buffer (used for both defined and new tone
    # detection)
    detector.add_audio_samples(filtered_audio)
    # Process audio - detects both defined tones and new tone pairs
    return detector.process_audio()
