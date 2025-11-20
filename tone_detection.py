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
    from mqtt_client import (
        publish_known_tone_detection,
        publish_new_tone_pair
    )
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


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of
    an inter-sample maximum when nearby samples are known.
    """
    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    try:
        xv = (1 / 2. * (f[x - 1] - f[x + 1]) /
              (f[x - 1] - 2 * f[x] + f[x + 1]) + x)
        yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
        return xv, yv
    except (ZeroDivisionError, IndexError):
        return float(x), float(f[x])


def freq_from_fft(sig: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """
    Estimate frequency from peak of FFT.
    Similar to reference implementation in ToneDetect/utils/tone.py

    Args:
        sig: Audio signal samples
        fs: Sample rate (default: 48000)

    Returns:
        Detected frequency in Hz
    """
    if len(sig) == 0:
        return 0.0

    # Compute Fourier transform of windowed signal
    windowed = sig * hanning(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    magnitudes = np.abs(f)
    if np.max(magnitudes) == 0:
        return 0.0

    i = np.argmax(magnitudes)
    true_i = parabolic(np.log(magnitudes + 1e-10), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


def is_frequency_in_range(detected_freq: float, target_freq: float,
                          range_hz: int) -> bool:
    """Check if detected frequency is within tolerance of target
    frequency."""
    return abs(detected_freq - target_freq) <= range_hz


def calculate_rms_volume(samples: np.ndarray) -> float:
    """Calculate RMS volume in dB."""
    if len(samples) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(samples ** 2))
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms)


class ChannelToneDetector:
    def __init__(
        self,
        channel_id: str,
        tone_definitions: List[Dict[str, Any]],
        new_tone_config: Optional[Dict[str, Any]] = None,
        passthrough_config: Optional[Dict[str, Any]] = None
    ):
        self.channel_id = channel_id
        self.tone_definitions = tone_definitions
        self.audio_buffer: List[float] = []
        self.mutex = threading.Lock()

        # New tone detection configuration
        if new_tone_config is None:
            new_tone_config = {
                "detect_new_tones": False,
                "new_tone_length_ms": 1000,
                "new_tone_range_hz": 3
            }
        self.detect_new_tones = new_tone_config.get(
            "detect_new_tones", False)
        self.new_tone_length_ms = new_tone_config.get(
            "new_tone_length_ms", 1000)
        self.new_tone_length_seconds = self.new_tone_length_ms / 1000.0
        self.new_tone_range_hz = new_tone_config.get("new_tone_range_hz", 3)
        self.new_tone_config = new_tone_config

        # Passthrough configuration
        if passthrough_config is None:
            passthrough_config = {
                "tone_passthrough": False,
                "passthrough_channel": ""
            }
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
                self.audio_buffer = (
                    self.audio_buffer[-MAX_BUFFER_SAMPLES:]
                )

    def _get_buffer_array(self) -> np.ndarray:
        """Get current audio buffer as numpy array."""
        # Minimize mutex hold time - just copy the list reference
        with self.mutex:
            buffer_copy = list(self.audio_buffer)  # Fast shallow copy
        # Convert to numpy array outside mutex to avoid blocking
        return np.array(buffer_copy, dtype=np.float32)

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
        buffer_array = self._get_buffer_array()

        # Get tone lengths in seconds
        tone_a_length_seconds = tone_def["tone_a_length_ms"] / 1000.0
        tone_b_length_seconds = tone_def["tone_b_length_ms"] / 1000.0

        # Calculate required samples
        tone_a_samples = int(tone_a_length_seconds * SAMPLE_RATE)
        tone_b_samples = int(tone_b_length_seconds * SAMPLE_RATE)
        total_samples = tone_a_samples + tone_b_samples

        # Need at least total_samples in buffer
        if len(buffer_array) < total_samples:
            return False

        # Check minimum time since last detection (avoid duplicates)
        tone_id = tone_def["tone_id"]
        current_time = time.time()
        time_since_last = (
            current_time - self.last_detection_time.get(tone_id, 0)
        )
        if time_since_last < MIN_DETECTION_INTERVAL_SECONDS:
            return False

        # Check volume threshold (already filtered, but double-check)
        recent_samples = buffer_array[-total_samples:]
        _ = calculate_rms_volume(recent_samples)
        # if volume_db < -20:  # -20dB threshold
        #     return False

        try:
            # Analyze Tone A window:
            # [-(total_samples):-tone_b_samples]
            tone_a_window = buffer_array[
                -(total_samples):-tone_b_samples
            ]
            tone_a_freq = freq_from_fft(tone_a_window, SAMPLE_RATE)

            # Analyze Tone B window: last tone_b_samples
            tone_b_window = buffer_array[-tone_b_samples:]
            tone_b_freq = freq_from_fft(tone_b_window, SAMPLE_RATE)

            # Check if frequencies match defined tone
            tone_a_match = is_frequency_in_range(
                tone_a_freq, tone_def["tone_a"],
                tone_def.get("tone_a_range", 10)
            )
            tone_b_match = is_frequency_in_range(
                tone_b_freq, tone_def["tone_b"],
                tone_def.get("tone_b_range", 10)
            )

            if tone_a_match and tone_b_match:
                # Valid tone sequence detected!
                self.last_detection_time[tone_id] = current_time

                confirmation_log = (
                    "\n" + "=" * 80 + "\n" +
                    " " * 20 +
                    "*** DEFINED TONE SEQUENCE DETECTED! ***\n" +
                    "=" * 80 + "\n" +
                    f"  Channel ID:     {self.channel_id}\n" +
                    f"  Tone ID:        {tone_def['tone_id']}\n" +
                    "  \n" +
                    "  Tone A Details:\n" +
                    f"    Detected:     {tone_a_freq:.1f} Hz\n" +
                    f"    Target:       {tone_def['tone_a']:.1f} Hz "
                    f"±{tone_def.get('tone_a_range', 10)} Hz\n" +
                    f"    Duration:     "
                    f"{tone_a_length_seconds:.2f} s "
                    f"({tone_def['tone_a_length_ms']} ms)\n" +
                    "  \n" +
                    "  Tone B Details:\n" +
                    f"    Detected:     {tone_b_freq:.1f} Hz\n" +
                    f"    Target:       {tone_def['tone_b']:.1f} Hz "
                    f"±{tone_def.get('tone_b_range', 10)} Hz\n" +
                    f"    Duration:     "
                    f"{tone_b_length_seconds:.2f} s "
                    f"({tone_def['tone_b_length_ms']} ms)\n" +
                    "  \n" +
                    "  Record Length:  "
                    f"{tone_def.get('record_length_ms', 0)} ms\n"
                )
                if tone_def.get("detection_tone_alert"):
                    confirmation_log += (
                        f"  Alert Type:     "
                        f"{tone_def['detection_tone_alert']}\n"
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
                        record_length_ms=tone_def.get(
                            "record_length_ms", 0),
                        detection_tone_alert=tone_def.get(
                            "detection_tone_alert")
                    )

                # Trigger passthrough if configured
                try:
                    from passthrough import global_passthrough_manager
                    if self.passthrough_config.get(
                        "tone_passthrough", False
                    ):
                        target_channel = (
                            self.passthrough_config.get(
                                "passthrough_channel", ""
                            )
                        )
                        record_length_ms = tone_def.get(
                            "record_length_ms", 0)
                        if target_channel and record_length_ms > 0:
                            global_passthrough_manager.start_passthrough(
                                self.channel_id, target_channel,
                                record_length_ms
                            )
                            print(
                                f"[PASSTHROUGH] Triggered: "
                                f"{self.channel_id} -> {target_channel}, "
                                f"duration={record_length_ms} ms"
                            )
                except Exception as e:
                    print(
                        f"[PASSTHROUGH] ERROR: Failed to trigger "
                        f"passthrough: {e}"
                    )

                # Trigger recording
                try:
                    from recording import global_recording_manager
                    record_length_ms = tone_def.get("record_length_ms", 0)
                    if record_length_ms > 0:
                        global_recording_manager.start_recording(
                            self.channel_id, "defined",
                            tone_def["tone_a"], tone_def["tone_b"],
                            record_length_ms
                        )
                except Exception as e:
                    print(
                        f"[RECORDING] ERROR: Failed to start "
                        f"recording: {e}"
                    )

                return True

        except Exception as e:
            print(
                f"[TONE DETECTION] ERROR in defined tone detection: {e}"
            )
            import traceback
            traceback.print_exc()

        return False

    def _detect_new_tone_pair(self) -> bool:
        """
        Detect new tone pairs using new_tone_length windows.
        Similar to reference implementation: analyzes pairs of windows.

        Returns:
            True if new tone pair detected, False otherwise
        """
        if not self.detect_new_tones:
            return False

        buffer_array = self._get_buffer_array()

        # Calculate required samples for new tone length
        new_tone_samples = int(
            self.new_tone_length_seconds * SAMPLE_RATE
        )
        total_samples = 2 * new_tone_samples  # Need 2 tones worth

        # Need at least total_samples in buffer
        if len(buffer_array) < total_samples:
            return False

        # Check minimum time since last detection
        current_time = time.time()
        time_since_last = (
            current_time - self.last_new_tone_detection_time
        )
        if time_since_last < MIN_DETECTION_INTERVAL_SECONDS:
            return False

        # Check volume threshold
        recent_samples = buffer_array[-total_samples:]
        _ = calculate_rms_volume(recent_samples)
        # if volume_db < -20:  # -20dB threshold
        #     return False

        try:
            # Analyze Tone A window:
            # [-(2*new_tone_samples):-new_tone_samples]
            tone_a_window = buffer_array[
                -(2 * new_tone_samples):-new_tone_samples
            ]
            tone_a_freq = freq_from_fft(tone_a_window, SAMPLE_RATE)

            # Analyze Tone B window: last new_tone_samples
            tone_b_window = buffer_array[-new_tone_samples:]
            tone_b_freq = freq_from_fft(tone_b_window, SAMPLE_RATE)

            # Voice rejection: Check if frequencies match any
            # defined tone
            is_known_tone = False
            for tone_def in self.tone_definitions:
                if (is_frequency_in_range(
                        tone_a_freq, tone_def["tone_a"],
                        tone_def.get("tone_a_range", 10)) or
                    is_frequency_in_range(
                        tone_a_freq, tone_def["tone_b"],
                        tone_def.get("tone_b_range", 10)) or
                    is_frequency_in_range(
                        tone_b_freq, tone_def["tone_a"],
                        tone_def.get("tone_a_range", 10)) or
                    is_frequency_in_range(
                        tone_b_freq, tone_def["tone_b"],
                        tone_def.get("tone_b_range", 10))):
                    is_known_tone = True
                    break

            if is_known_tone:
                return False

            # Voice rejection: Tones must be different
            # (like reference: >50 Hz difference)
            if abs(tone_a_freq - tone_b_freq) <= 50:
                return False

            # Voice rejection: Check frequency stability
            # Analyze multiple windows to ensure frequency is stable
            stability_check_samples = int(0.1 * SAMPLE_RATE)  # 100ms
            if len(buffer_array) >= total_samples + \
                    stability_check_samples:
                # Check tone A stability
                check_window_a = buffer_array[
                    -(2 * new_tone_samples + stability_check_samples):
                    -(2 * new_tone_samples)
                ]
                check_freq_a = freq_from_fft(
                    check_window_a, SAMPLE_RATE
                )
                if abs(check_freq_a - tone_a_freq) > \
                        self.new_tone_range_hz * 2:
                    return False  # Frequency not stable

                # Check tone B stability (before tone B starts)
                check_window_b = buffer_array[
                    -(new_tone_samples + stability_check_samples):
                    -new_tone_samples
                ]
                check_freq_b = freq_from_fft(
                    check_window_b, SAMPLE_RATE
                )
                if abs(check_freq_b - tone_b_freq) > \
                        self.new_tone_range_hz * 2:
                    return False  # Frequency not stable

            # Consistency check: Like reference, require 3 consecutive
            # identical detections
            current_pair = {
                'tone_a': int(tone_a_freq),
                'tone_b': int(tone_b_freq),
                'tone_a_length': self.new_tone_length_seconds,
                'tone_b_length': self.new_tone_length_seconds
            }

            if self.last_new_tone_pair is None:
                self.new_tone_pair_count = 0
                self.last_new_tone_pair = current_pair
            elif self.last_new_tone_pair == current_pair:
                if self.new_tone_pair_count < 3:
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
            try:
                from recording import global_recording_manager
                new_tone_length_ms = self.new_tone_config.get(
                    "new_tone_length_ms", 0
                )
                if new_tone_length_ms > 0:
                    global_recording_manager.start_recording(
                        self.channel_id, "new", tone_a_freq,
                        tone_b_freq, new_tone_length_ms
                    )
            except Exception as e:
                print(
                    f"[RECORDING] ERROR: Failed to start new tone "
                    f"recording: {e}"
                )

            return True

        except Exception as e:
            print(
                f"[TONE DETECTION] ERROR in new tone detection: {e}"
            )
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
    passthrough_config: Optional[Dict[str, Any]] = None
) -> None:
    """Initialize tone detector for a channel."""
    with _detectors_mutex:
        if tone_definitions or (
            new_tone_config and
            new_tone_config.get("detect_new_tones", False)
        ):
            _channel_detectors[channel_id] = ChannelToneDetector(
                channel_id, tone_definitions, new_tone_config,
                passthrough_config
            )
            print(
                f"[TONE DETECTION] Initialized detector for channel "
                f"{channel_id} with {len(tone_definitions)} tone "
                f"definition(s)"
            )
            if (new_tone_config and
                    new_tone_config.get("detect_new_tones", False)):
                print(
                    f"[TONE DETECTION] New tone detection enabled: "
                    f"length={new_tone_config.get('new_tone_length_ms', 1000)} ms, "  # noqa: E501
                    f"range=±{new_tone_config.get('new_tone_range_hz', 3)} Hz"  # noqa: E501
                )
            if (passthrough_config and
                    passthrough_config.get("tone_passthrough", False)):
                print(
                    f"[TONE DETECTION] Passthrough enabled: "
                    f"target={passthrough_config.get('passthrough_channel', 'N/A')}"  # noqa: E501
                )
            for i, tone_def in enumerate(tone_definitions, 1):
                print(f"[TONE DETECTION]   Definition {i}:")
                print(
                    f"    Tone ID: {tone_def.get('tone_id', 'N/A')}"
                )
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
                    f"    Record Length: "
                    f"{tone_def.get('record_length_ms', 0)} ms"
                )
        else:
            _channel_detectors.pop(channel_id, None)
            print(
                f"[TONE DETECTION] No tone definitions found for "
                f"channel {channel_id}"
            )


def add_audio_samples_for_channel(
    channel_id: str,
    filtered_audio: np.ndarray
) -> None:
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
    channel_id: str,
    filtered_audio: np.ndarray
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
