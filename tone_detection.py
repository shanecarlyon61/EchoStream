
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

class ToneDetectionState:
    def __init__(self):
        self.tone_definitions: List[ToneDefinition] = [ToneDefinition() for _ in range(MAX_TONE_DEFINITIONS)]
        self.filters: List = []
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

        self.tone_a_confirmed = False
        self.tone_b_confirmed = False
        self.tone_a_tracking = False
        self.tone_b_tracking = False
        self.tone_a_tracking_start = 0
        self.tone_b_tracking_start = 0
        self.active_tone_def: Optional[ToneDefinition] = None
        self.passthrough_active = False

        self.audio_buffer: List[float] = []
        self.max_buffer_samples = int(SAMPLE_RATE * 10)
        self.last_detect_time = 0

global_tone_detection = ToneDetectionState()

def init_tone_detection() -> bool:

    global global_tone_detection

    if global_tone_detection is None:
        global_tone_detection = ToneDetectionState()

    if not hasattr(global_tone_detection, 'audio_buffer'):
        global_tone_detection.audio_buffer = []
        global_tone_detection.max_buffer_samples = int(SAMPLE_RATE * 10)
        global_tone_detection.last_detect_time = 0

    print("[TONE_DETECT] Tone detection system initialized")
    return True

def start_tone_detection() -> bool:

    with global_tone_detection.mutex:
        global_tone_detection.active = True
    print("[TONE_DETECT] Tone detection started")
    return True

def stop_tone_detection():
    with global_tone_detection.mutex:
        global_tone_detection.active = False
    print("[TONE_DETECT] Tone detection stopped")

def is_tone_detect_enabled() -> bool:

    with global_tone_detection.mutex:
        return global_tone_detection.active

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
    with global_tone_detection.mutex:
        global_tone_detection.tone_a_confirmed = False
        global_tone_detection.tone_b_confirmed = False
        global_tone_detection.tone_a_tracking = False
        global_tone_detection.tone_b_tracking = False
        global_tone_detection.tone_a_tracking_start = 0
        global_tone_detection.tone_b_tracking_start = 0
        global_tone_detection.active_tone_def = None

def get_active_tone() -> Optional[ToneDefinition]:

    with global_tone_detection.mutex:
        return global_tone_detection.active_tone_def

def parabolic(f, x):

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

    if len(sig) < 2:
        return 0.0

    try:

        windowed = sig * np.hanning(len(sig))
        f = rfft(windowed)

        magnitudes = np.abs(f)
        i = int(np.argmax(magnitudes))

        if i == 0 or i >= len(magnitudes) - 1:

            return float(fs * i / len(windowed))

        try:

            log_magnitudes = np.log(magnitudes + 1e-10)
            true_i, _ = parabolic(log_magnitudes, i)
            true_i = float(true_i)
        except Exception:
            true_i = float(i)

        freq = fs * true_i / len(windowed)
        return float(freq)

    except Exception as e:
        print(f"[TONE_DETECT] ERROR: freq_from_fft failed: {e}")
        return 0.0

def is_frequency_in_range(detected_freq: float, target_freq: float, range_hz: int) -> bool:

    return abs(detected_freq - target_freq) <= range_hz

def process_audio_samples(samples: np.ndarray, sample_count: int) -> bool:

    if not global_tone_detection.active:
        return False

    current_time_ms = int(time.time() * 1000)

    with global_tone_detection.mutex:

        global_tone_detection.audio_buffer.extend(samples[:sample_count].tolist())

        if len(global_tone_detection.audio_buffer) > global_tone_detection.max_buffer_samples:
            excess = len(global_tone_detection.audio_buffer) - global_tone_detection.max_buffer_samples
            global_tone_detection.audio_buffer = global_tone_detection.audio_buffer[excess:]

        buffer_len = len(global_tone_detection.audio_buffer)

    static_start_logged = getattr(process_audio_samples, '_start_logged', False)
    if not static_start_logged:
        valid_tones = sum(1 for td in global_tone_detection.tone_definitions if td.valid)
        print(f"[TONE_DETECT] Tone detection active: {global_tone_detection.active}")
        print(f"[TONE_DETECT] Loaded {valid_tones} tone definition(s)")
        for i, td in enumerate(global_tone_detection.tone_definitions):
            if td.valid:
                print(f"[TONE_DETECT] Tone {i + 1}: ID={td.tone_id}, A={td.tone_a_freq}Hz±{td.tone_a_range_hz} ({td.tone_a_length_ms}ms), B={td.tone_b_freq}Hz±{td.tone_b_range_hz} ({td.tone_b_length_ms}ms)")
        process_audio_samples._start_logged = True

    if buffer_len < SAMPLE_RATE:
        return False

    threshold_db = -20
    try:
        import config
        for i in range(4):
            channel_config = config.get_channel_config(i)
            if channel_config and channel_config.valid and channel_config.tone_detect:
                threshold_db = channel_config.tone_config.db_threshold
                break
    except Exception:
        pass

    recent_samples = np.array(global_tone_detection.audio_buffer[-SAMPLE_RATE:])
    rms = np.sqrt(np.mean(recent_samples**2))
    volume_db = 20 * np.log10(rms + 1e-10)

    if volume_db < threshold_db:

        return False

    lengths = []
    for tone_def in global_tone_detection.tone_definitions:
        if tone_def.valid:
            l_a = tone_def.tone_a_length_ms / 1000.0
            l_b = tone_def.tone_b_length_ms / 1000.0
            lengths.append((l_a, l_b))

    unique_lengths = sorted(list(set(lengths)), key=lambda x: x[0] + x[1], reverse=True)

    if not unique_lengths:
        return False

    static_process_count = getattr(process_audio_samples, '_process_count', 0)
    process_audio_samples._process_count = static_process_count + 1

    for l_a, l_b in unique_lengths:

        try:

            if not np.isfinite(l_a) or not np.isfinite(l_b) or l_a <= 0 or l_b <= 0:
                continue

            total_length = l_a + l_b
            if not np.isfinite(total_length) or total_length <= 0:
                continue

            required_samples = int(np.round(total_length * SAMPLE_RATE))
            if required_samples <= 0:
                continue

            if buffer_len < required_samples:
                continue

            buf_array = np.array(global_tone_detection.audio_buffer, dtype=np.float32)

            start_idx_val = (l_a + l_b) * SAMPLE_RATE
            end_idx_val = l_b * SAMPLE_RATE

            if not np.isfinite(start_idx_val) or not np.isfinite(end_idx_val):
                continue

            start_idx = int(np.round(float(start_idx_val)))
            end_idx = int(np.round(float(end_idx_val)))

            start_idx = int(start_idx)
            end_idx = int(end_idx)

            if start_idx <= 0 or end_idx <= 0 or start_idx <= end_idx:
                continue
            if len(buf_array) < start_idx:
                continue

            tone_a_segment = buf_array[-start_idx:-end_idx] if end_idx > 0 else buf_array[-start_idx:]
            tone_b_segment = buf_array[-end_idx:]

            if len(tone_a_segment) < int(SAMPLE_RATE * 0.1) or len(tone_b_segment) < int(SAMPLE_RATE * 0.1):
                continue

            try:
                a_tone_freq = freq_from_fft(tone_a_segment, SAMPLE_RATE)
                b_tone_freq = freq_from_fft(tone_b_segment, SAMPLE_RATE)
            except Exception as e:
                if static_process_count % 100 == 0:
                    print(f"[TONE_DETECT] FFT error: {e}")
                continue

            tolerance = 10
            detected = False

            for tone_def in global_tone_detection.tone_definitions:
                if not tone_def.valid:
                    continue

                if abs(tone_def.tone_a_length_ms / 1000.0 - l_a) > 0.1 or \
                   abs(tone_def.tone_b_length_ms / 1000.0 - l_b) > 0.1:
                    continue

                a_match = abs(tone_def.tone_a_freq - a_tone_freq) <= max(tone_def.tone_a_range_hz, tolerance)
                b_match = abs(tone_def.tone_b_freq - b_tone_freq) <= max(tone_def.tone_b_range_hz, tolerance)

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

                    try:
                        import tone_passthrough
                        tone_passthrough.enable_passthrough(tone_def)
                    except ImportError:
                        print("[TONE_DETECT] WARNING: tone_passthrough module not available")

                    detected = True
                    break

            if static_process_count % 500 == 0:
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

            if static_process_count % 100 == 0:
                print(f"[TONE_DETECT] Index calculation error: {e}, l_a={l_a}, l_b={l_b}")
            continue
        except Exception as e:

            if static_process_count % 100 == 0:
                print(f"[TONE_DETECT] Unexpected error in tone detection: {e}")
            continue

    if global_tone_detection.recording_active:
        remaining = get_recording_time_remaining_ms()
        if remaining <= 0:

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

    if process_audio_samples(samples, len(samples)):
        return get_active_tone()
    return None

def is_recording_active() -> bool:

    with global_tone_detection.mutex:
        return global_tone_detection.recording_active

def get_recording_time_remaining_ms() -> int:

    with global_tone_detection.mutex:
        if not global_tone_detection.recording_active:
            return 0

        elapsed = int((time.time() * 1000) - global_tone_detection.recording_start_time)
        remaining = global_tone_detection.recording_duration_ms - elapsed
        return max(0, remaining)

def start_recording_timer(record_length_ms: int) -> bool:

    with global_tone_detection.mutex:
        global_tone_detection.recording_active = True
        global_tone_detection.recording_start_time = int(time.time() * 1000)
        global_tone_detection.recording_duration_ms = record_length_ms
    print(f"[TONE_DETECT] Recording timer started: {record_length_ms} ms")
    return True

def stop_recording_timer():
    with global_tone_detection.mutex:
        global_tone_detection.recording_active = False
    print("[TONE_DETECT] Recording timer stopped")

def process_audio_python_approach(samples: np.ndarray, sample_count: int) -> bool:

    return process_audio_samples(samples, sample_count)

