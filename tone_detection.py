import numpy as np
import threading
import time
import queue
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

class STFT:

    def __init__(self, fft_len: int = 8192, hop_len: int = 4096, sample_rate: int = SAMPLE_RATE, 
                 window_name: str = "Hanning", n_average: int = 1):

        if (fft_len & (fft_len - 1)) != 0:
            raise ValueError("fft_len must be a power of 2")

        self.fft_len = fft_len
        self.hop_len = hop_len
        self.sample_rate = sample_rate
        self.n_average = n_average

        self.spectrum_amp_in = np.zeros(fft_len, dtype=np.float32)
        self.spectrum_amp_pt = 0

        self.out_len = fft_len // 2 + 1
        self.spectrum_amp_out_cum = np.zeros(self.out_len, dtype=np.float64)
        self.spectrum_amp_out = np.zeros(self.out_len, dtype=np.float64)
        self.spectrum_amp_out_db = np.zeros(self.out_len, dtype=np.float64)

        self.n_analysed = 0

        self.window = self._init_window_function(fft_len, window_name)

        window_sum = np.sum(self.window)
        self.window_normalize = fft_len / window_sum

        window_energy = np.sum(self.window ** 2)
        self.wnd_energy_factor = fft_len / window_energy

        self.cum_rms = 0.0
        self.cnt_rms = 0
        self.out_rms = 0.0

        self.max_amp_freq = float('nan')
        self.max_amp_db = float('nan')

    def _init_window_function(self, fft_len: int, window_name: str) -> np.ndarray:

        if window_name == "Hanning":
            window = np.hanning(fft_len)

            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fft_len) / (fft_len - 1))) * 2
        elif window_name == "Blackman":

            i = np.arange(fft_len)
            window = (0.42 - 0.5 * np.cos(2 * np.pi * i / (fft_len - 1)) + 
                     0.08 * np.cos(4 * np.pi * i / (fft_len - 1)))
        else:

            window = np.hanning(fft_len)

        return window.astype(np.float32)

    def feed_data(self, samples: np.ndarray) -> None:

        ds_len = len(samples)
        ds_pt = 0

        while ds_pt < ds_len:

            while self.spectrum_amp_pt < self.fft_len and ds_pt < ds_len:
                s = float(samples[ds_pt])
                self.spectrum_amp_in[self.spectrum_amp_pt] = s
                self.spectrum_amp_pt += 1
                ds_pt += 1

                self.cum_rms += s * s
                self.cnt_rms += 1

            if self.spectrum_amp_pt == self.fft_len:

                windowed = self.spectrum_amp_in * self.window * self.window_normalize

                fft_result = rfft(windowed)

                scaler = 2.0 * 2.0 / (self.fft_len * self.fft_len)
                spectrum_amp_tmp = np.abs(fft_result) ** 2 * scaler

                self.spectrum_amp_out_cum += spectrum_amp_tmp
                self.n_analysed += 1

                if self.hop_len < self.fft_len:
                    remaining = self.fft_len - self.hop_len
                    self.spectrum_amp_in[:remaining] = self.spectrum_amp_in[self.hop_len:]
                    self.spectrum_amp_pt = remaining
                else:
                    self.spectrum_amp_pt = 0

    def get_spectrum_amp_db(self) -> np.ndarray:

        if self.n_analysed >= self.n_average:

            self.spectrum_amp_out = self.spectrum_amp_out_cum / self.n_analysed

            self.spectrum_amp_out_db = 10.0 * np.log10(
                np.maximum(self.spectrum_amp_out, 1e-10)
            )

            self.spectrum_amp_out_cum.fill(0.0)                              
            self.n_analysed = 0

        return self.spectrum_amp_out_db

    def calculate_peak(self) -> None:

        self.get_spectrum_amp_db()

        min_db_threshold = 20 * np.log10(0.125 / 32768)
        self.max_amp_db = min_db_threshold
        max_amp_idx = 0

        for i in range(1, len(self.spectrum_amp_out_db)):
            if self.spectrum_amp_out_db[i] > self.max_amp_db:
                self.max_amp_db = self.spectrum_amp_out_db[i]
                max_amp_idx = i

        self.max_amp_freq = max_amp_idx * self.sample_rate / self.fft_len

        freq_resolution = self.sample_rate / self.fft_len
        if freq_resolution < self.max_amp_freq < (self.sample_rate / 2 - freq_resolution):
            id = int(round(self.max_amp_freq / self.sample_rate * self.fft_len))

            if 1 <= id < len(self.spectrum_amp_out_db) - 1:
                x1 = self.spectrum_amp_out_db[id - 1]
                x2 = self.spectrum_amp_out_db[id]
                x3 = self.spectrum_amp_out_db[id + 1]

                c = x2
                a = (x3 + x1) / 2.0 - x2
                b = (x3 - x1) / 2.0

                if a < 0:
                    x_peak = -b / (2.0 * a)
                    if abs(x_peak) < 1.0:
                        self.max_amp_freq += x_peak * freq_resolution
                        self.max_amp_db = (4 * a * c - b * b) / (4 * a)

    def get_rms(self) -> float:

        if self.cnt_rms > 8000 / 30:
            self.out_rms = np.sqrt(self.cum_rms / self.cnt_rms * 2.0)
            self.cum_rms = 0.0
            self.cnt_rms = 0
        return self.out_rms

    def n_elem_spectrum_amp(self) -> int:

        return self.n_analysed

    def clear(self) -> None:

        self.spectrum_amp_pt = 0
        self.spectrum_amp_in.fill(0.0)
        self.spectrum_amp_out.fill(0.0)
        self.spectrum_amp_out_db.fill(-np.inf)
        self.spectrum_amp_out_cum.fill(0.0)
        self.n_analysed = 0

MAX_BUFFER_SECONDS = 10
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)

MIN_DETECTION_INTERVAL_SECONDS = 2.0

DURATION_TOLERANCE_MS = 0

def parabolic(f, x):

    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    try:
        x1 = f[x - 1]
        x2 = f[x]
        x3 = f[x + 1]

        c = x2
        a = (x3 + x1) / 2.0 - x2
        b = (x3 - x1) / 2.0

        if abs(a) < 1e-10:
            return float(x), float(f[x])

        if a >= 0:
            return float(x), float(f[x])

        xPeak = -b / (2.0 * a)

        if abs(xPeak) >= 1.0:
            return float(x), float(f[x])

        yv = (4 * a * c - b * b) / (4 * a)

        xv = x + xPeak

        return xv, yv
    except (ZeroDivisionError, IndexError):
        return float(x), float(f[x])

def calculate_peak_prominence(
    magnitudes: np.ndarray, peak_idx: int, window_size: int = 25
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
    min_separation_hz: float = 50.0,
    min_freq: float = 50.0,
    max_freq: float = 4000.0,
) -> List[float]:

    if len(sig) == 0:
        return []

    N = len(sig)
    window = hanning(N)
    windowed = sig * window
    scaling = np.sum(window) / N
    f = rfft(windowed) / scaling

    magnitudes = np.abs(f)
    max_magnitude = np.max(magnitudes)
    if max_magnitude == 0:
        return []

    magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
    max_magnitude_db = 20 * np.log10(max_magnitude + 1e-10)

    min_db_threshold = 20 * np.log10(0.125 / 32768)
    if max_magnitude_db < min_db_threshold:
        return []

    if max_magnitude < magnitude_threshold:
        return []

    signal_rms = np.sqrt(np.mean(sig**2))
    if signal_rms < 1e-6:
        return []

    candidate_peaks = []
    for i in range(1, len(magnitudes) - 1):
        if (
            magnitudes[i] > magnitudes[i - 1]
            and magnitudes[i] > magnitudes[i + 1]
        ):

            if magnitudes_db[i] > min_db_threshold:
                candidate_peaks.append(i)

    if len(candidate_peaks) == 0:
        return []

    peak_data = []

    for peak_idx in candidate_peaks:

        if peak_idx < 1 or peak_idx >= len(magnitudes_db) - 1:

            peak_freq = peak_idx * fs / N
        else:

            true_i, _ = parabolic(magnitudes_db, peak_idx)
            peak_freq = true_i * fs / N

            if abs(true_i - peak_idx) >= 1.0:

                peak_freq = peak_idx * fs / N

        if peak_freq < min_freq or peak_freq > max_freq:
            continue

        prominence = calculate_peak_prominence(magnitudes, peak_idx)
        width = calculate_peak_width(magnitudes, peak_idx)
        peak_mag = magnitudes[peak_idx]

        if peak_mag < magnitude_threshold:
            continue

        min_prominence = peak_mag * min_prominence_ratio
        if prominence < min_prominence:
            continue

        if width > max_width_bins:
            continue

        peak_data.append(
            {
                "index": peak_idx,
                "freq": peak_freq,
                "magnitude": peak_mag,
            }
        )

    if len(peak_data) == 0:
        return []

    peak_data.sort(key=lambda x: x["magnitude"], reverse=True)

    filtered_peaks = []
    min_separation_bins = max(1, int((min_separation_hz * N) / fs))

    for peak in peak_data:

        too_close = False
        for accepted_peak in filtered_peaks:
            bin_separation = abs(peak["index"] - accepted_peak["index"])
            if bin_separation < min_separation_bins:
                too_close = True
                break

        if not too_close:
            filtered_peaks.append(peak)

    if len(filtered_peaks) > 0:
        filtered_peaks = [filtered_peaks[0]]

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

def extract_peaks_from_chunk(samples: np.ndarray, chunk_ms: float = 100.0) -> List[float]:

    if len(samples) == 0:
        return []

    peaks = freq_from_fft(samples, SAMPLE_RATE)
    return peaks

def calculate_fft_resolution_hz(window_size_samples: int, sample_rate: int = SAMPLE_RATE) -> float:

    if window_size_samples <= 0:
        return float("inf")
    return sample_rate / window_size_samples

def calculate_default_range_hz(
    window_size_samples: int,
    sample_rate: int = SAMPLE_RATE,
    multiplier: float = 1.5,
) -> float:

    resolution = calculate_fft_resolution_hz(window_size_samples, sample_rate)
    return resolution * multiplier

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

        self.FFT_LEN = 8192
        self.HOP_LEN = 4096
        self.N_AVERAGE = 1
        self.WINDOW_NAME = "Hanning"

        self.stft = STFT(
            fft_len=self.FFT_LEN,
            hop_len=self.HOP_LEN,
            sample_rate=SAMPLE_RATE,
            window_name=self.WINDOW_NAME,
            n_average=self.N_AVERAGE
        )

        self.hop_time_ms = (self.HOP_LEN / SAMPLE_RATE) * 1000.0

        self.MAX_FREQUENCY_PEAKS_SIZE = max(
            1, int((MAX_BUFFER_SECONDS * 1000) / self.hop_time_ms)
        )
        self.max_frequency_peaks: List[Optional[float]] = [None] * self.MAX_FREQUENCY_PEAKS_SIZE
        self.max_frequency_peaks_mutex = threading.Lock()
        self.max_frequency_peaks_index = 0

        self.audio_queue = queue.Queue(maxsize=100)

        self.peak_storage_thread: Optional[threading.Thread] = None
        self.tone_detection_thread: Optional[threading.Thread] = None
        self.threads_running = threading.Event()

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
        self.new_tone_config = new_tone_config

        self.fft_resolution_hz = SAMPLE_RATE / self.FFT_LEN
        self.default_range_hz = self.fft_resolution_hz * 1.5

        if self.new_tone_range_hz < self.default_range_hz:
            print(
                f"[TONE DETECTION] Adjusting new_tone_range_hz from "
                f"{self.new_tone_range_hz:.1f} to {self.default_range_hz:.1f} Hz "
                f"(FFT resolution: {self.fft_resolution_hz:.1f} Hz)"
            )
            self.new_tone_range_hz = self.default_range_hz

        if passthrough_config is None:
            passthrough_config = {"tone_passthrough": False, "passthrough_channel": ""}
        self.passthrough_config = passthrough_config

        self.last_detection_time: Dict[str, float] = {}
        for tone_def in tone_definitions:
            tone_id = tone_def["tone_id"]
            self.last_detection_time[tone_id] = 0.0

        self.last_new_tone_pair: Dict[str, Any] = {"tone_a": 0.0, "tone_b": 0.0}
        self.last_tone_detection_time = 0.0

        self.SIMILARITY_THRESHOLD = 0.30

        self.threads_running.set()
        self._start_threads()

    def _start_threads(self):

        self.peak_storage_thread = threading.Thread(
            target=self._peak_storage_worker,
            name=f"PeakStorage-{self.channel_id}",
            daemon=True
        )
        self.peak_storage_thread.start()
        print(f"[TONE DETECT] Started peak storage thread for channel {self.channel_id}")

        self.tone_detection_thread = threading.Thread(
            target=self._tone_detection_worker,
            name=f"ToneDetection-{self.channel_id}",
            daemon=True
        )
        self.tone_detection_thread.start()
        print(f"[TONE DETECT] Started tone detection thread for channel {self.channel_id}")

    def _peak_storage_worker(self):

        while self.threads_running.is_set():
            try:
                try:
                    audio_samples = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self.stft.feed_data(audio_samples)

                if self.stft.n_elem_spectrum_amp() >= self.N_AVERAGE:

                    self.stft.calculate_peak()

                    max_peak = None
                    if not np.isnan(self.stft.max_amp_freq) and self.stft.max_amp_freq > 0:

                        min_db_threshold = 20 * np.log10(0.125 / 32768)
                        if self.stft.max_amp_db > min_db_threshold:
                            max_peak = self.stft.max_amp_freq

                    with self.max_frequency_peaks_mutex:
                        self.max_frequency_peaks[self.max_frequency_peaks_index] = max_peak
                        self.max_frequency_peaks_index = (self.max_frequency_peaks_index + 1) % self.MAX_FREQUENCY_PEAKS_SIZE

                self.audio_queue.task_done()

            except Exception as e:
                print(f"[PEAK STORAGE] ERROR in peak storage worker for {self.channel_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _get_recent_peaks(self, num_elements: int) -> List[Optional[float]]:

        with self.max_frequency_peaks_mutex:
            current_index = self.max_frequency_peaks_index
            peaks = self.max_frequency_peaks

            num_elements = min(num_elements, self.MAX_FREQUENCY_PEAKS_SIZE)

            result = []

            for i in range(num_elements):

                idx = (current_index - num_elements + i) % self.MAX_FREQUENCY_PEAKS_SIZE
                result.append(peaks[idx])

            return result

    def _tone_detection_worker(self):

        DETECTION_INTERVAL = 0.1

        while self.threads_running.is_set():
            try:
                time.sleep(DETECTION_INTERVAL)

                for tone_def in self.tone_definitions:
                    tone_a_ms = tone_def.get('tone_a_length_ms', 1000)
                    tone_b_ms = tone_def.get('tone_b_length_ms', 1000)
                    tone_a_target = tone_def.get('tone_a', 0.0)
                    tone_b_target = tone_def.get('tone_b', 0.0)
                    tone_a_range = tone_def.get('tone_a_range', 10)
                    tone_b_range = tone_def.get('tone_b_range', 10)

                    if tone_a_range < self.default_range_hz:
                        tone_a_range = self.default_range_hz
                    if tone_b_range < self.default_range_hz:
                        tone_b_range = self.default_range_hz

                    elements_a = int(round(tone_a_ms / self.hop_time_ms))
                    elements_b = int(round(tone_b_ms / self.hop_time_ms))
                    total_elements = elements_a + elements_b

                    recent_peaks = self._get_recent_peaks(total_elements)
                    if len(recent_peaks) < total_elements:
                        continue

                    tone_a_elements = recent_peaks[-total_elements:-elements_b] if elements_b > 0 else recent_peaks[-elements_a:]
                    tone_b_elements = recent_peaks[-elements_b:]

                    is_very_short_a = tone_a_ms < 1500
                    is_very_short_b = tone_b_ms < 1500
                    is_short_tone = is_very_short_a or is_very_short_b

                    boundary_check_size = min(3, max(1, int(elements_a * 0.1), int(elements_b * 0.1)))

                    tone_a_end = tone_a_elements[-boundary_check_size:] if len(tone_a_elements) >= boundary_check_size else tone_a_elements
                    tone_b_start = tone_b_elements[:boundary_check_size] if len(tone_b_elements) >= boundary_check_size else tone_b_elements

                    boundary_silence_count = sum(1 for f in tone_a_end if f is None) + sum(1 for f in tone_b_start if f is None)
                    boundary_total = len(tone_a_end) + len(tone_b_start)

                    if is_very_short_a or is_very_short_b:
                        silence_threshold = 0.8
                    elif tone_a_ms < 2500 or tone_b_ms < 2500:
                        silence_threshold = 0.7
                    else:
                        silence_threshold = 0.5

                    if boundary_total > 0 and boundary_silence_count > boundary_total * silence_threshold:
                        continue

                    tone_a_in_range = [f for f in tone_a_elements if f is not None and abs(f - tone_a_target) <= tone_a_range]
                    tone_b_in_range = [f for f in tone_b_elements if f is not None and abs(f - tone_b_target) <= tone_b_range]

                    tone_a_matches = len(tone_a_in_range)
                    tone_b_matches = len(tone_b_in_range)

                    tone_a_detected_freq = self._find_convergence_point(tone_a_in_range, tone_a_range) if tone_a_in_range else None
                    tone_b_detected_freq = self._find_convergence_point(tone_b_in_range, tone_b_range) if tone_b_in_range else None

                    tone_a_valid_count = len([f for f in tone_a_elements if f is not None])
                    tone_b_valid_count = len([f for f in tone_b_elements if f is not None])

                    tone_a_match_ratio = tone_a_matches / tone_a_valid_count if tone_a_valid_count > 0 else 0
                    tone_b_match_ratio = tone_b_matches / tone_b_valid_count if tone_b_valid_count > 0 else 0

                    tone_a_threshold = 0.20 if is_very_short_a else (0.25 if tone_a_ms < 2500 else self.SIMILARITY_THRESHOLD)
                    tone_b_threshold = 0.20 if is_very_short_b else (0.25 if tone_b_ms < 2500 else self.SIMILARITY_THRESHOLD)

                    if tone_a_match_ratio >= tone_a_threshold and tone_b_match_ratio >= tone_b_threshold:
                        if tone_a_detected_freq is None or tone_b_detected_freq is None:
                            continue

                        is_very_short_a = tone_a_ms < 1500
                        is_very_short_b = tone_b_ms < 1500
                        is_short_tone = is_very_short_a or is_very_short_b

                        tone_a_end_check_size = min(5, max(1, int(elements_a * 0.2)))
                        tone_a_end_check = [f for f in tone_a_elements[-tone_a_end_check_size:] if f is not None]
                        tone_a_end_valid = False
                        if tone_a_end_check:
                            tone_a_end_matches = sum(1 for f in tone_a_end_check if abs(f - tone_a_detected_freq) <= tone_a_range)
                            tone_a_end_ratio = tone_a_end_matches / len(tone_a_end_check)

                            if is_very_short_a:
                                tone_a_end_valid = tone_a_end_ratio >= 0.2
                            elif tone_a_ms < 2500:
                                tone_a_end_valid = tone_a_end_ratio >= 0.3
                            else:
                                tone_a_end_valid = tone_a_end_ratio >= 0.5

                        tone_b_start_check_size = min(5, max(1, int(elements_b * 0.2)))
                        tone_b_start_check = [f for f in tone_b_elements[:tone_b_start_check_size] if f is not None]
                        tone_b_start_valid = False
                        if tone_b_start_check:
                            tone_b_start_matches = sum(1 for f in tone_b_start_check if abs(f - tone_b_detected_freq) <= tone_b_range)
                            tone_b_start_ratio = tone_b_start_matches / len(tone_b_start_check)

                            if is_very_short_b:
                                tone_b_start_valid = tone_b_start_ratio >= 0.2
                            elif tone_b_ms < 2500:
                                tone_b_start_valid = tone_b_start_ratio >= 0.3
                            else:
                                tone_b_start_valid = tone_b_start_ratio >= 0.5

                        if is_very_short_a or is_very_short_b:

                            if not (tone_a_end_valid or tone_b_start_valid):
                                continue
                        elif is_short_tone:

                            if not (tone_a_end_valid or tone_b_start_valid):
                                continue
                        else:

                            if not (tone_a_end_valid and tone_b_start_valid):
                                continue

                        min_interval = (tone_a_ms + tone_b_ms) / 1000.0
                        if time.time() - self.last_tone_detection_time < min_interval:
                            continue

                        tone_a_freq = tone_a_detected_freq
                        tone_b_freq = tone_b_detected_freq

                        if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                            continue

                        if abs(tone_a_freq - tone_b_freq) <= 5:
                            continue

                        tone_a_freq = round(tone_a_freq, 1)
                        tone_b_freq = round(tone_b_freq, 1)

                        self.last_tone_detection_time = time.time()
                        self._defined_tone_alert(tone_def, tone_a_freq, tone_b_freq)
                        self._defined_tone_passthrough(tone_def)
                        self._defined_tone_recording(tone_def)

                if self.detect_new_tones:
                    elements_new = int(round(self.new_tone_length_ms / self.hop_time_ms))
                    total_elements_new = elements_new * 2

                    recent_peaks = self._get_recent_peaks(total_elements_new)
                    if len(recent_peaks) >= total_elements_new:

                        tone_a_elements = recent_peaks[-total_elements_new:-elements_new]
                        tone_b_elements = recent_peaks[-elements_new:]

                        boundary_check_size = min(3, max(1, int(elements_new * 0.1)))

                        tone_a_end = tone_a_elements[-boundary_check_size:] if len(tone_a_elements) >= boundary_check_size else tone_a_elements
                        tone_b_start = tone_b_elements[:boundary_check_size] if len(tone_b_elements) >= boundary_check_size else tone_b_elements

                        boundary_silence_count = sum(1 for f in tone_a_end if f is None) + sum(1 for f in tone_b_start if f is None)
                        boundary_total = len(tone_a_end) + len(tone_b_start)

                        if boundary_total > 0 and boundary_silence_count > boundary_total * 0.5:
                            continue

                        tone_a_freq = self._find_convergence_point(tone_a_elements, self.new_tone_range_hz)
                        tone_b_freq = self._find_convergence_point(tone_b_elements, self.new_tone_range_hz)

                        if tone_a_freq and tone_b_freq:

                            if abs(tone_a_freq - tone_b_freq) <= 50:
                                continue

                            tone_a_end_check_size = min(5, max(1, int(elements_new * 0.2)))
                            tone_a_end_check = [f for f in tone_a_elements[-tone_a_end_check_size:] if f is not None]
                            tone_a_end_valid = False
                            if tone_a_end_check:
                                tone_a_end_matches = sum(1 for f in tone_a_end_check if abs(f - tone_a_freq) <= self.new_tone_range_hz)
                                tone_a_end_ratio = tone_a_end_matches / len(tone_a_end_check)
                                tone_a_end_valid = tone_a_end_ratio >= 0.5

                            tone_b_start_check_size = min(5, max(1, int(elements_new * 0.2)))
                            tone_b_start_check = [f for f in tone_b_elements[:tone_b_start_check_size] if f is not None]
                            tone_b_start_valid = False
                            if tone_b_start_check:
                                tone_b_start_matches = sum(1 for f in tone_b_start_check if abs(f - tone_b_freq) <= self.new_tone_range_hz)
                                tone_b_start_ratio = tone_b_start_matches / len(tone_b_start_check)
                                tone_b_start_valid = tone_b_start_ratio >= 0.5

                            if not (tone_a_end_valid and tone_b_start_valid):
                                continue

                            tone_a_stable = sum(1 for f in tone_a_elements if f is not None and abs(f - tone_a_freq) <= self.new_tone_range_hz)
                            tone_b_stable = sum(1 for f in tone_b_elements if f is not None and abs(f - tone_b_freq) <= self.new_tone_range_hz)

                            tone_a_stability = tone_a_stable / len([f for f in tone_a_elements if f is not None]) if len([f for f in tone_a_elements if f is not None]) > 0 else 0
                            tone_b_stability = tone_b_stable / len([f for f in tone_b_elements if f is not None]) if len([f for f in tone_b_elements if f is not None]) > 0 else 0

                            if tone_a_stability >= self.SIMILARITY_THRESHOLD and tone_b_stability >= self.SIMILARITY_THRESHOLD:

                                matches_defined_tone = False
                                for tone_def in self.tone_definitions:
                                    tone_a_target = tone_def.get('tone_a', 0.0)
                                    tone_b_target = tone_def.get('tone_b', 0.0)
                                    tone_a_range = tone_def.get('tone_a_range', 10)
                                    tone_b_range = tone_def.get('tone_b_range', 10)

                                    if tone_a_range < self.default_range_hz:
                                        tone_a_range = self.default_range_hz
                                    if tone_b_range < self.default_range_hz:
                                        tone_b_range = self.default_range_hz

                                    if (abs(tone_a_freq - tone_a_target) <= tone_a_range and 
                                        abs(tone_b_freq - tone_b_target) <= tone_b_range):
                                        matches_defined_tone = True
                                        break
                                    elif (abs(tone_a_freq - tone_b_target) <= tone_b_range and 
                                          abs(tone_b_freq - tone_a_target) <= tone_a_range):

                                        matches_defined_tone = True
                                        break

                                if matches_defined_tone:

                                    continue

                                min_interval = self.new_tone_length_seconds * 2 + 0.2
                                if time.time() - self.last_tone_detection_time < min_interval:
                                    continue

                                if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                                    continue

                                self.last_tone_detection_time = time.time()

                                tone_a_freq = round(tone_a_freq, 1)
                                tone_b_freq = round(tone_b_freq, 1)

                                print(
                                    f"[NEW TONE PAIR] Channel {self.channel_id}: "
                                    f"A={tone_a_freq:.1f} Hz, B={tone_b_freq:.1f} Hz "
                                    f"(each {self.new_tone_length_seconds:.2f} s, "
                                    f"±{self.new_tone_range_hz} Hz stable, "
                                    f"stability: A={tone_a_stability:.1%}, B={tone_b_stability:.1%})"
                                )

                                if MQTT_AVAILABLE:
                                    publish_new_tone_pair(tone_a_freq, tone_b_freq)

            except Exception as e:
                print(f"[TONE DETECTION] ERROR in tone detection worker for {self.channel_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _calculate_similarity_percentage(
        self, elements: List[Optional[float]], target_freq: float, range_hz: float
    ) -> float:

        if not elements:
            return 0.0

        similar_count = 0
        total_count = 0

        for freq in elements:
            if freq is not None:
                total_count += 1
                if is_frequency_in_range(freq, target_freq, range_hz):
                    similar_count += 1

        if total_count == 0:
            return 0.0

        return similar_count / total_count

    def _get_average_frequency(
        self, elements: List[Optional[float]], target_freq: float, range_hz: float
    ) -> float:

        similar_freqs = []
        for freq in elements:
            if freq is not None:
                if is_frequency_in_range(freq, target_freq, range_hz):
                    similar_freqs.append(freq)

        if similar_freqs:
            return sum(similar_freqs) / len(similar_freqs)
        return target_freq

    def _get_dominant_frequency(self, elements: List[Optional[float]]) -> Optional[float]:

        valid_freqs = [f for f in elements if f is not None]
        if not valid_freqs:
            return None

        groups = {}
        for freq in valid_freqs:
            found_group = False
            for group_freq in groups.keys():
                if abs(freq - group_freq) <= 10.0:
                    groups[group_freq].append(freq)
                    found_group = True
                    break

            if not found_group:
                groups[freq] = [freq]

        if not groups:
            return None

        largest_group = max(groups.items(), key=lambda x: len(x[1]))
        return sum(largest_group[1]) / len(largest_group[1])

    def _find_convergence_point(self, elements: List[Optional[float]], tolerance_hz: float) -> Optional[float]:

        valid_freqs = [f for f in elements if f is not None]
        if not valid_freqs:
            return None

        if len(valid_freqs) == 1:
            return valid_freqs[0]

        clusters = []
        for freq in valid_freqs:

            added_to_cluster = False
            for cluster in clusters:

                cluster_centroid = sum(cluster) / len(cluster)

                if abs(freq - cluster_centroid) <= tolerance_hz:
                    cluster.append(freq)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                clusters.append([freq])

        if not clusters:
            return None

        largest_cluster = max(clusters, key=len)

        convergence_freq = sum(largest_cluster) / len(largest_cluster)

        similar_count = sum(1 for freq in valid_freqs if abs(freq - convergence_freq) <= tolerance_hz)
        similarity_ratio = similar_count / len(valid_freqs) if len(valid_freqs) > 0 else 0.0

        if similarity_ratio >= 0.30:
            return convergence_freq
        else:
            return None

    def stop_threads(self):

        self.threads_running.clear()
        if self.peak_storage_thread and self.peak_storage_thread.is_alive():
            self.peak_storage_thread.join(timeout=1.0)
        if self.tone_detection_thread and self.tone_detection_thread.is_alive():
            self.tone_detection_thread.join(timeout=1.0)
        print(f"[TONE DETECT] Stopped threads for channel {self.channel_id}")

    def add_audio_samples(self, samples: np.ndarray):

        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:

            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(samples)
            except queue.Empty:
                pass

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

    def _defined_tone_alert(self, tone_def: Dict[str, Any], tone_a_freq: float, tone_b_freq: float):
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
                    + f"    Duration:     (target: {tone_def['tone_a_length_ms']} ms, "
                    f"tolerance: ±{DURATION_TOLERANCE_MS} ms)\n"
                    + "  \n"
                    + "  Tone B Details:\n"
                    + f"    Detected:     {tone_b_freq:.1f} Hz\n"
                    + f"    Target:       {tone_def['tone_b']:.1f} Hz "
                    f"±{tone_def.get('tone_b_range', 10)} Hz\n"
                    + f"    Duration:     (target: {tone_def['tone_b_length_ms']} ms, "
                    f"tolerance: ±{DURATION_TOLERANCE_MS} ms)\n"
                    + "  \n"
                    + "  Record Length:  "
                    f"{tone_def.get('record_length_ms', 0)} ms\n"
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

    def _defined_tone_passthrough(self, tone_def: Dict[str, Any]):

        try:
            from passthrough import global_passthrough_manager

            print(f"[PASSTHROUGH DEBUG] _defined_tone_passthrough called for channel {self.channel_id}, passthrough_config={self.passthrough_config}")
            if self.passthrough_config.get("tone_passthrough", False):
                target_channel = self.passthrough_config.get(
                    "passthrough_channel", ""
                )
                record_length_ms = tone_def.get("record_length_ms", 0)
                if target_channel and record_length_ms > 0:
                    print(
                        f"[PASSTHROUGH] Attempting to start passthrough: "
                        f"{self.channel_id} -> {target_channel}, "
                        f"duration={record_length_ms} ms"
                    )
                    success = global_passthrough_manager.start_passthrough(
                        self.channel_id, target_channel, record_length_ms
                    )
                    if success:
                        print(
                            f"[PASSTHROUGH] Successfully triggered: "
                            f"{self.channel_id} -> {target_channel}, "
                            f"duration={record_length_ms} ms"
                        )
                    else:
                        print(
                            f"[PASSTHROUGH] FAILED to start passthrough: "
                            f"{self.channel_id} -> {target_channel}, "
                            f"duration={record_length_ms} ms"
                        )
                else:
                    if not target_channel:
                        print(
                            f"[PASSTHROUGH] WARNING: No target channel configured for passthrough on {self.channel_id}"
                        )
                    if record_length_ms <= 0:
                        print(
                            f"[PASSTHROUGH] WARNING: Invalid record_length_ms ({record_length_ms}) for passthrough on {self.channel_id}"
                        )
            else:
                print(f"[PASSTHROUGH DEBUG] Passthrough not enabled for channel {self.channel_id} (tone_passthrough=False or missing from config)")
        except Exception as e:
            print(
                f"[PASSTHROUGH] ERROR: Failed to trigger passthrough: {e}"
            )
            import traceback
            traceback.print_exc()

    def _defined_tone_recording(self, tone_def: Dict[str, Any]):

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

_channel_detectors: Dict[str, ChannelToneDetector] = {}
_detectors_mutex = threading.Lock()

def stop_all_detectors():

    with _detectors_mutex:
        for channel_id, detector in _channel_detectors.items():
            try:
                detector.stop_threads()
                print(f"[TONE DETECTION] Stopped threads for channel {channel_id}")
            except Exception as e:
                print(f"[TONE DETECTION] ERROR stopping threads for {channel_id}: {e}")
        _channel_detectors.clear()

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
            old_detector = _channel_detectors.pop(channel_id, None)
            if old_detector:
                old_detector.stop_threads()
            print(
                f"[TONE DETECTION] No tone definitions found for "
                f"channel {channel_id}"
            )

def add_audio_samples_for_channel(channel_id: str, filtered_audio: np.ndarray) -> None:

    try:
        from audio_visualizer import add_audio_to_visualizer
        add_audio_to_visualizer(channel_id, filtered_audio)
    except ImportError:
        pass

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

    return None
