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

MAX_BUFFER_SECONDS = 10
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)

MIN_DETECTION_INTERVAL_SECONDS = 2.0

DURATION_TOLERANCE_MS = 0

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
    
    signal_rms = np.sqrt(np.mean(sig**2))
    if signal_rms < 1e-6:
        return []

    candidate_peaks = []
    for i in range(1, len(magnitudes) - 1):
        if (
            magnitudes[i] > magnitudes[i - 1]
            and magnitudes[i] > magnitudes[i + 1]
        ):
            candidate_peaks.append(i)

    if len(candidate_peaks) == 0:
        return []
    
    peak_data = []
    fft_len = len(magnitudes)

    for peak_idx in candidate_peaks:
        true_i = parabolic(np.log(magnitudes + 1e-10), peak_idx)[0]
        peak_freq = true_i * fs / N
        
        if peak_freq < min_freq or peak_freq > max_freq:
            continue
        
        prominence = calculate_peak_prominence(magnitudes, peak_idx)
        width = calculate_peak_width(magnitudes, peak_idx)
        peak_mag = magnitudes[peak_idx]

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
    """Extract peak frequencies from a single audio chunk using FFT."""
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
        
        # Global array: 100 elements for 10s audio (100ms per element)
        self.CHUNK_MS = 100.0
        self.CHUNK_SIZE = int(SAMPLE_RATE * self.CHUNK_MS / 1000.0)
        self.MAX_FREQUENCY_PEAKS_SIZE = max(
            1, int((MAX_BUFFER_SECONDS * 1000) / self.CHUNK_MS)
        )
        self.max_frequency_peaks: List[Optional[float]] = [None] * self.MAX_FREQUENCY_PEAKS_SIZE
        self.max_frequency_peaks_mutex = threading.Lock()
        self.max_frequency_peaks_index = 0  # Circular buffer index
        
        # Audio queue for Thread 1 (peak storage)
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Thread management
        self.peak_storage_thread: Optional[threading.Thread] = None
        self.tone_detection_thread: Optional[threading.Thread] = None
        self.threads_running = threading.Event()
        
        # Frequency filters
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
        self.new_tone_config = new_tone_config

        self.fft_resolution_hz = calculate_fft_resolution_hz(
            self.CHUNK_SIZE, SAMPLE_RATE
        )
        self.default_range_hz = calculate_default_range_hz(
            self.CHUNK_SIZE, SAMPLE_RATE, multiplier=1.5
        )

        if self.new_tone_range_hz < self.default_range_hz:
            print(
                f"[TONE DETECTION] Adjusting new_tone_range_hz from "
                f"{self.new_tone_range_hz:.1f} to {self.default_range_hz:.1f} Hz "
                f"(FFT resolution: {self.fft_resolution_hz:.1f} Hz)"
            )
            self.new_tone_range_hz = self.default_range_hz

        # Passthrough configuration
        if passthrough_config is None:
            passthrough_config = {"tone_passthrough": False, "passthrough_channel": ""}
        self.passthrough_config = passthrough_config

        # Detection state
        self.last_detection_time: Dict[str, float] = {}
        for tone_def in tone_definitions:
            tone_id = tone_def["tone_id"]
            self.last_detection_time[tone_id] = 0.0

        self.last_new_tone_pair: Dict[str, Any] = {"tone_a": 0.0, "tone_b": 0.0}
        self.last_tone_detection_time = 0.0
        
        # Similarity threshold (70%)
        self.SIMILARITY_THRESHOLD = 0.30
        
        self.threads_running.set()
        self._start_threads()
    
    def _start_threads(self):
        """Start the two independent threads for tone detection."""
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
        """Thread 1: Process CHUNK_MS audio windows and store max frequency peak in global array."""
        chunk_buffer: List[float] = []
        
        while self.threads_running.is_set():
            try:
                try:
                    audio_samples = self.audio_queue.get(timeout=0.1)
                    chunk_buffer.extend(audio_samples.tolist())
                except queue.Empty:
                    continue
                
                while len(chunk_buffer) >= self.CHUNK_SIZE:
                    chunk_samples = np.array(chunk_buffer[:self.CHUNK_SIZE], dtype=np.float32)
                    chunk_buffer = chunk_buffer[self.CHUNK_SIZE:]
                    
                    peaks = extract_peaks_from_chunk(chunk_samples, self.CHUNK_MS)
                    
                    max_peak = None
                    if peaks:
                        max_peak = max(peaks)
                    
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
        """Get the most recent N elements from the circular buffer.
        
        Args:
            num_elements: Number of elements to retrieve
            
        Returns:
            List of frequency peaks (oldest first, newest last)
        """
        with self.max_frequency_peaks_mutex:
            current_index = self.max_frequency_peaks_index
            peaks = self.max_frequency_peaks
            
            # Limit to available elements
            num_elements = min(num_elements, self.MAX_FREQUENCY_PEAKS_SIZE)
            
            result = []
            # Get elements starting from (current_index - num_elements) backwards
            for i in range(num_elements):
                # Calculate index going backwards from current_index
                idx = (current_index - num_elements + i) % self.MAX_FREQUENCY_PEAKS_SIZE
                result.append(peaks[idx])
            
            return result
    
    def _tone_detection_worker(self):
        """Thread 2: Check global array and calculate similarity percentages for tone detection."""
        DETECTION_INTERVAL = 0.1
        
        while self.threads_running.is_set():
            try:
                time.sleep(DETECTION_INTERVAL)
                
                # Check defined tones
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
                    
                    # Calculate number of elements (CHUNK_MS per element)
                    elements_a = int(round(tone_a_ms / self.CHUNK_MS))
                    elements_b = int(round(tone_b_ms / self.CHUNK_MS))
                    total_elements = elements_a + elements_b
                    
                    # Get recent peaks
                    recent_peaks = self._get_recent_peaks(total_elements)
                    if len(recent_peaks) < total_elements:
                        continue
                    
                    # Extract tone A and tone B elements (most recent is last)
                    tone_a_elements = recent_peaks[-total_elements:-elements_b] if elements_b > 0 else recent_peaks[-elements_a:]
                    tone_b_elements = recent_peaks[-elements_b:]
                    
                    # Calculate similarity percentages
                    tone_a_similarity = self._calculate_similarity_percentage(
                        tone_a_elements, tone_a_target, tone_a_range
                    )
                    tone_b_similarity = self._calculate_similarity_percentage(
                        tone_b_elements, tone_b_target, tone_b_range
                    )
                    
                    # Check if both tones meet 70% similarity threshold
                    if tone_a_similarity >= self.SIMILARITY_THRESHOLD and tone_b_similarity >= self.SIMILARITY_THRESHOLD:
                        # Prevent duplicate detections
                        min_interval = (tone_a_ms + tone_b_ms) / 1000.0
                        if time.time() - self.last_tone_detection_time < min_interval:
                            continue
                        
                        # Get average frequencies
                        tone_a_freq = self._get_average_frequency(tone_a_elements, tone_a_target, tone_a_range)
                        tone_b_freq = self._get_average_frequency(tone_b_elements, tone_b_target, tone_b_range)
                        
                        # Apply frequency filters
                        if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                            continue
                        
                        # Ensure tones are different
                        if abs(tone_a_freq - tone_b_freq) <= 50:
                            continue

                        tone_a_freq = round(tone_a_freq, 0)
                        tone_b_freq = round(tone_b_freq, 0)
                        
                        self.last_tone_detection_time = time.time()
                        self._defined_tone_alert(tone_def, tone_a_freq, tone_b_freq)
                        self._defined_tone_passthrough(tone_def)
                        self._defined_tone_recording(tone_def)
                
                # Check for new tones
                if self.detect_new_tones:
                    elements_new = int(round(self.new_tone_length_ms / self.CHUNK_MS))
                    total_elements_new = elements_new * 2
                    
                    recent_peaks = self._get_recent_peaks(total_elements_new)
                    if len(recent_peaks) >= total_elements_new:
                        # Extract two consecutive windows
                        tone_a_elements = recent_peaks[-total_elements_new:-elements_new]
                        tone_b_elements = recent_peaks[-elements_new:]
                        
                        # Find dominant frequency in each window
                        tone_a_freq = self._get_dominant_frequency(tone_a_elements)
                        tone_b_freq = self._get_dominant_frequency(tone_b_elements)
                        
                        if tone_a_freq and tone_b_freq:
                            # Ensure tones are different
                            if abs(tone_a_freq - tone_b_freq) <= 50:
                                continue
                            
                            # Calculate similarity percentages using dominant frequencies
                            tone_a_similarity = self._calculate_similarity_percentage(
                                tone_a_elements, tone_a_freq, self.new_tone_range_hz
                            )
                            tone_b_similarity = self._calculate_similarity_percentage(
                                tone_b_elements, tone_b_freq, self.new_tone_range_hz
                            )
                            print(f"tone_a_freq: {tone_a_freq}, tone_b_freq: {tone_b_freq}")
                            print(f"tone_a_similarity: {tone_a_similarity}, tone_b_similarity: {tone_b_similarity}")
                            
                            if tone_a_similarity >= self.SIMILARITY_THRESHOLD and tone_b_similarity >= self.SIMILARITY_THRESHOLD:
                                # Prevent duplicate detections
                                min_interval = self.new_tone_length_seconds * 2 + 0.2
                                if time.time() - self.last_tone_detection_time < min_interval:
                                    continue
                                
                                # Apply frequency filters
                                if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                                    continue
                                
                                self.last_tone_detection_time = time.time()
                                
                                tone_a_freq = round(tone_a_freq, 0)
                                tone_b_freq = round(tone_b_freq, 0)
                                
                                print(
                                    f"[NEW TONE PAIR] Channel {self.channel_id}: "
                                    f"A={tone_a_freq:.1f} Hz, B={tone_b_freq:.1f} Hz "
                                    f"(each {self.new_tone_length_seconds:.2f} s, "
                                    f"±{self.new_tone_range_hz} Hz stable, "
                                    f"similarity: A={tone_a_similarity:.1%}, B={tone_b_similarity:.1%})"
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
        """Calculate the percentage of similar elements in the given list.
        
        Args:
            elements: List of frequency values (or None)
            target_freq: Target frequency to match against
            range_hz: Frequency range tolerance
        
        Returns:
            Similarity percentage (0.0 to 1.0)
        """
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
        """Get the average frequency of similar elements.
        
        Args:
            elements: List of frequency values (or None)
            target_freq: Target frequency to match against
            range_hz: Frequency range tolerance
        
        Returns:
            Average frequency of similar elements, or target_freq if none found
        """
        similar_freqs = []
        for freq in elements:
            if freq is not None:
                if is_frequency_in_range(freq, target_freq, range_hz):
                    similar_freqs.append(freq)
        
        if similar_freqs:
            return sum(similar_freqs) / len(similar_freqs)
        return target_freq
    
    def _get_dominant_frequency(self, elements: List[Optional[float]]) -> Optional[float]:
        """Get the dominant (most common) frequency in the elements.
        
        Groups similar frequencies together and returns the most common one.
        """
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
    
    def stop_threads(self):
        """Stop the two independent threads."""
        self.threads_running.clear()
        if self.peak_storage_thread and self.peak_storage_thread.is_alive():
            self.peak_storage_thread.join(timeout=1.0)
        if self.tone_detection_thread and self.tone_detection_thread.is_alive():
            self.tone_detection_thread.join(timeout=1.0)
        print(f"[TONE DETECT] Stopped threads for channel {self.channel_id}")

    def add_audio_samples(self, samples: np.ndarray):
        """Add audio samples to the queue for Thread 1 (peak storage thread)."""
        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            # If queue is full, remove oldest item and add new one
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
    """Stop all tone detection threads for graceful shutdown."""
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
    """Add audio samples for processing. Detection is handled by background threads."""
    with _detectors_mutex:
        detector = _channel_detectors.get(channel_id)
        if not detector:
            return None

    detector.add_audio_samples(filtered_audio)
    
    # Detection is now handled by background threads, so return None
    return None
