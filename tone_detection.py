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
    """
    Short Time Fourier Transform with overlapping windows (matching Android approach).
    Uses sliding window buffer with hopLen for overlap control.
    """
    def __init__(self, fft_len: int = 8192, hop_len: int = 4096, sample_rate: int = SAMPLE_RATE, 
                 window_name: str = "Hanning", n_average: int = 1):
        """
        Initialize STFT.
        
        Args:
            fft_len: FFT length (must be power of 2, matching Android)
            hop_len: Hop length for overlap (typically fft_len/2 for 50% overlap)
            sample_rate: Sample rate in Hz
            window_name: Window function name ("Hanning", "Blackman", etc.)
            n_average: Number of FFT results to average
        """
        if (fft_len & (fft_len - 1)) != 0:
            raise ValueError("fft_len must be a power of 2")
        
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.sample_rate = sample_rate
        self.n_average = n_average
        
        # Buffer for accumulating input samples
        self.spectrum_amp_in = np.zeros(fft_len, dtype=np.float32)
        self.spectrum_amp_pt = 0
        
        # Output arrays
        self.out_len = fft_len // 2 + 1
        self.spectrum_amp_out_cum = np.zeros(self.out_len, dtype=np.float64)
        self.spectrum_amp_out = np.zeros(self.out_len, dtype=np.float64)
        self.spectrum_amp_out_db = np.zeros(self.out_len, dtype=np.float64)
        
        self.n_analysed = 0
        
        # Initialize window function
        self.window = self._init_window_function(fft_len, window_name)
        
        # Window normalization factor (matching Android)
        window_sum = np.sum(self.window)
        self.window_normalize = fft_len / window_sum
        
        # Energy normalization factor
        window_energy = np.sum(self.window ** 2)
        self.wnd_energy_factor = fft_len / window_energy
        
        # RMS calculation
        self.cum_rms = 0.0
        self.cnt_rms = 0
        self.out_rms = 0.0
        
        # Peak detection results
        self.max_amp_freq = float('nan')
        self.max_amp_db = float('nan')
    
    def _init_window_function(self, fft_len: int, window_name: str) -> np.ndarray:
        """Initialize window function (matching Android STFT)."""
        if window_name == "Hanning":
            window = np.hanning(fft_len)
            # Match Android: 0.5*(1-cos(2*PI*i/(len-1)))*2
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fft_len) / (fft_len - 1))) * 2
        elif window_name == "Blackman":
            # Match Android: 0.42-0.5*cos(...)+0.08*cos(...)
            i = np.arange(fft_len)
            window = (0.42 - 0.5 * np.cos(2 * np.pi * i / (fft_len - 1)) + 
                     0.08 * np.cos(4 * np.pi * i / (fft_len - 1)))
        else:
            # Default to Hanning
            window = np.hanning(fft_len)
        
        return window.astype(np.float32)
    
    def feed_data(self, samples: np.ndarray) -> None:
        """
        Feed audio samples into STFT buffer (matching Android feedData approach).
        
        Args:
            samples: Audio samples as float32 array (normalized -1.0 to 1.0)
        """
        ds_len = len(samples)
        ds_pt = 0
        
        while ds_pt < ds_len:
            # Fill buffer
            while self.spectrum_amp_pt < self.fft_len and ds_pt < ds_len:
                s = float(samples[ds_pt])
                self.spectrum_amp_in[self.spectrum_amp_pt] = s
                self.spectrum_amp_pt += 1
                ds_pt += 1
                
                # Accumulate RMS
                self.cum_rms += s * s
                self.cnt_rms += 1
            
            # When buffer is full, perform FFT
            if self.spectrum_amp_pt == self.fft_len:
                # Apply window
                windowed = self.spectrum_amp_in * self.window * self.window_normalize
                
                # Perform FFT
                fft_result = rfft(windowed)
                
                # Convert to power spectrum (matching Android fftToAmp)
                # Android uses: (re^2 + im^2) * scaler where scaler = 2*2/(N*N)
                scaler = 2.0 * 2.0 / (self.fft_len * self.fft_len)
                spectrum_amp_tmp = np.abs(fft_result) ** 2 * scaler
                
                # Accumulate
                self.spectrum_amp_out_cum += spectrum_amp_tmp
                self.n_analysed += 1
                
                # Slide window (copy remaining data)
                if self.hop_len < self.fft_len:
                    remaining = self.fft_len - self.hop_len
                    self.spectrum_amp_in[:remaining] = self.spectrum_amp_in[self.hop_len:]
                    self.spectrum_amp_pt = remaining
                else:
                    self.spectrum_amp_pt = 0
    
    def get_spectrum_amp_db(self) -> np.ndarray:
        """
        Get spectrum amplitude in dB (matching Android getSpectrumAmpDB).
        Returns averaged spectrum if n_analysed >= n_average.
        """
        if self.n_analysed >= self.n_average:
            # Average the accumulated spectrum
            self.spectrum_amp_out = self.spectrum_amp_out_cum / self.n_analysed
            
            # Convert to dB (matching Android: 10.0 * log10 for power spectrum)
            self.spectrum_amp_out_db = 10.0 * np.log10(
                np.maximum(self.spectrum_amp_out, 1e-10)
            )
            
            # Reset accumulation
            self.spectrum_amp_out_cum.fill(0.0)                              
            self.n_analysed = 0
        
        return self.spectrum_amp_out_db
    
    def calculate_peak(self) -> None:
        """
        Calculate peak frequency using quadratic interpolation (matching Android calculatePeak).
        """
        self.get_spectrum_amp_db()
        
        # Find maximum peak (matching Android)
        min_db_threshold = 20 * np.log10(0.125 / 32768)  # ≈ -88.3 dB
        self.max_amp_db = min_db_threshold
        max_amp_idx = 0
        
        for i in range(1, len(self.spectrum_amp_out_db)):
            if self.spectrum_amp_out_db[i] > self.max_amp_db:
                self.max_amp_db = self.spectrum_amp_out_db[i]
                max_amp_idx = i
        
        # Initial frequency calculation
        self.max_amp_freq = max_amp_idx * self.sample_rate / self.fft_len
        
        # Quadratic interpolation for better accuracy (matching Android)
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
                
                if a < 0:  # Ensure it's a peak (parabola opens downward)
                    x_peak = -b / (2.0 * a)
                    if abs(x_peak) < 1.0:
                        self.max_amp_freq += x_peak * freq_resolution
                        self.max_amp_db = (4 * a * c - b * b) / (4 * a)
    
    def get_rms(self) -> float:
        """Get RMS value (matching Android getRMS)."""
        if self.cnt_rms > 8000 / 30:
            self.out_rms = np.sqrt(self.cum_rms / self.cnt_rms * 2.0)
            self.cum_rms = 0.0
            self.cnt_rms = 0
        return self.out_rms
    
    def n_elem_spectrum_amp(self) -> int:
        """Get number of accumulated FFT results."""
        return self.n_analysed
    
    def clear(self) -> None:
        """Clear internal buffers."""
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
    """
    Quadratic interpolation for peak refinement (matching Android STFT approach).
    Fits a parabola through three points: (x-1, f[x-1]), (x, f[x]), (x+1, f[x+1])
    Returns the interpolated peak position and value.
    
    Formula matches Android: xPeak = -b/(2*a) where:
    a = (f[x+1] + f[x-1])/2 - f[x]
    b = (f[x+1] - f[x-1])/2
    """
    if x == 0 or x == len(f) - 1:
        return float(x), float(f[x])
    try:
        x1 = f[x - 1]
        x2 = f[x]
        x3 = f[x + 1]
        
        # Match Android's quadratic interpolation formula
        # a*x^2 + b*x + c = y
        # where: a - b + c = x1, c = x2, a + b + c = x3
        c = x2
        a = (x3 + x1) / 2.0 - x2
        b = (x3 - x1) / 2.0
        
        if abs(a) < 1e-10:  # Avoid division by zero (flat line)
            return float(x), float(f[x])
        
        # Ensure it's a peak (parabola opens downward), not a valley (matching Android)
        if a >= 0:
            return float(x), float(f[x])
        
        # Calculate peak position offset from center bin
        xPeak = -b / (2.0 * a)
        
        # Ensure the peak is within reasonable bounds (within 1 bin, matching Android)
        if abs(xPeak) >= 1.0:
            return float(x), float(f[x])
        
        # Calculate interpolated peak value
        yv = (4 * a * c - b * b) / (4 * a)
        
        # Return absolute position (x + offset)
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
    
    # Convert to dB for threshold checking (matching Android approach)
    magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
    max_magnitude_db = 20 * np.log10(max_magnitude + 1e-10)
    
    # Apply minimum threshold (matching Android: 20*log10(0.125/32768) ≈ -88.3 dB)
    # This prevents noise from being detected as tones
    min_db_threshold = 20 * np.log10(0.125 / 32768)  # Android's minimum threshold
    if max_magnitude_db < min_db_threshold:
        return []  # Signal too weak, likely just noise
    
    # Apply magnitude threshold in linear scale
    if max_magnitude < magnitude_threshold:
        return []  # Signal below magnitude threshold
    
    signal_rms = np.sqrt(np.mean(sig**2))
    if signal_rms < 1e-6:
        return []

    candidate_peaks = []
    for i in range(1, len(magnitudes) - 1):
        if (
            magnitudes[i] > magnitudes[i - 1]
            and magnitudes[i] > magnitudes[i + 1]
        ):
            # Additional check: peak must be above minimum dB threshold
            # This filters out noise peaks before processing
            if magnitudes_db[i] > min_db_threshold:
                candidate_peaks.append(i)

    if len(candidate_peaks) == 0:
        return []
    
    peak_data = []

    for peak_idx in candidate_peaks:
        # Bounds checking (matching Android approach)
        # Ensure we're not at the edges and can interpolate
        if peak_idx < 1 or peak_idx >= len(magnitudes_db) - 1:
            # Can't interpolate at edges, use direct calculation
            peak_freq = peak_idx * fs / N
        else:
            # Use dB values for interpolation (matching Android STFT approach)
            true_i, _ = parabolic(magnitudes_db, peak_idx)
            peak_freq = true_i * fs / N
            
            # Verify interpolation result is reasonable (within 1 bin, matching Android)
            if abs(true_i - peak_idx) >= 1.0:
                # Interpolation gave unreasonable result, use direct calculation
                peak_freq = peak_idx * fs / N
        
        if peak_freq < min_freq or peak_freq > max_freq:
            continue
        
        prominence = calculate_peak_prominence(magnitudes, peak_idx)
        width = calculate_peak_width(magnitudes, peak_idx)
        peak_mag = magnitudes[peak_idx]
        
        # Apply magnitude threshold check (matching Android approach)
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
        
        # STFT configuration (matching Android approach)
        self.FFT_LEN = 8192  # Power of 2, matching Android default
        self.HOP_LEN = 4096  # 50% overlap
        self.N_AVERAGE = 1  # Number of FFT results to average
        self.WINDOW_NAME = "Hanning"
        
        # Initialize STFT (matching Android STFT class)
        self.stft = STFT(
            fft_len=self.FFT_LEN,
            hop_len=self.HOP_LEN,
            sample_rate=SAMPLE_RATE,
            window_name=self.WINDOW_NAME,
            n_average=self.N_AVERAGE
        )
        
        # Time resolution: STFT produces results approximately every hop_len/sample_rate seconds
        self.hop_time_ms = (self.HOP_LEN / SAMPLE_RATE) * 1000.0
        
        # Global array: Store peak frequencies at each STFT update
        # Estimate size based on MAX_BUFFER_SECONDS
        self.MAX_FREQUENCY_PEAKS_SIZE = max(
            1, int((MAX_BUFFER_SECONDS * 1000) / self.hop_time_ms)
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

        # FFT resolution (matching Android approach)
        self.fft_resolution_hz = SAMPLE_RATE / self.FFT_LEN
        self.default_range_hz = self.fft_resolution_hz * 1.5

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
        """Thread 1: Feed audio samples to STFT and store peak frequency when FFT result is ready (matching Android approach)."""
        while self.threads_running.is_set():
            try:
                try:
                    audio_samples = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Feed samples to STFT (matching Android stft.feedData approach)
                self.stft.feed_data(audio_samples)
                
                # Check if we have enough accumulated FFT results (matching Android nElemSpectrumAmp check)
                if self.stft.n_elem_spectrum_amp() >= self.N_AVERAGE:
                    # Calculate peak frequency (matching Android calculatePeak)
                    self.stft.calculate_peak()
                    
                    # Store the peak frequency (only if valid)
                    max_peak = None
                    if not np.isnan(self.stft.max_amp_freq) and self.stft.max_amp_freq > 0:
                        # Apply minimum threshold check (matching Android)
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
        """Thread 2: Check peak frequencies and detect tones (matching Android approach)."""
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
                    
                    # Calculate number of elements (hop_time_ms per element, matching STFT update rate)
                    elements_a = int(round(tone_a_ms / self.hop_time_ms))
                    elements_b = int(round(tone_b_ms / self.hop_time_ms))
                    total_elements = elements_a + elements_b
                    
                    # Get recent peaks
                    recent_peaks = self._get_recent_peaks(total_elements)
                    if len(recent_peaks) < total_elements:
                        continue
                    
                    # Extract tone A and tone B elements (most recent is last)
                    tone_a_elements = recent_peaks[-total_elements:-elements_b] if elements_b > 0 else recent_peaks[-elements_a:]
                    tone_b_elements = recent_peaks[-elements_b:]
                    
                    # Match Android approach: Use peak frequencies directly (no averaging)
                    # Check if peaks are within target range (matching Android frequency matching)
                    tone_a_matches = 0
                    tone_b_matches = 0
                    tone_a_detected_freq = None
                    tone_b_detected_freq = None
                    
                    # Check tone A: count how many peaks are within range, use most recent valid peak
                    for freq in reversed(tone_a_elements):
                        if freq is not None:
                            if abs(freq - tone_a_target) <= tone_a_range:
                                tone_a_matches += 1
                                if tone_a_detected_freq is None:
                                    # Use the most recent peak frequency (matching Android - direct peak value)
                                    tone_a_detected_freq = freq
                    
                    # Check tone B: count how many peaks are within range, use most recent valid peak
                    for freq in reversed(tone_b_elements):
                        if freq is not None:
                            if abs(freq - tone_b_target) <= tone_b_range:
                                tone_b_matches += 1
                                if tone_b_detected_freq is None:
                                    # Use the most recent peak frequency (matching Android - direct peak value)
                                    tone_b_detected_freq = freq
                    
                    # Require at least 70% of peaks to be in range (matching previous threshold)
                    tone_a_match_ratio = tone_a_matches / len([f for f in tone_a_elements if f is not None]) if len([f for f in tone_a_elements if f is not None]) > 0 else 0
                    tone_b_match_ratio = tone_b_matches / len([f for f in tone_b_elements if f is not None]) if len([f for f in tone_b_elements if f is not None]) > 0 else 0
                    
                    if tone_a_match_ratio >= self.SIMILARITY_THRESHOLD and tone_b_match_ratio >= self.SIMILARITY_THRESHOLD:
                        if tone_a_detected_freq is None or tone_b_detected_freq is None:
                            continue
                        
                        # Prevent duplicate detections
                        min_interval = (tone_a_ms + tone_b_ms) / 1000.0
                        if time.time() - self.last_tone_detection_time < min_interval:
                            continue
                        
                        # Use detected peak frequencies directly (matching Android - no averaging)
                        tone_a_freq = tone_a_detected_freq
                        tone_b_freq = tone_b_detected_freq
                        
                        # Apply frequency filters
                        if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                            continue
                        
                        # Ensure tones are different
                        if abs(tone_a_freq - tone_b_freq) <= 50:
                            continue

                        # Round to 1 decimal place (matching Android precision)
                        tone_a_freq = round(tone_a_freq, 1)
                        tone_b_freq = round(tone_b_freq, 1)
                        
                        self.last_tone_detection_time = time.time()
                        self._defined_tone_alert(tone_def, tone_a_freq, tone_b_freq)
                        self._defined_tone_passthrough(tone_def)
                        self._defined_tone_recording(tone_def)
                
                # Check for new tones (using direct peak approach)
                if self.detect_new_tones:
                    elements_new = int(round(self.new_tone_length_ms / self.hop_time_ms))
                    total_elements_new = elements_new * 2
                    
                    recent_peaks = self._get_recent_peaks(total_elements_new)
                    if len(recent_peaks) >= total_elements_new:
                        # Extract two consecutive windows
                        tone_a_elements = recent_peaks[-total_elements_new:-elements_new]
                        tone_b_elements = recent_peaks[-elements_new:]
                        
                        # Match Android: Use most recent peak frequency directly
                        # Find most recent valid peak in each window
                        tone_a_freq = None
                        tone_b_freq = None
                        
                        for freq in reversed(tone_a_elements):
                            if freq is not None:
                                tone_a_freq = freq
                                break
                        
                        for freq in reversed(tone_b_elements):
                            if freq is not None:
                                tone_b_freq = freq
                                break
                        
                        if tone_a_freq and tone_b_freq:
                            # Ensure tones are different
                            if abs(tone_a_freq - tone_b_freq) <= 50:
                                continue
                            
                            # Check stability: count peaks within range of detected frequency
                            tone_a_stable = sum(1 for f in tone_a_elements if f is not None and abs(f - tone_a_freq) <= self.new_tone_range_hz)
                            tone_b_stable = sum(1 for f in tone_b_elements if f is not None and abs(f - tone_b_freq) <= self.new_tone_range_hz)
                            
                            tone_a_stability = tone_a_stable / len([f for f in tone_a_elements if f is not None]) if len([f for f in tone_a_elements if f is not None]) > 0 else 0
                            tone_b_stability = tone_b_stable / len([f for f in tone_b_elements if f is not None]) if len([f for f in tone_b_elements if f is not None]) > 0 else 0
                            
                            if tone_a_stability >= self.SIMILARITY_THRESHOLD and tone_b_stability >= self.SIMILARITY_THRESHOLD:
                                # Prevent duplicate detections
                                min_interval = self.new_tone_length_seconds * 2 + 0.2
                                if time.time() - self.last_tone_detection_time < min_interval:
                                    continue
                                
                                # Apply frequency filters
                                if self._is_frequency_filtered(tone_a_freq) or self._is_frequency_filtered(tone_b_freq):
                                    continue
                                
                                self.last_tone_detection_time = time.time()
                                
                                # Round to 1 decimal place (matching Android precision)
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
