"""
Tone detection module - detects specific tone sequences in audio streams.
Uses FFT-based frequency analysis to identify tone pairs (tone A followed by tone B).
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import time

try:
    from communication.mqtt import publish_known_tone_detection, publish_new_tone_pair
    from processing.recording import global_recording_manager
    from processing.passthrough import global_passthrough_manager
    HAS_MQTT = True
    HAS_RECORDING = True
    HAS_PASSTHROUGH = True
except ImportError:
    HAS_MQTT = False
    HAS_RECORDING = False
    HAS_PASSTHROUGH = False
    publish_known_tone_detection = None
    publish_new_tone_pair = None
    global_recording_manager = None
    global_passthrough_manager = None

SAMPLE_RATE = 48000
FFT_SIZE = 1024
FREQ_BINS = FFT_SIZE // 2

# Global detector registry
_channel_detectors: Dict[str, 'ChannelToneDetector'] = {}


def frequency_to_bin(frequency: float) -> float:
    """Convert frequency (Hz) to FFT bin index."""
    return (frequency * FFT_SIZE) / SAMPLE_RATE


def bin_to_frequency(bin_index: float) -> float:
    """Convert FFT bin index to frequency (Hz)."""
    return (bin_index * SAMPLE_RATE) / FFT_SIZE


class ChannelToneDetector:
    """Detects tone sequences for a single audio channel."""
    
    def __init__(self, channel_id: str, tone_definitions: List[Dict[str, Any]], 
                 new_tone_config: Optional[Dict[str, Any]] = None,
                 passthrough_config: Optional[Dict[str, Any]] = None):
        self.channel_id = channel_id
        self.tone_definitions = tone_definitions
        self.volume_threshold_db = -20.0  # Fixed -20 dB threshold
        
        # New tone detection config
        self.new_tone_config = new_tone_config or {}
        self.detect_new_tones = self.new_tone_config.get("detect_new_tones", False)
        self.new_tone_length_ms = self.new_tone_config.get("new_tone_length_ms", 1000)
        self.new_tone_range_hz = self.new_tone_config.get("new_tone_range_hz", 3)
        
        # Passthrough config
        self.passthrough_config = passthrough_config or {}
        self.tone_passthrough = self.passthrough_config.get("tone_passthrough", False)
        self.passthrough_channel = self.passthrough_config.get("passthrough_channel", "")
        
        # Tone sequence tracking
        self.tone_sequence: deque = deque(maxlen=100)  # Store recent detected tones
        self.last_volume_log_time = 0
        self.volume_log_interval = 5.0  # Log volume every 5 seconds
        
        print(f"[TONE] Initialized detector for channel {channel_id}: "
              f"{len(tone_definitions)} defined tone(s), "
              f"new_tone_detection={'ON' if self.detect_new_tones else 'OFF'}, "
              f"volume_threshold={self.volume_threshold_db} dB")
    
    def _calculate_volume_db(self, audio_samples: np.ndarray) -> float:
        """Calculate RMS volume in dB."""
        if len(audio_samples) == 0:
            return -np.inf
        
        rms = np.sqrt(np.mean(audio_samples ** 2))
        if rms == 0:
            return -np.inf
        
        # Convert to dB: 20 * log10(rms)
        # Assuming full scale is 1.0
        db = 20.0 * np.log10(rms)
        return db
    
    def _find_peak_frequencies(self, audio_samples: np.ndarray, 
                               relative_threshold: float = 0.3) -> List[Tuple[float, float]]:
        """
        Find peak frequencies in audio using FFT.
        
        Args:
            audio_samples: Audio samples to analyze
            relative_threshold: Relative magnitude threshold (0.0-1.0)
        
        Returns:
            List of (frequency_hz, magnitude) tuples for detected peaks
        """
        if len(audio_samples) < FFT_SIZE:
            return []
        
        # Use a windowed FFT
        window = np.hanning(FFT_SIZE)
        fft_input = audio_samples[:FFT_SIZE].astype(np.float64) * window
        fft_output = np.fft.rfft(fft_input)
        
        # Get magnitudes
        magnitudes = np.abs(fft_output)
        max_magnitude = np.max(magnitudes)
        
        if max_magnitude == 0:
            return []
        
        # Find peaks above threshold
        threshold = max_magnitude * relative_threshold
        peaks = []
        
        # Find local maxima
        for i in range(1, len(magnitudes) - 1):
            if magnitudes[i] > threshold and magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                freq = bin_to_frequency(i)
                peaks.append((freq, magnitudes[i]))
        
        return peaks
    
    def _check_tone_match(self, detected_freq: float, target_freq: float, 
                         tolerance_hz: int) -> bool:
        """Check if detected frequency matches target within tolerance."""
        return abs(detected_freq - target_freq) <= tolerance_hz
    
    def _detect_defined_tones(self, peaks: List[Tuple[float, float]], 
                              current_time_ms: int) -> Optional[Dict[str, Any]]:
        """Check if detected peaks match any defined tone sequences."""
        if not self.tone_definitions:
            return None
        
        for tone_def in self.tone_definitions:
            tone_id = tone_def.get("tone_id", "")
            tone_a = tone_def.get("tone_a", 0.0)
            tone_b = tone_def.get("tone_b", 0.0)
            tone_a_range = tone_def.get("tone_a_range", 0)
            tone_b_range = tone_def.get("tone_b_range", 0)
            tone_a_length_ms = tone_def.get("tone_a_length_ms", 0)
            tone_b_length_ms = tone_def.get("tone_b_length_ms", 0)
            
            if tone_a <= 0 or tone_b <= 0:
                continue
            
            # Check if we have tone A in recent sequence
            tone_a_detected = False
            tone_b_detected = False
            tone_a_time = None
            tone_b_time = None
            
            # Look through recent tone sequence
            for entry in self.tone_sequence:
                entry_time, entry_freq = entry
                
                # Check for tone A
                if not tone_a_detected and self._check_tone_match(entry_freq, tone_a, tone_a_range):
                    tone_a_detected = True
                    tone_a_time = entry_time
                
                # Check for tone B after tone A
                if tone_a_detected and not tone_b_detected:
                    if self._check_tone_match(entry_freq, tone_b, tone_b_range):
                        tone_b_detected = True
                        tone_b_time = entry_time
                        break
            
            # Also check current peaks
            for freq, _ in peaks:
                if not tone_a_detected and self._check_tone_match(freq, tone_a, tone_a_range):
                    tone_a_detected = True
                    tone_a_time = current_time_ms
                
                if tone_a_detected and not tone_b_detected:
                    if self._check_tone_match(freq, tone_b, tone_b_range):
                        tone_b_detected = True
                        tone_b_time = current_time_ms
                        break
            
            # If we have both tones in sequence, validate timing
            if tone_a_detected and tone_b_detected and tone_a_time and tone_b_time:
                time_diff = tone_b_time - tone_a_time
                # Tone B should come after tone A within reasonable time
                if 0 < time_diff < 5000:  # Max 5 seconds between tones
                    # Found a match!
                    record_length_ms = tone_def.get("record_length_ms", 0)
                    detection_tone_alert = tone_def.get("detection_tone_alert", "")
                    
                    result = {
                        "tone_id": tone_id,
                        "tone_a": tone_a,
                        "tone_b": tone_b,
                        "tone_a_duration_ms": tone_a_length_ms,
                        "tone_b_duration_ms": tone_b_length_ms,
                        "tone_a_range_hz": tone_a_range,
                        "tone_b_range_hz": tone_b_range,
                        "record_length_ms": record_length_ms,
                        "detection_tone_alert": detection_tone_alert
                    }
                    
                    # Publish via MQTT
                    if HAS_MQTT and publish_known_tone_detection:
                        publish_known_tone_detection(
                            tone_id=tone_id,
                            tone_a_hz=tone_a,
                            tone_b_hz=tone_b,
                            tone_a_duration_ms=tone_a_length_ms,
                            tone_b_duration_ms=tone_b_length_ms,
                            tone_a_range_hz=tone_a_range,
                            tone_b_range_hz=tone_b_range,
                            channel_id=self.channel_id,
                            record_length_ms=record_length_ms,
                            detection_tone_alert=detection_tone_alert
                        )
                    
                    # Trigger recording if enabled
                    if HAS_RECORDING and global_recording_manager and record_length_ms > 0:
                        try:
                            global_recording_manager.start_recording(
                                self.channel_id, record_length_ms
                            )
                        except Exception as e:
                            print(f"[TONE] WARNING: Failed to start recording: {e}")
                    
                    # Trigger passthrough if enabled
                    if HAS_PASSTHROUGH and global_passthrough_manager and self.tone_passthrough:
                        if self.passthrough_channel:
                            try:
                                global_passthrough_manager.start_passthrough(
                                    self.channel_id, self.passthrough_channel, record_length_ms
                                )
                            except Exception as e:
                                print(f"[TONE] WARNING: Failed to start passthrough: {e}")
                    
                    return result
        
        return None
    
    def _detect_new_tones(self, peaks: List[Tuple[float, float]], 
                         current_time_ms: int) -> Optional[Dict[str, Any]]:
        """Detect new (unknown) tone pairs if enabled."""
        if not self.detect_new_tones or len(peaks) < 2:
            return None
        
        # Look for two distinct peaks in current detection
        if len(peaks) >= 2:
            # Sort by magnitude (strongest first)
            sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
            freq_a, mag_a = sorted_peaks[0]
            freq_b, mag_b = sorted_peaks[1]
            
            # Check if these frequencies are consistently detected
            # (simplified: just check if we have two strong peaks now)
            if mag_a > 0 and mag_b > 0:
                # Check if this pair is in recent history
                recent_pairs = []
                for entry in self.tone_sequence:
                    recent_pairs.append(entry[1])  # frequency
                
                # If we have enough history, check for pattern
                if len(recent_pairs) >= 2:
                    # Simple check: do we see these two frequencies repeatedly?
                    count_a = sum(1 for f in recent_pairs if self._check_tone_match(f, freq_a, self.new_tone_range_hz))
                    count_b = sum(1 for f in recent_pairs if self._check_tone_match(f, freq_b, self.new_tone_range_hz))
                    
                    if count_a >= 2 and count_b >= 2:
                        # Publish new tone pair
                        if HAS_MQTT and publish_new_tone_pair:
                            publish_new_tone_pair(freq_a, freq_b)
                        
                        # Trigger recording
                        if HAS_RECORDING and global_recording_manager:
                            try:
                                global_recording_manager.start_new_tone_recording(
                                    self.channel_id, freq_a, freq_b, self.new_tone_length_ms
                                )
                            except Exception as e:
                                print(f"[TONE] WARNING: Failed to start new tone recording: {e}")
                        
                        return {
                            "tone_id": "new_tone",
                            "tone_a": freq_a,
                            "tone_b": freq_b
                        }
        
        return None
    
    def process_audio(self, audio_samples: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process audio samples and detect tone sequences.
        
        Args:
            audio_samples: Audio samples to analyze (numpy array, float32)
        
        Returns:
            Dict with tone detection result, or None if no tone detected
        """
        if len(audio_samples) == 0:
            return None
        
        # Check volume threshold
        volume_db = self._calculate_volume_db(audio_samples)
        
        # Periodic volume logging for debugging
        current_time = time.time()
        if current_time - self.last_volume_log_time >= self.volume_log_interval:
            print(f"[TONE] Channel {self.channel_id}: volume={volume_db:.1f} dB "
                  f"(threshold={self.volume_threshold_db} dB)")
            self.last_volume_log_time = current_time
        
        # Skip if volume is too low
        if volume_db < self.volume_threshold_db:
            return None
        
        # Find peak frequencies
        peaks = self._find_peak_frequencies(audio_samples, relative_threshold=0.3)
        
        if not peaks:
            return None
        
        current_time_ms = int(time.time() * 1000)
        
        # Add current peaks to sequence history
        for freq, mag in peaks:
            self.tone_sequence.append((current_time_ms, freq))
        
        # Check for defined tones first
        result = self._detect_defined_tones(peaks, current_time_ms)
        if result:
            return result
        
        # Check for new tones if enabled
        if self.detect_new_tones:
            result = self._detect_new_tones(peaks, current_time_ms)
            if result:
                return result
        
        return None


def init_channel_detector(channel_id: str, tone_definitions: List[Dict[str, Any]],
                         new_tone_config: Optional[Dict[str, Any]] = None,
                         passthrough_config: Optional[Dict[str, Any]] = None):
    """Initialize a tone detector for a specific channel."""
    detector = ChannelToneDetector(channel_id, tone_definitions, new_tone_config, passthrough_config)
    _channel_detectors[channel_id] = detector
    return detector


def process_audio_for_channel(channel_id: str, audio_samples: np.ndarray) -> Optional[Dict[str, Any]]:
    """Process audio for a specific channel and return detected tone if any."""
    if channel_id not in _channel_detectors:
        return None
    
    detector = _channel_detectors[channel_id]
    return detector.process_audio(audio_samples)

