"""
Frequency filtering module - FFT-based filtering following C code implementation.
Removes unwanted frequencies from audio before encoding.
"""
import numpy as np
from typing import List, Dict, Any

SAMPLE_RATE = 48000
FFT_SIZE = 1024
FREQ_BINS = FFT_SIZE // 2


def frequency_to_bin(frequency: float) -> float:
    """Convert frequency (Hz) to FFT bin index."""
    return (frequency * FFT_SIZE) / SAMPLE_RATE


def apply_audio_frequency_filters(
    audio_samples: np.ndarray,
    filters: List[Dict[str, Any]],
    sample_rate: int = SAMPLE_RATE,
    db_threshold: float = -20.0
) -> np.ndarray:
    """
    Apply frequency filters to actual audio samples using FFT-based filtering.
    This function applies dB attenuation to audio in the specified frequency ranges,
    and removes all audio below db_threshold (noise gate).
    Uses FFT-based filtering for precise frequency domain manipulation.
    
    Process:
    1. Convert time-domain audio to frequency domain (Forward FFT)
    2. Apply filters in frequency domain (attenuate unwanted bins by db_threshold)
    3. Remove all frequency bins below db_threshold relative to maximum
    4. Convert back to time domain (Inverse FFT)
    5. Normalize the result
    
    Args:
        audio_samples: Input audio samples (float32, mono)
        filters: List of filter dictionaries with keys: frequency, filter_range, type
        sample_rate: Sample rate in Hz (default 48000)
        db_threshold: dB threshold for attenuation and noise gate (default -20.0)
    
    Returns:
        Filtered audio samples (same shape as input)
    """
    if len(filters) == 0:
        return audio_samples
    
    # Debug: Log when filters are applied (once per function call, not per chunk)
    if not hasattr(apply_audio_frequency_filters, '_call_count'):
        apply_audio_frequency_filters._call_count = 0
        apply_audio_frequency_filters._filter_logged = set()
    
    apply_audio_frequency_filters._call_count += 1
    if apply_audio_frequency_filters._call_count <= 3:  # Log first 3 calls
        print(f"[FILTER] apply_audio_frequency_filters called with {len(filters)} filter(s)")
        for i, f in enumerate(filters):
            print(f"[FILTER]   Filter {i+1}: type='{f.get('type')}', frequency={f.get('frequency')} Hz, range={f.get('filter_range')} Hz")
    
    if len(audio_samples) == 0:
        return audio_samples
    
    if len(audio_samples) < FFT_SIZE:
        return audio_samples
    
    if not isinstance(audio_samples, np.ndarray):
        audio_samples = np.array(audio_samples, dtype=np.float32)
    
    audio_samples = audio_samples.astype(np.float32)
    output_samples = np.zeros_like(audio_samples)
    
    processed_samples = 0
    window = np.hanning(FFT_SIZE)
    
    # dB attenuation factor: 10^(db_threshold/20)
    attenuation_factor = 10.0 ** (db_threshold / 20.0)
    
    while processed_samples + FFT_SIZE <= len(audio_samples):
        chunk = audio_samples[processed_samples:processed_samples + FFT_SIZE]
        fft_input = chunk.astype(np.float64) * window
        
        fft_output = np.fft.rfft(fft_input)
        
        for filter_data in filters:
            frequency = filter_data["frequency"]
            filter_range = filter_data["filter_range"]
            filter_type = filter_data["type"]
            
            target_bin = frequency_to_bin(frequency)
            range_bins = int(round((filter_range * FFT_SIZE) / sample_rate))
            target_bin_int = int(target_bin)
            
            # Calculate actual frequency range for this bin
            bin_freq_low = (target_bin_int * sample_rate) / FFT_SIZE
            bin_freq_high = ((target_bin_int + 1) * sample_rate) / FFT_SIZE
            
            filter_key = f"{filter_type}_{frequency}"
            is_first_log = filter_key not in apply_audio_frequency_filters._filter_logged
            if is_first_log:
                apply_audio_frequency_filters._filter_logged.add(filter_key)
                print(f"[FILTER] Applying filter: type='{filter_type}', frequency={frequency} Hz, "
                      f"range={filter_range} Hz -> bin={target_bin_int} "
                      f"(freq range: {bin_freq_low:.1f}-{bin_freq_high:.1f} Hz)")
            
            if filter_type == "below":
                if target_bin_int > 0:
                    bins_zeroed = min(target_bin_int, len(fft_output))
                    fft_output[:bins_zeroed] = 0.0
                    if is_first_log:
                        print(f"[FILTER] Zeroed bins 0-{bins_zeroed-1} (frequencies below {frequency} Hz)")
            elif filter_type == "above":
                if target_bin_int < len(fft_output):
                    bins_zeroed = len(fft_output) - target_bin_int
                    max_freq = (len(fft_output) * sample_rate) / FFT_SIZE
                    fft_output[target_bin_int:] = 0.0
                    if is_first_log:
                        print(f"[FILTER] Zeroed bins {target_bin_int}-{len(fft_output)-1} "
                              f"(frequencies {frequency:.1f} Hz to {max_freq:.1f} Hz, {bins_zeroed} bins)")
            # elif filter_type == "center":
            #     mask = np.abs(np.arange(len(fft_output)) - target_bin_int) <= range_bins
            #     fft_output[mask] *= attenuation_factor
        
        # Remove all audio below db_threshold (noise gate)
        magnitudes = np.abs(fft_output)
        max_magnitude = np.max(magnitudes)
        if max_magnitude > 0:
            # db_threshold: 10^(db_threshold/20)
            threshold_factor = 10.0 ** (db_threshold / 20.0)
            magnitude_threshold = max_magnitude * threshold_factor
            
            # Set bins below threshold to zero
            mask_below_threshold = magnitudes < magnitude_threshold
            fft_output[mask_below_threshold] = 0.0
        
        filtered_samples = np.fft.irfft(fft_output, n=FFT_SIZE)
        output_samples[processed_samples:processed_samples + FFT_SIZE] = (
            filtered_samples.astype(np.float32) / FFT_SIZE
        )
        
        processed_samples += FFT_SIZE
    
    if processed_samples < len(audio_samples):
        output_samples[processed_samples:] = audio_samples[processed_samples:]
    
    return output_samples

