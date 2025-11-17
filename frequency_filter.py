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
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Apply frequency filters to actual audio samples using FFT-based filtering.
    This function completely removes audio in the specified frequency ranges.
    Uses FFT-based filtering for precise frequency domain manipulation.
    
    Process:
    1. Convert time-domain audio to frequency domain (Forward FFT)
    2. Apply filters in frequency domain (zero out unwanted bins)
    3. Convert back to time domain (Inverse FFT)
    4. Normalize the result
    
    Args:
        audio_samples: Input audio samples (float32, mono)
        filters: List of filter dictionaries with keys: frequency, filter_range, type
        sample_rate: Sample rate in Hz (default 48000)
    
    Returns:
        Filtered audio samples (same shape as input)
    """
    if len(filters) == 0:
        return audio_samples
    
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
            
            if filter_type == "below":
                if target_bin_int > 0:
                    fft_output[:min(target_bin_int, len(fft_output))] = 0.0
            elif filter_type == "above":
                if target_bin_int < len(fft_output):
                    fft_output[target_bin_int:] = 0.0
            elif filter_type == "center":
                mask = np.abs(np.arange(len(fft_output)) - target_bin_int) <= range_bins
                fft_output[mask] = 0.0
        
        filtered_samples = np.fft.irfft(fft_output, n=FFT_SIZE)
        output_samples[processed_samples:processed_samples + FFT_SIZE] = (
            filtered_samples.astype(np.float32) / FFT_SIZE
        )
        
        processed_samples += FFT_SIZE
    
    if processed_samples < len(audio_samples):
        output_samples[processed_samples:] = audio_samples[processed_samples:]
    
    return output_samples

