"""
Frequency filtering module - FFT-based filtering following C code implementation.
Removes unwanted frequencies from audio before encoding.
"""
import numpy as np
from typing import List, Dict, Any
import math

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
    while processed_samples + FFT_SIZE <= len(audio_samples):
        fft_input = np.zeros(FFT_SIZE, dtype=np.float64)
        
        for i in range(FFT_SIZE):
            fft_input[i] = float(audio_samples[processed_samples + i])
        
        for i in range(FFT_SIZE):
            window = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (FFT_SIZE - 1)))
            fft_input[i] *= window
        
        fft_output = np.fft.rfft(fft_input)
        
        for filter_data in filters:
            frequency = filter_data["frequency"]
            filter_range = filter_data["filter_range"]
            filter_type = filter_data["type"]
            
            target_bin = frequency_to_bin(frequency)
            range_bins = int(round((filter_range * FFT_SIZE) / sample_rate))
            target_bin_int = int(target_bin)
            
            if filter_type == "below":
                for i in range(min(target_bin_int, len(fft_output))):
                    fft_output[i] = 0.0 + 0.0j
            elif filter_type == "above":
                for i in range(target_bin_int, len(fft_output)):
                    fft_output[i] = 0.0 + 0.0j
            elif filter_type == "center":
                for i in range(len(fft_output)):
                    if abs(i - target_bin_int) > range_bins:
                        fft_output[i] = 0.0 + 0.0j
        
        filtered_samples = np.fft.irfft(fft_output, n=FFT_SIZE)
        
        for i in range(FFT_SIZE):
            output_samples[processed_samples + i] = np.float32(filtered_samples[i] / FFT_SIZE)
        
        processed_samples += FFT_SIZE
    
    if processed_samples < len(audio_samples):
        output_samples[processed_samples:] = audio_samples[processed_samples:]
    
    return output_samples

