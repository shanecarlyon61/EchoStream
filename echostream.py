"""
EchoStream - Common definitions and constants

This module provides system-wide constants and shared data structures
for the EchoStream audio streaming system.
"""
import threading
import numpy as np
from typing import List, Optional


# ============================================================================
# System Constants
# ============================================================================

# Audio configuration
SAMPLE_RATE = 48000  # Hz
SAMPLES_PER_FRAME = 1920  # Samples per Opus frame (40ms at 48kHz)
FFT_SIZE = 4096  # FFT size for tone detection
FREQ_BINS = FFT_SIZE // 2  # Number of frequency bins

# Channel configuration
MAX_CHANNELS = 4  # Maximum number of audio channels
CHANNEL_ID_LEN = 64  # Maximum channel ID length

# Tone detection configuration
MAX_TONE_DEFINITIONS = 100  # Maximum number of tone definitions
MAX_FILTERS = 50  # Maximum number of frequency filters

# Buffer configuration
JITTER_BUFFER_SIZE = 8  # Number of frames in jitter buffer


# ============================================================================
# Global State Management
# ============================================================================

# Global shutdown event for coordinating thread shutdown
global_interrupted = threading.Event()

# Global channel tracking
global_channel_ids: List[str] = [""] * MAX_CHANNELS
global_channel_count = 0


# ============================================================================
# Shared Data Structures
# ============================================================================

class JitterBuffer:
    """
    Jitter buffer for audio playback to handle network delay variations.
    
    Thread-safe buffer that stores decoded float32 audio frames.
    Matches C implementation: stores float samples (not Opus bytes).
    """
    def __init__(self, buffer_size: int = JITTER_BUFFER_SIZE):
        """Initialize jitter buffer with specified size."""
        self.buffer_size = buffer_size
        self.frames: List[Optional[np.ndarray]] = [None] * buffer_size
        self.sample_counts: List[int] = [0] * buffer_size
        self.write_index = 0
        self.read_index = 0
        self.frame_count = 0
        self.mutex = threading.Lock()
    
    def write(self, samples: np.ndarray, sample_count: int) -> bool:
        """
        Write decoded float samples to the buffer.
        
        Args:
            samples: Audio samples as numpy array (float32)
            sample_count: Number of valid samples in the array
            
        Returns:
            True if frame was written, False if buffer is full
        """
        with self.mutex:
            if self.frame_count >= self.buffer_size:
                self.read_index = (self.read_index + 1) % self.buffer_size
                self.frame_count -= 1
            
            self.frames[self.write_index] = samples.copy()
            self.sample_counts[self.write_index] = sample_count
            self.write_index = (self.write_index + 1) % self.buffer_size
            self.frame_count += 1
            return True
    
    def read(self) -> Optional[tuple[np.ndarray, int]]:
        """
        Read a frame from the buffer.
        
        Returns:
            Tuple of (samples array, sample_count) if available, None if buffer is empty
        """
        with self.mutex:
            if self.frame_count == 0:
                return None
            
            samples = self.frames[self.read_index]
            sample_count = self.sample_counts[self.read_index]
            self.frames[self.read_index] = None
            self.sample_counts[self.read_index] = 0
            self.read_index = (self.read_index + 1) % self.buffer_size
            self.frame_count -= 1
            return samples, sample_count
    
    def clear(self):
        """Clear all frames from the buffer."""
        with self.mutex:
            self.frames = [None] * self.buffer_size
            self.sample_counts = [0] * self.buffer_size
            self.write_index = 0
            self.read_index = 0
            self.frame_count = 0
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self.mutex:
            return self.frame_count == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self.mutex:
            return self.frame_count >= self.buffer_size
    
    def get_frame_count(self) -> int:
        """Get current number of frames in buffer."""
        with self.mutex:
            return self.frame_count
