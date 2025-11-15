"""
EchoStream - Common definitions and constants

This module provides system-wide constants and shared data structures
for the EchoStream audio streaming system.
"""
import threading
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
    
    Thread-safe buffer that stores audio frames with sequence numbers
    for proper playback order and timing.
    """
    def __init__(self, buffer_size: int = JITTER_BUFFER_SIZE):
        """Initialize jitter buffer with specified size."""
        self.buffer_size = buffer_size
        self.frames: List[Optional[bytes]] = [None] * buffer_size
        self.write_index = 0
        self.read_index = 0
        self.frame_count = 0
        self.mutex = threading.Lock()
    
    def write(self, frame: bytes) -> bool:
        """
        Write a frame to the buffer.
        
        Args:
            frame: Audio frame data (Opus encoded)
            
        Returns:
            True if frame was written, False if buffer is full
        """
        with self.mutex:
            if self.frame_count >= self.buffer_size:
                # Buffer full, drop oldest frame
                self.read_index = (self.read_index + 1) % self.buffer_size
                self.frame_count -= 1
            
            self.frames[self.write_index] = frame
            self.write_index = (self.write_index + 1) % self.buffer_size
            self.frame_count += 1
            return True
    
    def read(self) -> Optional[bytes]:
        """
        Read a frame from the buffer.
        
        Returns:
            Frame data if available, None if buffer is empty
        """
        with self.mutex:
            if self.frame_count == 0:
                return None
            
            frame = self.frames[self.read_index]
            self.frames[self.read_index] = None
            self.read_index = (self.read_index + 1) % self.buffer_size
            self.frame_count -= 1
            return frame
    
    def clear(self):
        """Clear all frames from the buffer."""
        with self.mutex:
            self.frames = [None] * self.buffer_size
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
