import threading
import numpy as np
from typing import List, Optional

SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 1920
FFT_SIZE = 4096
FREQ_BINS = FFT_SIZE // 2

MAX_CHANNELS = 4
CHANNEL_ID_LEN = 64

MAX_TONE_DEFINITIONS = 100
MAX_FILTERS = 50

JITTER_BUFFER_SIZE = 32

global_interrupted = threading.Event()

global_channel_ids: List[str] = [""] * MAX_CHANNELS
global_channel_count = 0

class JitterBuffer:
    def __init__(self, buffer_size: int = JITTER_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.frames: List[Optional[np.ndarray]] = [None] * buffer_size
        self.sample_counts: List[int] = [0] * buffer_size
        self.write_index = 0
        self.read_index = 0
        self.frame_count = 0
        self.mutex = threading.Lock()
    
    def write(self, samples: np.ndarray, sample_count: int) -> bool:
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
        with self.mutex:
            self.frames = [None] * self.buffer_size
            self.sample_counts = [0] * self.buffer_size
            self.write_index = 0
            self.read_index = 0
            self.frame_count = 0
    
    def is_empty(self) -> bool:
        with self.mutex:
            return self.frame_count == 0
    
    def is_full(self) -> bool:
        with self.mutex:
            return self.frame_count >= self.buffer_size
    
    def get_frame_count(self) -> int:
        with self.mutex:
            return self.frame_count
