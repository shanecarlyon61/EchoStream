"""
EchoStream - Common definitions and constants
"""
import threading

# Constants
JITTER_BUFFER_SIZE = 8
SAMPLES_PER_FRAME = 1920
MAX_CHANNELS = 4
CHANNEL_ID_LEN = 64
MAX_TONE_DEFINITIONS = 100
MAX_FILTERS = 50
FFT_SIZE = 4096
SAMPLE_RATE = 48000
FREQ_BINS = FFT_SIZE // 2

# Global state
global_interrupted = threading.Event()
global_channel_ids = [""] * MAX_CHANNELS
global_channel_count = 0
