"""
Audio Stream Manager - Manage individual audio channel streams

This module handles AudioStream class for managing audio input/output streams,
Opus encoding/decoding, jitter buffering, and encryption keys per channel.
"""
import pyaudio
import opuslib
import numpy as np
from typing import Optional
from echostream import JitterBuffer, SAMPLE_RATE, SAMPLES_PER_FRAME


# ============================================================================
# AudioStream Class
# ============================================================================

class AudioStream:
    """
    Audio stream for a single channel.
    
    Manages input/output streams, Opus encoder/decoder, jitter buffer,
    and encryption key for one audio channel.
    """
    def __init__(self):
        """Initialize an AudioStream with default values."""
        # Channel identification
        self.channel_id: str = ""
        self.device_index: int = -1
        
        # PyAudio streams
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Opus codec
        self.encoder: Optional[opuslib.Encoder] = None
        self.decoder: Optional[opuslib.Decoder] = None
        
        # Jitter buffer for output playback
        self.output_jitter: JitterBuffer = JitterBuffer()
        
        # Encryption key (32 bytes for AES-256)
        self.encryption_key: bytes = bytes(32)
        
        # State flags
        self.gpio_active: bool = False  # GPIO pin is active (PTT pressed)
        self.transmitting: bool = False  # Audio transmission is active
        
        # Audio buffers
        self.input_buffer: np.ndarray = np.zeros(4800, dtype=np.float32)
        self.input_buffer_pos: int = 0
        self.current_output_frame_pos: int = 0
        
        # Buffer size
        self.buffer_size: int = 4800  # Samples (100ms at 48kHz)


# ============================================================================
# Stream Creation and Management
# ============================================================================

def create_stream(channel_id: str, device_index: int, pa_instance: pyaudio.PyAudio) -> Optional[AudioStream]:
    """
    Create a new audio stream for a channel.
    
    Args:
        channel_id: Channel ID string
        device_index: Audio device index
        pa_instance: PyAudio instance
        
    Returns:
        AudioStream instance or None on error
    """
    if pa_instance is None:
        print(f"[AUDIO_STREAM] ERROR: PyAudio instance not available")
        return None
    
    stream = AudioStream()
    stream.channel_id = channel_id
    stream.device_index = device_index
    
    # Initialize Opus encoder
    try:
        stream.encoder = opuslib.Encoder(SAMPLE_RATE, 1, opuslib.APPLICATION_VOIP)
        try:
            stream.encoder.bitrate = 64000
            stream.encoder.vbr = True
        except AttributeError:
            # Some opuslib versions use different API
            pass
        print(f"[AUDIO_STREAM] Opus encoder created for channel {channel_id}")
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to create encoder for {channel_id}: {e}")
        return None
    
    # Initialize Opus decoder
    try:
        stream.decoder = opuslib.Decoder(SAMPLE_RATE, 1)
        print(f"[AUDIO_STREAM] Opus decoder created for channel {channel_id}")
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to create decoder for {channel_id}: {e}")
        return None
    
    # Initialize jitter buffer
    stream.output_jitter = JitterBuffer()
    
    # Initialize input buffer
    stream.input_buffer = np.zeros(4800, dtype=np.float32)
    stream.input_buffer_pos = 0
    
    print(f"[AUDIO_STREAM] AudioStream created for channel {channel_id} (device {device_index})")
    return stream


def start_stream(stream: AudioStream, pa_instance: pyaudio.PyAudio) -> bool:
    """
    Start audio input and output streams.
    
    Args:
        stream: AudioStream instance
        pa_instance: PyAudio instance
        
    Returns:
        True if streams started successfully, False otherwise
    """
    if pa_instance is None:
        print(f"[AUDIO_STREAM] ERROR: PyAudio instance not available")
        return False
    
    if stream.transmitting:
        print(f"[AUDIO_STREAM] Stream already transmitting for channel {stream.channel_id}")
        return True
    
    try:
        # Check if device index is valid
        device_info = pa_instance.get_device_info_by_index(stream.device_index)
        
        # Create input stream (required for transmission, but continue if it fails)
        try:
            stream.input_stream = pa_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=stream.device_index,
                frames_per_buffer=1024,
                stream_callback=None  # We'll use blocking I/O
            )
            stream.input_stream.start_stream()
            print(f"[AUDIO_STREAM] Input stream started for channel {stream.channel_id}")
        except Exception as e:
            print(f"[AUDIO_STREAM] WARNING: Input stream failed for {stream.channel_id}: {e}")
            stream.input_stream = None
            # Continue anyway - output stream is more important for receiving audio
        
        # Create output stream (CRITICAL - needed for audio playback)
        # This should always succeed if device supports output
        try:
            stream.output_stream = pa_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                output_device_index=stream.device_index,
                frames_per_buffer=1024,
                stream_callback=None
            )
            stream.output_stream.start_stream()
            print(f"[AUDIO_STREAM] ✅ Output stream started for channel {stream.channel_id} (device {stream.device_index})")
        except Exception as e:
            print(f"[AUDIO_STREAM] ❌ ERROR: Output stream FAILED for {stream.channel_id} (device {stream.device_index}): {e}")
            print(f"[AUDIO_STREAM] This channel will NOT be able to play received audio!")
            stream.output_stream = None
            # Don't fail completely - other channels might work
        
        stream.transmitting = True
        print(f"[AUDIO_STREAM] Audio transmission started for channel {stream.channel_id}")
        return True
        
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to start streams for {stream.channel_id}: {e}")
        cleanup_stream(stream)
        return False


def stop_stream(stream: AudioStream):
    """
    Stop audio input and output streams.
    
    Args:
        stream: AudioStream instance
    """
    stream.transmitting = False
    
    # Stop and close input stream
    if stream.input_stream:
        try:
            if stream.input_stream.is_active():
                stream.input_stream.stop_stream()
            stream.input_stream.close()
            print(f"[AUDIO_STREAM] Input stream stopped for channel {stream.channel_id}")
        except Exception as e:
            print(f"[AUDIO_STREAM] ERROR: Exception stopping input stream for {stream.channel_id}: {e}")
        finally:
            stream.input_stream = None
    
    # Stop and close output stream
    if stream.output_stream:
        try:
            if stream.output_stream.is_active():
                stream.output_stream.stop_stream()
            stream.output_stream.close()
            print(f"[AUDIO_STREAM] Output stream stopped for channel {stream.channel_id}")
        except Exception as e:
            print(f"[AUDIO_STREAM] ERROR: Exception stopping output stream for {stream.channel_id}: {e}")
        finally:
            stream.output_stream = None


def cleanup_stream(stream: AudioStream):
    """
    Cleanup and release all stream resources.
    
    Args:
        stream: AudioStream instance
    """
    stop_stream(stream)
    
    # Cleanup encoder
    if stream.encoder:
        try:
            del stream.encoder
        except Exception:
            pass
        stream.encoder = None
    
    # Cleanup decoder
    if stream.decoder:
        try:
            del stream.decoder
        except Exception:
            pass
        stream.decoder = None
    
    # Clear jitter buffer
    if stream.output_jitter:
        stream.output_jitter.clear()
    
    # Reset state
    stream.gpio_active = False
    stream.transmitting = False
    stream.input_buffer_pos = 0
    stream.current_output_frame_pos = 0
    
    print(f"[AUDIO_STREAM] Stream cleaned up for channel {stream.channel_id}")


# ============================================================================
# Audio Encoding/Decoding
# ============================================================================

def encode_audio(stream: AudioStream, samples: np.ndarray) -> Optional[bytes]:
    """
    Encode audio samples to Opus format.
    
    Args:
        stream: AudioStream instance
        samples: Audio samples as numpy array (float32, mono)
        
    Returns:
        Opus encoded data bytes or None on error
    """
    if stream.encoder is None:
        return None
    
    try:
        # Ensure samples are float32 and in correct range [-1.0, 1.0]
        samples_float32 = samples.astype(np.float32)
        
        # Clip to valid range
        samples_float32 = np.clip(samples_float32, -1.0, 1.0)
        
        # Convert to bytes for Opus encoder
        samples_bytes = samples_float32.tobytes()
        
        # Encode to Opus
        opus_data = stream.encoder.encode(samples_bytes, SAMPLES_PER_FRAME)
        
        return opus_data
        
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to encode audio for {stream.channel_id}: {e}")
        return None


def decode_audio(stream: AudioStream, opus_data: bytes) -> Optional[np.ndarray]:
    """
    Decode Opus data to audio samples.
    
    Args:
        stream: AudioStream instance
        opus_data: Opus encoded data bytes
        
    Returns:
        Audio samples as numpy array (float32, mono) or None on error
    """
    if stream.decoder is None:
        return None
    
    try:
        # Decode from Opus
        decoded_bytes = stream.decoder.decode(opus_data, SAMPLES_PER_FRAME)
        
        # Convert bytes to numpy array (float32)
        samples = np.frombuffer(decoded_bytes, dtype=np.float32)
        
        return samples
        
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to decode audio for {stream.channel_id}: {e}")
        return None


# ============================================================================
# Encryption Key Management
# ============================================================================

def set_encryption_key(stream: AudioStream, key: bytes):
    """
    Set encryption key for the stream.
    
    Args:
        stream: AudioStream instance
        key: Encryption key bytes (must be 32 bytes for AES-256)
    """
    if len(key) != 32:
        print(f"[AUDIO_STREAM] WARNING: Encryption key length is {len(key)}, expected 32 bytes")
        # Pad or truncate to 32 bytes
        if len(key) < 32:
            key = key + b'\x00' * (32 - len(key))
        else:
            key = key[:32]
    
    stream.encryption_key = key
    print(f"[AUDIO_STREAM] Encryption key set for channel {stream.channel_id}")


def get_encryption_key(stream: AudioStream) -> bytes:
    """
    Get encryption key for the stream.
    
    Args:
        stream: AudioStream instance
        
    Returns:
        Encryption key bytes (32 bytes)
    """
    return stream.encryption_key

