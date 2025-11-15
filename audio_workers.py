"""
Audio Workers - Background threads for audio I/O

This module provides worker threads for audio input (microphone capture)
and audio output (speaker playback), plus a shared audio buffer for
tone detection processing.
"""
import threading
import time
import numpy as np
from typing import Optional
from echostream import (
    global_interrupted, SAMPLE_RATE, SAMPLES_PER_FRAME,
    JitterBuffer
)
# Import at runtime to avoid circular dependencies
# from audio_stream import AudioStream, encode_audio, decode_audio


# ============================================================================
# Shared Audio Buffer for Tone Detection
# ============================================================================

class SharedAudioBuffer:
    """
    Thread-safe shared audio buffer for tone detection.
    
    Stores audio samples that can be accessed by the tone detection worker
    thread for analysis.
    """
    def __init__(self, buffer_size: int = SAMPLES_PER_FRAME):
        """
        Initialize shared audio buffer.
        
        Args:
            buffer_size: Size of sample buffer
        """
        self.samples = np.zeros(buffer_size, dtype=np.float32)
        self.sample_count = 0
        self.valid = False
        self.mutex = threading.Lock()
        self.data_ready = threading.Condition(self.mutex)
    
    def write(self, samples: np.ndarray, sample_count: int):
        """
        Write samples to the buffer (thread-safe).
        
        Args:
            samples: Audio samples array
            sample_count: Number of samples to write
        """
        with self.mutex:
            # Copy samples (up to buffer size)
            copy_count = min(sample_count, len(self.samples))
            if copy_count > 0:
                self.samples[:copy_count] = samples[:copy_count]
                self.sample_count = copy_count
                self.valid = True
                # Notify waiting threads
                self.data_ready.notify_all()
    
    def read(self) -> tuple[Optional[np.ndarray], int]:
        """
        Read samples from the buffer (thread-safe).
        
        Returns:
            Tuple of (samples array, sample_count) or (None, 0) if no data
        """
        with self.mutex:
            if not self.valid or self.sample_count == 0:
                return None, 0
            
            # Copy samples
            samples = self.samples[:self.sample_count].copy()
            count = self.sample_count
            return samples, count
    
    def clear(self):
        """Clear the buffer."""
        with self.mutex:
            self.samples.fill(0.0)
            self.sample_count = 0
            self.valid = False


# Global shared buffer instance
global_shared_buffer = SharedAudioBuffer()


# ============================================================================
# Audio Input Worker Thread
# ============================================================================

def audio_input_worker(stream):
    """
    Audio input worker thread - captures audio from microphone.
    
    This worker continuously reads audio samples from the input stream,
    encodes them to Opus, and can send them via UDP. Also writes samples
    to shared buffer for tone detection.
    
    Args:
        stream: AudioStream instance to process
    """
    # Import at runtime to avoid circular dependencies
    import audio_stream
    
    print(f"[AUDIO_INPUT] Input worker started for channel {stream.channel_id}")
    
    frame_count = 0
    error_count = 0
    
    try:
        while not global_interrupted.is_set() and stream.transmitting:
            try:
                # Check if GPIO is active (PTT pressed)
                if not stream.gpio_active:
                    # GPIO not active, sleep and skip
                    time.sleep(0.01)
                    continue
                
                # Read audio samples from input stream
                if stream.input_stream is None or not stream.input_stream.is_active():
                    time.sleep(0.01)
                    continue
                
                # Read one frame of samples (1024 samples = ~21ms at 48kHz)
                try:
                    raw_data = stream.input_stream.read(1024, exception_on_overflow=False)
                except Exception as e:
                    error_count += 1
                    if error_count % 100 == 0:
                        print(f"[AUDIO_INPUT] WARNING: Read error for {stream.channel_id} (error #{error_count}): {e}")
                    time.sleep(0.01)
                    continue
                
                # Convert bytes to numpy array (float32)
                samples = np.frombuffer(raw_data, dtype=np.float32)
                
                # Accumulate samples in input buffer
                remaining = len(samples)
                while remaining > 0:
                    available = min(remaining, len(stream.input_buffer) - stream.input_buffer_pos)
                    if available > 0:
                        stream.input_buffer[
                            stream.input_buffer_pos:stream.input_buffer_pos + available
                        ] = samples[len(samples) - remaining:len(samples) - remaining + available]
                        stream.input_buffer_pos += available
                        remaining -= available
                    
                    # When buffer has enough samples for one Opus frame, process it
                    if stream.input_buffer_pos >= SAMPLES_PER_FRAME:
                        # Extract one frame
                        frame_samples = stream.input_buffer[:SAMPLES_PER_FRAME]
                        
                        # Write to shared buffer for tone detection
                        write_to_shared_buffer(frame_samples, SAMPLES_PER_FRAME)
                        
                        # Encode to Opus
                        opus_data = audio_stream.encode_audio(stream, frame_samples)
                        
                        if opus_data:
                            # Send via UDP (will be handled by UDP manager)
                            try:
                                import udp_manager
                                if udp_manager.is_udp_ready():
                                    udp_manager.send_audio_packet(
                                        stream.channel_id,
                                        opus_data,
                                        stream.encryption_key
                                    )
                            except ImportError:
                                pass
                        
                        # Shift buffer
                        stream.input_buffer[:-SAMPLES_PER_FRAME] = stream.input_buffer[SAMPLES_PER_FRAME:]
                        stream.input_buffer_pos -= SAMPLES_PER_FRAME
                
                frame_count += 1
                
                # Log periodically
                if frame_count % 5000 == 0:  # Every ~100 seconds at 48kHz
                    print(f"[AUDIO_INPUT] Channel {stream.channel_id}: Processed {frame_count} frames")
                
            except Exception as e:
                if not global_interrupted.is_set():
                    error_count += 1
                    if error_count % 100 == 0:
                        print(f"[AUDIO_INPUT] ERROR: Exception in input worker for {stream.channel_id}: {e}")
                time.sleep(0.01)
    
    except Exception as e:
        if not global_interrupted.is_set():
            print(f"[AUDIO_INPUT] FATAL: Input worker crashed for {stream.channel_id}: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        print(f"[AUDIO_INPUT] Input worker stopped for channel {stream.channel_id}")


# ============================================================================
# Audio Output Worker Thread
# ============================================================================

def audio_output_worker(stream):
    """
    Audio output worker thread - plays audio to speaker.
    
    This worker continuously reads Opus-encoded audio from the jitter buffer,
    decodes it, and plays it to the output stream.
    
    Args:
        stream: AudioStream instance to process
    """
    # Import at runtime to avoid circular dependencies
    import audio_stream
    
    print(f"[AUDIO_OUTPUT] Output worker started for channel {stream.channel_id}")
    
    frame_count = 0
    silence_count = 0
    error_count = 0
    
    # Generate silence frame for when buffer is empty
    silence_samples = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
    
    try:
        while not global_interrupted.is_set() and stream.transmitting:
            try:
                if stream.output_stream is None or not stream.output_stream.is_active():
                    time.sleep(0.01)
                    continue
                
                # Read decoded float samples from jitter buffer (no Opus decode needed)
                result = stream.output_jitter.read()
                
                if result:
                    samples, sample_count = result
                    if samples is not None and sample_count > 0:
                        # Write samples to output stream (only valid samples)
                        try:
                            samples_to_write = samples[:sample_count] if sample_count < len(samples) else samples
                            stream.output_stream.write(
                                samples_to_write.tobytes(),
                                exception_on_underflow=False
                            )
                            silence_count = 0
                            frame_count += 1
                            
                            # Log periodically
                            if frame_count % 5000 == 0:  # Every ~100 seconds
                                jitter_frames = stream.output_jitter.get_frame_count()
                                print(f"[AUDIO_OUTPUT] Channel {stream.channel_id}: Played {frame_count} frames (jitter={jitter_frames})")
                        except Exception as e:
                            error_count += 1
                            if error_count % 100 == 0:
                                print(f"[AUDIO_OUTPUT] WARNING: Write error for {stream.channel_id} (error #{error_count}): {e}")
                            time.sleep(0.01)
                    else:
                        # Decode failed, play silence
                        stream.output_stream.write(
                            silence_samples.tobytes(),
                            exception_on_underflow=False
                        )
                        silence_count += 1
                else:
                    # No data in buffer, play silence
                    try:
                        stream.output_stream.write(
                            silence_samples.tobytes(),
                            exception_on_underflow=False
                        )
                        silence_count += 1
                        
                        # Log if playing silence for extended period
                        if silence_count % 1000 == 0:  # Every ~10 seconds
                            print(f"[AUDIO_OUTPUT] Channel {stream.channel_id}: Playing silence (fallback, jitter empty)")
                    except Exception as e:
                        error_count += 1
                        if error_count % 100 == 0:
                            print(f"[AUDIO_OUTPUT] WARNING: Silence write error for {stream.channel_id}: {e}")
                        time.sleep(0.01)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                if not global_interrupted.is_set():
                    error_count += 1
                    if error_count % 100 == 0:
                        print(f"[AUDIO_OUTPUT] ERROR: Exception in output worker for {stream.channel_id}: {e}")
                time.sleep(0.01)
    
    except Exception as e:
        if not global_interrupted.is_set():
            print(f"[AUDIO_OUTPUT] FATAL: Output worker crashed for {stream.channel_id}: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        print(f"[AUDIO_OUTPUT] Output worker stopped for channel {stream.channel_id}")


# ============================================================================
# Shared Buffer and Jitter Buffer Functions
# ============================================================================

def write_to_shared_buffer(samples: np.ndarray, sample_count: int):
    """
    Write samples to shared audio buffer for tone detection.
    
    Args:
        samples: Audio samples array
        sample_count: Number of samples
    """
    global_shared_buffer.write(samples, sample_count)




def write_to_jitter_buffer(stream, samples: np.ndarray, sample_count: int) -> bool:
    """
    Write decoded float samples to jitter buffer.
    Matches C code: stores float samples, not Opus bytes.
    
    Args:
        stream: AudioStream instance
        samples: Decoded float32 audio samples
        sample_count: Number of valid samples
        
    Returns:
        True if written successfully, False if buffer is full
    """
    if stream.output_jitter is None:
        return False
    
    return stream.output_jitter.write(samples, sample_count)


def init_shared_audio_buffer() -> bool:
    """
    Initialize shared audio buffer (called at startup).
    
    Returns:
        True if initialization successful
    """
    global global_shared_buffer
    
    if global_shared_buffer is None:
        global_shared_buffer = SharedAudioBuffer()
    
    print("[AUDIO_WORKERS] Shared audio buffer initialized")
    return True

