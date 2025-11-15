
import threading
import time
import numpy as np
from typing import Optional
from echostream import (
    global_interrupted, SAMPLE_RATE, SAMPLES_PER_FRAME,
    JitterBuffer
)

class SharedAudioBuffer:

    def __init__(self, buffer_size: int = SAMPLES_PER_FRAME):

        self.samples = np.zeros(buffer_size, dtype=np.float32)
        self.sample_count = 0
        self.valid = False
        self.mutex = threading.Lock()
        self.data_ready = threading.Condition(self.mutex)

    def write(self, samples: np.ndarray, sample_count: int):

        with self.mutex:

            copy_count = min(sample_count, len(self.samples))
            if copy_count > 0:
                self.samples[:copy_count] = samples[:copy_count]
                self.sample_count = copy_count
                self.valid = True

                self.data_ready.notify_all()

    def read(self) -> tuple[Optional[np.ndarray], int]:

        with self.mutex:
            if not self.valid or self.sample_count == 0:
                return None, 0

            samples = self.samples[:self.sample_count].copy()
            count = self.sample_count
            return samples, count

    def clear(self):
        with self.mutex:
            self.samples.fill(0.0)
            self.sample_count = 0
            self.valid = False

global_shared_buffer = SharedAudioBuffer()

def audio_input_worker(stream):

    import audio_stream

    print(f"[AUDIO_INPUT] Input worker started for channel {stream.channel_id}")

    frame_count = 0
    error_count = 0

    try:
        while not global_interrupted.is_set() and stream.transmitting:
            try:

                if not stream.gpio_active:

                    time.sleep(0.01)
                    continue

                if stream.input_stream is None or not stream.input_stream.is_active():
                    time.sleep(0.01)
                    continue

                try:
                    raw_data = stream.input_stream.read(1024, exception_on_overflow=False)
                except Exception as e:
                    error_count += 1
                    if error_count % 100 == 0:
                        print(f"[AUDIO_INPUT] WARNING: Read error for {stream.channel_id} (error #{error_count}): {e}")
                    time.sleep(0.01)
                    continue

                samples = np.frombuffer(raw_data, dtype=np.float32)

                remaining = len(samples)
                while remaining > 0:
                    available = min(remaining, len(stream.input_buffer) - stream.input_buffer_pos)
                    if available > 0:
                        stream.input_buffer[
                            stream.input_buffer_pos:stream.input_buffer_pos + available
                        ] = samples[len(samples) - remaining:len(samples) - remaining + available]
                        stream.input_buffer_pos += available
                        remaining -= available

                    if stream.input_buffer_pos >= SAMPLES_PER_FRAME:

                        frame_samples = stream.input_buffer[:SAMPLES_PER_FRAME]

                        write_to_shared_buffer(frame_samples, SAMPLES_PER_FRAME)

                        opus_data = audio_stream.encode_audio(stream, frame_samples)

                        if opus_data:

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

                        stream.input_buffer[:-SAMPLES_PER_FRAME] = stream.input_buffer[SAMPLES_PER_FRAME:]
                        stream.input_buffer_pos -= SAMPLES_PER_FRAME

                frame_count += 1

                if frame_count % 5000 == 0:
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

def audio_output_worker(stream):

    import audio_stream

    print(f"[AUDIO_OUTPUT] Output worker started for channel {stream.channel_id}")

    frame_count = 0
    silence_count = 0
    error_count = 0

    silence_samples = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)

    try:
        while not global_interrupted.is_set() and stream.transmitting:
            try:
                if stream.output_stream is None or not stream.output_stream.is_active():
                    time.sleep(0.01)
                    continue

                result = stream.output_jitter.read()

                if result:
                    samples, sample_count = result
                    if samples is not None and sample_count > 0:

                        try:

                            samples_to_write = samples[:sample_count] if sample_count < len(samples) else samples

                            if samples_to_write.dtype != np.float32:
                                samples_to_write = samples_to_write.astype(np.float32)

                            stream.output_stream.write(
                                samples_to_write.tobytes(),
                                exception_on_underflow=False
                            )
                            silence_count = 0
                            frame_count += 1

                            if frame_count % 5000 == 0:
                                jitter_frames = stream.output_jitter.get_frame_count()
                                print(f"[AUDIO_OUTPUT] Channel {stream.channel_id}: Played {frame_count} frames (jitter={jitter_frames})")
                        except Exception as e:
                            error_count += 1
                            if error_count % 100 == 0:
                                print(f"[AUDIO_OUTPUT] WARNING: Write error for {stream.channel_id} (error #{error_count}): {e}")
                            time.sleep(0.01)
                    else:

                        stream.output_stream.write(
                            silence_samples.tobytes(),
                            exception_on_underflow=False
                        )
                        silence_count += 1
                else:

                    try:
                        stream.output_stream.write(
                            silence_samples.tobytes(),
                            exception_on_underflow=False
                        )
                        silence_count += 1

                        if silence_count % 1000 == 0:
                            print(f"[AUDIO_OUTPUT] Channel {stream.channel_id}: Playing silence (fallback, jitter empty)")
                    except Exception as e:
                        error_count += 1
                        if error_count % 100 == 0:
                            print(f"[AUDIO_OUTPUT] WARNING: Silence write error for {stream.channel_id}: {e}")
                        time.sleep(0.01)

                time.sleep(0.001)

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

def write_to_shared_buffer(samples: np.ndarray, sample_count: int):

    global_shared_buffer.write(samples, sample_count)

def write_to_jitter_buffer(stream, samples: np.ndarray, sample_count: int) -> bool:

    if stream.output_jitter is None:
        return False

    return stream.output_jitter.write(samples, sample_count)

def init_shared_audio_buffer() -> bool:

    global global_shared_buffer

    if global_shared_buffer is None:
        global_shared_buffer = SharedAudioBuffer()

    print("[AUDIO_WORKERS] Shared audio buffer initialized")
    return True

