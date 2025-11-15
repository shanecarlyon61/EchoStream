
import pyaudio
import opuslib
import numpy as np
from typing import Optional
from echostream import JitterBuffer, SAMPLE_RATE, SAMPLES_PER_FRAME

class AudioStream:

    def __init__(self):

        self.channel_id: str = ""
        self.device_index: int = -1

        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

        self.encoder: Optional[opuslib.Encoder] = None
        self.decoder: Optional[opuslib.Decoder] = None

        self.output_jitter: JitterBuffer = JitterBuffer()

        self.encryption_key: bytes = bytes(32)

        self.gpio_active: bool = False
        self.transmitting: bool = False

        self.input_buffer: np.ndarray = np.zeros(9600, dtype=np.float32)
        self.input_buffer_pos: int = 0
        self.current_output_frame_pos: int = 0

        self.buffer_size: int = 9600

def create_stream(channel_id: str, device_index: int, pa_instance: pyaudio.PyAudio) -> Optional[AudioStream]:

    if pa_instance is None:
        print(f"[AUDIO_STREAM] ERROR: PyAudio instance not available")
        return None

    stream = AudioStream()
    stream.channel_id = channel_id
    stream.device_index = device_index

    try:
        stream.encoder = opuslib.Encoder(SAMPLE_RATE, 1, opuslib.APPLICATION_VOIP)
        try:
            stream.encoder.bitrate = 64000
            stream.encoder.vbr = True
        except AttributeError:

            pass
        print(f"[AUDIO_STREAM] Opus encoder created for channel {channel_id}")
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to create encoder for {channel_id}: {e}")
        return None

    try:
        stream.decoder = opuslib.Decoder(SAMPLE_RATE, 1)
        print(f"[AUDIO_STREAM] Opus decoder created for channel {channel_id}")
    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to create decoder for {channel_id}: {e}")
        return None

    stream.output_jitter = JitterBuffer()

    stream.input_buffer = np.zeros(9600, dtype=np.float32)
    stream.input_buffer_pos = 0

    print(f"[AUDIO_STREAM] AudioStream created for channel {channel_id} (device {device_index})")
    return stream

def start_stream(stream: AudioStream, pa_instance: pyaudio.PyAudio) -> bool:

    if pa_instance is None:
        print(f"[AUDIO_STREAM] ERROR: PyAudio instance not available")
        return False

    if stream.transmitting:
        print(f"[AUDIO_STREAM] Stream already transmitting for channel {stream.channel_id}")
        return True

    try:

        device_info = pa_instance.get_device_info_by_index(stream.device_index)

        try:
            stream.input_stream = pa_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=stream.device_index,
                frames_per_buffer=2048,
                stream_callback=None
            )
            stream.input_stream.start_stream()
            print(f"[AUDIO_STREAM] Input stream started for channel {stream.channel_id}")
        except Exception as e:
            print(f"[AUDIO_STREAM] WARNING: Input stream failed for {stream.channel_id}: {e}")
            stream.input_stream = None

        try:
            stream.output_stream = pa_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                output_device_index=stream.device_index,
                frames_per_buffer=2048,
                stream_callback=None
            )
            stream.output_stream.start_stream()
            print(f"[AUDIO_STREAM] ✅ Output stream started for channel {stream.channel_id} (device {stream.device_index})")
        except Exception as e:
            print(f"[AUDIO_STREAM] ❌ ERROR: Output stream FAILED for {stream.channel_id} (device {stream.device_index}): {e}")
            print(f"[AUDIO_STREAM] This channel will NOT be able to play received audio!")
            stream.output_stream = None

        stream.transmitting = True
        print(f"[AUDIO_STREAM] Audio transmission started for channel {stream.channel_id}")
        return True

    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to start streams for {stream.channel_id}: {e}")
        cleanup_stream(stream)
        return False

def stop_stream(stream: AudioStream):

    stream.transmitting = False

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

    stop_stream(stream)

    if stream.encoder:
        try:
            del stream.encoder
        except Exception:
            pass
        stream.encoder = None

    if stream.decoder:
        try:
            del stream.decoder
        except Exception:
            pass
        stream.decoder = None

    if stream.output_jitter:
        stream.output_jitter.clear()

    stream.gpio_active = False
    stream.transmitting = False
    stream.input_buffer_pos = 0
    stream.current_output_frame_pos = 0

    print(f"[AUDIO_STREAM] Stream cleaned up for channel {stream.channel_id}")

def encode_audio(stream: AudioStream, samples: np.ndarray) -> Optional[bytes]:

    if stream.encoder is None:
        return None

    try:

        samples_float32 = samples.astype(np.float32)

        samples_float32 = np.clip(samples_float32, -1.0, 1.0)

        samples_bytes = samples_float32.tobytes()

        opus_data = stream.encoder.encode(samples_bytes, SAMPLES_PER_FRAME)

        return opus_data

    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to encode audio for {stream.channel_id}: {e}")
        return None

def decode_audio(stream: AudioStream, opus_data: bytes) -> Optional[np.ndarray]:

    if stream.decoder is None:
        return None

    try:

        decoded_bytes = stream.decoder.decode(opus_data, SAMPLES_PER_FRAME)

        samples = np.frombuffer(decoded_bytes, dtype=np.float32)

        return samples

    except Exception as e:
        print(f"[AUDIO_STREAM] ERROR: Failed to decode audio for {stream.channel_id}: {e}")
        return None

def set_encryption_key(stream: AudioStream, key: bytes):

    if len(key) != 32:
        print(f"[AUDIO_STREAM] WARNING: Encryption key length is {len(key)}, expected 32 bytes")

        if len(key) < 32:
            key = key + b'\x00' * (32 - len(key))
        else:
            key = key[:32]

    stream.encryption_key = key
    print(f"[AUDIO_STREAM] Encryption key set for channel {stream.channel_id}")

def get_encryption_key(stream: AudioStream) -> bytes:

    return stream.encryption_key

