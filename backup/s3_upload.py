"""
S3 upload - handles audio recording and S3 upload functionality
"""

import os
import struct
import time
import threading
from typing import Optional

try:
    import boto3

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("[WARNING] boto3 not available, S3 upload functionality disabled")


# Audio recording context
class AudioRecordingContext:
    def __init__(self):
        self.recording_file: Optional[object] = None
        self.is_recording = False
        self.tone_a_hz = 0.0
        self.tone_b_hz = 0.0
        self.duration_ms = 0
        self.start_time_ms = 0
        self.filename = ""
        self.sample_rate = 48000
        self.bits_per_sample = 16
        self.channels = 1
        self.samples_written = 0
        self.mutex = threading.Lock()


recording_state = AudioRecordingContext()
known_recording_state = AudioRecordingContext()


def write_wav_header(
    file, sample_rate: int, bits_per_sample: int, channels: int, data_bytes: int
):
    """Write WAV file header"""
    file.write(b"RIFF")
    file.write(struct.pack("<I", 36 + data_bytes))
    file.write(b"WAVE")
    file.write(b"fmt ")
    file.write(struct.pack("<I", 16))  # fmt chunk size
    file.write(struct.pack("<H", 1))  # audio format (PCM)
    file.write(struct.pack("<H", channels))
    file.write(struct.pack("<I", sample_rate))
    file.write(struct.pack("<I", sample_rate * channels * bits_per_sample // 8))
    file.write(struct.pack("<H", channels * bits_per_sample // 8))
    file.write(struct.pack("<H", bits_per_sample))
    file.write(b"data")
    file.write(struct.pack("<I", data_bytes))


def start_new_tone_audio_recording(
    tone_a_hz: float, tone_b_hz: float, duration_ms: int
) -> bool:
    """Start new tone audio recording"""
    global recording_state

    with recording_state.mutex:
        if recording_state.is_recording:
            return False

        recording_state.tone_a_hz = tone_a_hz
        recording_state.tone_b_hz = tone_b_hz
        recording_state.duration_ms = duration_ms
        recording_state.start_time_ms = int(time.time() * 1000)
        recording_state.samples_written = 0

        # Create filename
        timestamp = int(time.time())
        recording_state.filename = f"/tmp/tone_recording_{timestamp}.wav"

        try:
            recording_state.recording_file = open(recording_state.filename, "wb")
            # Write placeholder header (will be updated when recording stops)
            write_wav_header(
                recording_state.recording_file,
                recording_state.sample_rate,
                recording_state.bits_per_sample,
                recording_state.channels,
                0,
            )
            recording_state.is_recording = True
            print(f"[INFO] Started new tone recording: {recording_state.filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            return False


def write_audio_samples_to_recording(
    samples: list, sample_count: int, sample_rate: int
) -> bool:
    """Write audio samples to recording"""
    global recording_state

    with recording_state.mutex:
        if not recording_state.is_recording or recording_state.recording_file is None:
            return False

        try:
            # Convert float samples to int16
            int16_samples = []
            for sample in samples[:sample_count]:
                # Clamp and convert
                sample = max(-1.0, min(1.0, sample))
                int16_samples.append(int(sample * 32767))

            # Write samples
            for sample in int16_samples:
                recording_state.recording_file.write(struct.pack("<h", sample))

            recording_state.samples_written += len(int16_samples)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write samples: {e}")
            return False


def stop_new_tone_audio_recording():
    """Stop new tone audio recording"""
    global recording_state

    with recording_state.mutex:
        if not recording_state.is_recording:
            return

        try:
            if recording_state.recording_file:
                # Update WAV header with actual data size
                data_bytes = (
                    recording_state.samples_written * 2
                )  # 16-bit = 2 bytes per sample
                recording_state.recording_file.seek(0)
                write_wav_header(
                    recording_state.recording_file,
                    recording_state.sample_rate,
                    recording_state.bits_per_sample,
                    recording_state.channels,
                    data_bytes,
                )
                recording_state.recording_file.close()
                recording_state.recording_file = None

            recording_state.is_recording = False
            print(f"[INFO] Stopped new tone recording: {recording_state.filename}")
        except Exception as e:
            print(f"[ERROR] Failed to stop recording: {e}")


def is_new_tone_recording_active() -> bool:
    """Check if new tone recording is active"""
    with recording_state.mutex:
        return recording_state.is_recording


def start_known_tone_audio_recording(
    tone_a_hz: float, tone_b_hz: float, duration_ms: int
) -> bool:
    """Start known tone audio recording"""
    global known_recording_state
    # Similar to start_new_tone_audio_recording
    return start_new_tone_audio_recording(tone_a_hz, tone_b_hz, duration_ms)


def write_audio_samples_to_known_recording(
    samples: list, sample_count: int, sample_rate: int
) -> bool:
    """Write audio samples to known recording"""
    # Similar to write_audio_samples_to_recording
    return write_audio_samples_to_recording(samples, sample_count, sample_rate)


def stop_known_tone_audio_recording():
    """Stop known tone audio recording"""
    stop_new_tone_audio_recording()


def is_known_tone_recording_active() -> bool:
    """Check if known tone recording is active"""
    return is_new_tone_recording_active()


def upload_audio_to_s3(file_path: str, tone_a_hz: float, tone_b_hz: float) -> bool:
    """Upload audio file to S3"""
    if not S3_AVAILABLE:
        print("[WARNING] S3 upload not available - boto3 library not installed")
        return False

    try:
        s3_client = boto3.client("s3")

        # Generate S3 key
        timestamp = int(time.time())
        s3_key = f"audio_recordings/{timestamp}_{tone_a_hz}_{tone_b_hz}.wav"

        # Upload file
        s3_client.upload_file(file_path, "your-bucket-name", s3_key)

        print(f"[INFO] Uploaded {file_path} to S3: {s3_key}")
        return True
    except Exception as e:
        print(f"[ERROR] S3 upload failed: {e}")
        return False


def play_recorded_audio_on_passthrough(file_path: str):
    """Play recorded audio file on passthrough channel"""
    # Simplified - would need audio playback implementation
    print(f"[INFO] Would play {file_path} on passthrough channel")
