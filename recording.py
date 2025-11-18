import threading
import time
import struct
import os
from typing import Dict, Optional
from collections import deque
import numpy as np

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("[RECORDING] WARNING: boto3 not available, S3 upload disabled")

S3_BUCKET_NAME = "redenes02679517"
S3_REGION = "us-west-2"
SAMPLE_RATE = 48000
BITS_PER_SAMPLE = 16
CHANNELS = 1


class RecordingSession:
    def __init__(self, channel_id: str, tone_type: str, tone_a_hz: float, tone_b_hz: float, duration_ms: int):
        self.channel_id = channel_id
        self.tone_type = tone_type
        self.tone_a_hz = tone_a_hz
        self.tone_b_hz = tone_b_hz
        self.duration_ms = duration_ms
        self.start_time_ms = int(time.time() * 1000)
        self.end_time_ms = self.start_time_ms + duration_ms
        self.filename = ""
        self.recording_file = None
        self.samples_written = 0
        self.audio_queue = deque(maxlen=100)
        self.write_thread = None
        self.write_thread_running = False


def write_wav_header(file, sample_rate: int, bits_per_sample: int, channels: int, data_bytes: int):
    file.write(b'RIFF')
    file.write(struct.pack('<I', 36 + data_bytes))
    file.write(b'WAVE')
    file.write(b'fmt ')
    file.write(struct.pack('<I', 16))
    file.write(struct.pack('<H', 1))
    file.write(struct.pack('<H', channels))
    file.write(struct.pack('<I', sample_rate))
    file.write(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))
    file.write(struct.pack('<H', channels * bits_per_sample // 8))
    file.write(struct.pack('<H', bits_per_sample))
    file.write(b'data')
    file.write(struct.pack('<I', data_bytes))


class RecordingManager:
    def __init__(self):
        self.mutex = threading.Lock()
        self.active_sessions: Dict[str, RecordingSession] = {}
        self.temp_dir = "/tmp"
        
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
            except Exception:
                self.temp_dir = os.path.expanduser("~/tmp")
                os.makedirs(self.temp_dir, exist_ok=True)
    
    def start_recording(self, channel_id: str, tone_type: str, tone_a_hz: float, tone_b_hz: float, duration_ms: int) -> bool:
        if duration_ms <= 0:
            return False
        
        with self.mutex:
            current_time_ms = int(time.time() * 1000)
            
            if channel_id in self.active_sessions:
                existing = self.active_sessions[channel_id]
                if existing.end_time_ms > current_time_ms:
                    remaining_time_ms = existing.end_time_ms - current_time_ms
                    new_duration_ms = max(remaining_time_ms, duration_ms)
                    existing.end_time_ms = current_time_ms + new_duration_ms
                    existing.duration_ms = new_duration_ms
                    print(f"[RECORDING] Session already active for {channel_id}, "
                          f"extending duration to {new_duration_ms} ms")
                    return True
                else:
                    self._stop_session(channel_id)
            
            timestamp = int(time.time() * 1000)
            if tone_type == "new":
                filename = f"{self.temp_dir}/recording_{channel_id}_new_{tone_a_hz:.1f}_{tone_b_hz:.1f}_{timestamp}.wav"
            else:
                filename = f"{self.temp_dir}/recording_{channel_id}_{tone_a_hz:.1f}_{tone_b_hz:.1f}_{timestamp}.wav"
            
            try:
                recording_file = open(filename, 'wb')
                write_wav_header(recording_file, SAMPLE_RATE, BITS_PER_SAMPLE, CHANNELS, 0)
                
                session = RecordingSession(channel_id, tone_type, tone_a_hz, tone_b_hz, duration_ms)
                session.filename = filename
                session.recording_file = recording_file
                session.write_thread_running = True
                
                write_thread = threading.Thread(
                    target=self._write_worker,
                    args=(channel_id,),
                    daemon=True
                )
                session.write_thread = write_thread
                write_thread.start()
                
                self.active_sessions[channel_id] = session
                print(f"[RECORDING] Started recording: {channel_id}, type={tone_type}, "
                      f"tones=({tone_a_hz:.1f}, {tone_b_hz:.1f}) Hz, duration={duration_ms} ms")
                print(f"[RECORDING] File: {filename}")
                return True
            except Exception as e:
                print(f"[RECORDING] ERROR: Failed to start recording: {e}")
                if recording_file:
                    try:
                        recording_file.close()
                    except Exception:
                        pass
                return False
    
    def _write_worker(self, channel_id: str):
        while True:
            try:
                session = None
                with self.mutex:
                    if channel_id not in self.active_sessions:
                        break
                    session = self.active_sessions[channel_id]
                    if not session.write_thread_running:
                        break
                
                if not session or not session.recording_file:
                    time.sleep(0.01)
                    continue
                
                queue = session.audio_queue
                if not queue:
                    time.sleep(0.01)
                    continue
                
                if queue:
                    try:
                        audio_samples = queue.popleft()
                        int16_samples = (np.clip(audio_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
                        for sample in int16_samples:
                            session.recording_file.write(struct.pack('<h', sample))
                        session.samples_written += len(int16_samples)
                    except IndexError:
                        pass
                
                time.sleep(0.001)
            except Exception as e:
                with self.mutex:
                    if channel_id in self.active_sessions:
                        print(f"[RECORDING] ERROR in write worker: {e}")
                time.sleep(0.01)
        
        print(f"[RECORDING] Write worker stopped for channel {channel_id}")
    
    def route_audio(self, channel_id: str, audio_samples: np.ndarray) -> bool:
        try:
            with self.mutex:
                if channel_id not in self.active_sessions:
                    return False
                
                session = self.active_sessions[channel_id]
                current_time_ms = int(time.time() * 1000)
                
                if current_time_ms >= session.end_time_ms:
                    self._stop_session(channel_id)
                    return False
                
                queue = session.audio_queue
            
            if queue is not None:
                try:
                    queue.append(audio_samples.copy())
                    return True
                except Exception as e:
                    print(f"[RECORDING] ERROR: Failed to queue audio: {e}")
            return False
        except Exception as e:
            print(f"[RECORDING] ERROR: Exception in route_audio: {e}")
            return False
    
    def is_active(self, channel_id: str) -> bool:
        with self.mutex:
            if channel_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[channel_id]
            current_time_ms = int(time.time() * 1000)
            
            if current_time_ms >= session.end_time_ms:
                self._stop_session(channel_id)
                return False
            
            return True
    
    def _stop_session(self, channel_id: str):
        session = None
        with self.mutex:
            if channel_id not in self.active_sessions:
                return
            session = self.active_sessions[channel_id]
        
        if session:
            session.write_thread_running = False
            
            if session.write_thread and session.write_thread.is_alive():
                session.write_thread.join(timeout=1.0)
            
            if session.recording_file:
                try:
                    data_bytes = session.samples_written * 2
                    session.recording_file.seek(0)
                    write_wav_header(session.recording_file, SAMPLE_RATE, BITS_PER_SAMPLE, CHANNELS, data_bytes)
                    session.recording_file.close()
                    print(f"[RECORDING] Stopped recording: {session.filename} ({session.samples_written} samples)")
                except Exception as e:
                    print(f"[RECORDING] ERROR: Failed to finalize WAV file: {e}")
            
            filename = session.filename
            tone_type = session.tone_type
            tone_a_hz = session.tone_a_hz
            tone_b_hz = session.tone_b_hz
        
        with self.mutex:
            if channel_id in self.active_sessions:
                del self.active_sessions[channel_id]
        
        if filename and os.path.exists(filename):
            self._upload_to_s3(filename, tone_type, tone_a_hz, tone_b_hz, channel_id)
    
    def _upload_to_s3(self, file_path: str, tone_type: str, tone_a_hz: float, tone_b_hz: float, channel_id: str):
        if not S3_AVAILABLE:
            print(f"[RECORDING] WARNING: S3 upload not available - keeping local file: {file_path}")
            return
        
        def upload_thread():
            try:
                s3_client = boto3.client('s3', region_name=S3_REGION)
                
                timestamp = int(time.time() * 1000)
                if tone_type == "new":
                    s3_key = f"audio_recordings/{channel_id}_new_{tone_a_hz:.1f}_{tone_b_hz:.1f}_{timestamp}.wav"
                else:
                    s3_key = f"audio_recordings/{channel_id}_{tone_a_hz:.1f}_{tone_b_hz:.1f}_{timestamp}.wav"
                
                print(f"[RECORDING] Uploading to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
                s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
                print(f"[RECORDING] ✓ Successfully uploaded to S3: {s3_key}")
                
                try:
                    os.remove(file_path)
                    print(f"[RECORDING] Removed local file: {file_path}")
                except Exception as e:
                    print(f"[RECORDING] WARNING: Failed to remove local file: {e}")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                print(f"[RECORDING] ERROR: S3 upload failed ({error_code}): {e}")
                print(f"[RECORDING] Keeping local file: {file_path}")
            except BotoCoreError as e:
                print(f"[RECORDING] ERROR: S3 client error: {e}")
                print(f"[RECORDING] Keeping local file: {file_path}")
            except Exception as e:
                print(f"[RECORDING] ERROR: Unexpected error during S3 upload: {e}")
                print(f"[RECORDING] Keeping local file: {file_path}")
        
        upload_thread_obj = threading.Thread(target=upload_thread, daemon=True)
        upload_thread_obj.start()
    
    def cleanup_expired_sessions(self):
        with self.mutex:
            current_time_ms = int(time.time() * 1000)
            expired = [ch_id for ch_id, session in self.active_sessions.items() 
                      if current_time_ms >= session.end_time_ms]
            for ch_id in expired:
                self._stop_session(ch_id)
    
    def stop_recording(self, channel_id: str):
        with self.mutex:
            self._stop_session(channel_id)


global_recording_manager = RecordingManager()

