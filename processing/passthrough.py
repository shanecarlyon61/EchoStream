import threading
import time
from typing import Dict, Optional, Any
from collections import deque
import numpy as np

try:
    from audio.devices import select_output_device_for_channel, open_output_stream, close_stream
    HAS_AUDIO_DEVICES = True
except ImportError:
    HAS_AUDIO_DEVICES = False


class PassthroughSession:
    def __init__(self, source_channel_id: str, target_channel_id: str, duration_ms: int):
        self.source_channel_id = source_channel_id
        self.target_channel_id = target_channel_id
        self.duration_ms = duration_ms
        self.start_time_ms = int(time.time() * 1000)
        self.end_time_ms = self.start_time_ms + duration_ms
        self.target_channel_index: Optional[int] = None
        self.output_stream = None
        self.pa = None


class PassthroughManager:
    def __init__(self):
        self.mutex = threading.Lock()
        self.active_sessions: Dict[str, PassthroughSession] = {}
        self.channel_id_to_index: Dict[str, int] = {}
        self.audio_queues: Dict[str, deque] = {}
        self.write_threads: Dict[str, threading.Thread] = {}
        self.write_threads_running: Dict[str, bool] = {}
    
    def set_channel_mapping(self, channel_ids: list):
        with self.mutex:
            self.channel_id_to_index = {ch_id: idx for idx, ch_id in enumerate(channel_ids)}
    
    def _get_target_channel_index(self, target_channel_id: str) -> Optional[int]:
        if target_channel_id in self.channel_id_to_index:
            return self.channel_id_to_index[target_channel_id]
        
        channel_name_to_index = {
            "channel_one": 0,
            "channel_two": 1,
            "channel_three": 2,
            "channel_four": 3
        }
        
        if target_channel_id in channel_name_to_index:
            return channel_name_to_index[target_channel_id]
        
        return None
    
    def start_passthrough(self, source_channel_id: str, target_channel_id: str, duration_ms: int) -> bool:
        if duration_ms <= 0:
            return False
        
        with self.mutex:
            current_time_ms = int(time.time() * 1000)
            
            if source_channel_id in self.active_sessions:
                existing = self.active_sessions[source_channel_id]
                if existing.end_time_ms > current_time_ms:
                    remaining_time_ms = existing.end_time_ms - current_time_ms
                    new_duration_ms = max(remaining_time_ms, duration_ms)
                    existing.end_time_ms = current_time_ms + new_duration_ms
                    existing.duration_ms = new_duration_ms
                    print(f"[PASSTHROUGH] Session already active for {source_channel_id}, "
                          f"remaining={remaining_time_ms} ms, new={duration_ms} ms, "
                          f"using longer={new_duration_ms} ms")
                    return True
                else:
                    self._stop_session(source_channel_id)
            
            target_index = self._get_target_channel_index(target_channel_id)
            if target_index is None:
                print(f"[PASSTHROUGH] ERROR: Cannot find target channel index for '{target_channel_id}'")
                print(f"[PASSTHROUGH] Available channel mappings: {list(self.channel_id_to_index.keys())}")
                print(f"[PASSTHROUGH] Supported channel names: channel_one, channel_two, channel_three, channel_four")
                return False
            
            session = PassthroughSession(source_channel_id, target_channel_id, duration_ms)
            session.target_channel_index = target_index
            
            if HAS_AUDIO_DEVICES:
                device_index = select_output_device_for_channel(target_index)
                if device_index is not None:
                    pa, stream = open_output_stream(device_index, frames_per_buffer=1024)
                    if pa and stream:
                        try:
                            stream.start_stream()
                            session.pa = pa
                            session.output_stream = stream
                            print(f"[PASSTHROUGH] Opened output stream for target channel {target_channel_id} (index {target_index}, device {device_index})")
                        except Exception as e:
                            print(f"[PASSTHROUGH] WARNING: Failed to start output stream: {e}")
                            close_stream(pa, stream)
                else:
                    print(f"[PASSTHROUGH] WARNING: No audio device available for channel index {target_index} (target: {target_channel_id})")
            
            self.active_sessions[source_channel_id] = session
            
            if source_channel_id not in self.audio_queues:
                self.audio_queues[source_channel_id] = deque(maxlen=10)
                print(f"[PASSTHROUGH] Created audio queue for {source_channel_id}")
            
            if source_channel_id not in self.write_threads_running:
                self.write_threads_running[source_channel_id] = True
                thread = threading.Thread(
                    target=self._write_worker,
                    args=(source_channel_id,),
                    daemon=True
                )
                self.write_threads[source_channel_id] = thread
                thread.start()
                print(f"[PASSTHROUGH] Started write thread for channel {source_channel_id}")
            
            print(f"[PASSTHROUGH] Started: {source_channel_id} -> {target_channel_id}, duration={duration_ms} ms")
            return True
    
    def is_active(self, source_channel_id: str) -> bool:
        with self.mutex:
            if source_channel_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[source_channel_id]
            current_time_ms = int(time.time() * 1000)
            
            if current_time_ms >= session.end_time_ms:
                self._stop_session(source_channel_id)
                return False
            
            return True
    
    def _write_worker(self, source_channel_id: str):
        max_queue_size = 5
        write_count = 0
        print(f"[PASSTHROUGH] Write worker started for channel {source_channel_id}")
        
        while self.write_threads_running.get(source_channel_id, False):
            try:
                session = None
                queue = None
                with self.mutex:
                    if source_channel_id not in self.active_sessions:
                        break
                    session = self.active_sessions[source_channel_id]
                    queue = self.audio_queues.get(source_channel_id)
                
                if not session or not session.output_stream or not session.pa:
                    time.sleep(0.01)
                    continue
                
                if not queue:
                    time.sleep(0.01)
                    continue
                
                if len(queue) > max_queue_size:
                    dropped = 0
                    while len(queue) > max_queue_size:
                        try:
                            queue.popleft()
                            dropped += 1
                        except IndexError:
                            break
                    if dropped > 0 and write_count % 100 == 0:
                        print(f"[PASSTHROUGH] Dropped {dropped} audio chunks from queue for {source_channel_id}")
                
                if queue:
                    try:
                        pcm_bytes = queue.popleft()
                        if not session.output_stream.is_active():
                            try:
                                session.output_stream.start_stream()
                                print(f"[PASSTHROUGH] Started output stream for {source_channel_id}")
                            except Exception as e:
                                if write_count % 100 == 0:
                                    print(f"[PASSTHROUGH] ERROR: Failed to start stream: {e}")
                                continue
                        
                        try:
                            session.output_stream.write(pcm_bytes, exception_on_underflow=False)
                            write_count += 1
                            if write_count <= 5 or write_count % 500 == 0:
                                print(f"[PASSTHROUGH] Wrote audio chunk #{write_count} for {source_channel_id} ({len(pcm_bytes)} bytes)")
                        except Exception as e:
                            if write_count % 100 == 0:
                                print(f"[PASSTHROUGH] ERROR: Failed to write audio: {e}")
                    except IndexError:
                        pass
                
                time.sleep(0.001)
            except Exception as e:
                if write_count % 100 == 0:
                    print(f"[PASSTHROUGH] ERROR in write worker: {e}")
                time.sleep(0.01)
        
        print(f"[PASSTHROUGH] Write worker stopped for channel {source_channel_id}")
        with self.mutex:
            if source_channel_id in self.write_threads_running:
                del self.write_threads_running[source_channel_id]
    
    def route_audio(self, source_channel_id: str, audio_samples: np.ndarray) -> bool:
        try:
            with self.mutex:
                if source_channel_id not in self.active_sessions:
                    return False
                
                session = self.active_sessions[source_channel_id]
                current_time_ms = int(time.time() * 1000)
                
                if current_time_ms >= session.end_time_ms:
                    self._stop_session(source_channel_id)
                    return False
                
                if source_channel_id not in self.audio_queues:
                    self.audio_queues[source_channel_id] = deque(maxlen=10)
                    print(f"[PASSTHROUGH] Created audio queue for {source_channel_id} in route_audio")
                
                queue = self.audio_queues[source_channel_id]
            
            if queue is not None:
                try:
                    pcm = (np.clip(audio_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
                    pcm_bytes = pcm.tobytes()
                    queue.append(pcm_bytes)
                    return True
                except Exception as e:
                    print(f"[PASSTHROUGH] ERROR: Failed to queue audio: {e}")
            else:
                print(f"[PASSTHROUGH] WARNING: Queue is None for {source_channel_id}")
            
            return False
        except Exception as e:
            print(f"[PASSTHROUGH] ERROR: Exception in route_audio: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _stop_session(self, source_channel_id: str):
        if source_channel_id in self.write_threads_running:
            self.write_threads_running[source_channel_id] = False
        
        if source_channel_id in self.write_threads:
            thread = self.write_threads[source_channel_id]
            if thread.is_alive():
                thread.join(timeout=0.1)
            del self.write_threads[source_channel_id]
        
        if source_channel_id in self.audio_queues:
            self.audio_queues[source_channel_id].clear()
            del self.audio_queues[source_channel_id]
        
        if source_channel_id in self.active_sessions:
            session = self.active_sessions[source_channel_id]
            if session.output_stream and session.pa:
                try:
                    close_stream(session.pa, session.output_stream)
                except Exception:
                    pass
            del self.active_sessions[source_channel_id]
            print(f"[PASSTHROUGH] Stopped session for {source_channel_id}")
    
    def stop_passthrough(self, source_channel_id: str):
        with self.mutex:
            self._stop_session(source_channel_id)
    
    def cleanup_expired_sessions(self):
        with self.mutex:
            current_time_ms = int(time.time() * 1000)
            expired = [ch_id for ch_id, session in self.active_sessions.items() 
                      if current_time_ms >= session.end_time_ms]
            for ch_id in expired:
                self._stop_session(ch_id)


global_passthrough_manager = PassthroughManager()

