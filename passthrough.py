import threading
import time
import queue
from typing import Dict, Optional, Any
from collections import deque
import numpy as np

try:
    from audio_devices import (
        select_output_device_for_channel,
        open_output_stream,
        close_stream,
    )

    HAS_AUDIO_DEVICES = True
except ImportError:
    HAS_AUDIO_DEVICES = False


class PassthroughSession:
    def __init__(
        self, source_channel_id: str, target_channel_id: str, duration_ms: int
    ):
        self.source_channel_id = source_channel_id
        self.target_channel_id = target_channel_id
        self.duration_ms = duration_ms
        self.start_time_ms = int(time.time() * 1000)
        self.end_time_ms = self.start_time_ms + duration_ms
        self.target_channel_index: Optional[int] = None
        self.output_stream = None
        self.pa = None
        self.first_chunk_logged = False  # Track if we've logged the first chunk message


class PassthroughManager:
    def __init__(self):
        self.mutex = threading.Lock()
        self.active_sessions: Dict[str, PassthroughSession] = {}
        self.channel_id_to_index: Dict[str, int] = {}
        self.audio_queues: Dict[str, queue.Queue] = {}  # Use thread-safe queue
        self.write_threads: Dict[str, threading.Thread] = {}
        self.write_threads_running: Dict[str, bool] = {}

    def set_channel_mapping(self, channel_ids: list):
        with self.mutex:
            self.channel_id_to_index = {
                ch_id: idx for idx, ch_id in enumerate(channel_ids)
            }

    def _get_target_channel_index(self, target_channel_id: str) -> Optional[int]:
        if target_channel_id in self.channel_id_to_index:
            return self.channel_id_to_index[target_channel_id]

        channel_name_to_index = {
            "channel_one": 0,
            "channel_two": 1,
            "channel_three": 2,
            "channel_four": 3,
        }

        if target_channel_id in channel_name_to_index:
            return channel_name_to_index[target_channel_id]

        return None

    def start_passthrough(
        self, source_channel_id: str, target_channel_id: str, duration_ms: int
    ) -> bool:
        if duration_ms <= 0:
            return False

        with self.mutex:
            current_time_ms = int(time.time() * 1000)

            # Check if session already exists
            existing_session = None
            if source_channel_id in self.active_sessions:
                existing_session = self.active_sessions[source_channel_id]
                if existing_session.end_time_ms > current_time_ms:
                    remaining_time_ms = existing_session.end_time_ms - current_time_ms
                    new_duration_ms = max(remaining_time_ms, duration_ms)
                    existing_session.end_time_ms = current_time_ms + new_duration_ms
                    existing_session.duration_ms = new_duration_ms
                    
                    # CRITICAL FIX: Ensure stream still exists and is active
                    stream_valid = False
                    if existing_session.output_stream and existing_session.pa:
                        try:
                            if existing_session.output_stream.is_active():
                                stream_valid = True
                            else:
                                # Stream exists but not active - try to restart it
                                try:
                                    existing_session.output_stream.start_stream()
                                    stream_valid = True
                                    print(
                                        f"[PASSTHROUGH] Restarted inactive stream for {source_channel_id}"
                                    )
                                except Exception as e:
                                    print(
                                        f"[PASSTHROUGH] WARNING: Failed to restart stream: {e}"
                                    )
                        except Exception as e:
                            print(
                                f"[PASSTHROUGH] WARNING: Error checking stream status: {e}"
                            )
                    
                    # If stream is missing or invalid, recreate it
                    if not stream_valid:
                        print(
                            f"[PASSTHROUGH] WARNING: Stream missing/invalid for existing session, recreating..."
                        )
                        # Close old stream if it exists
                        if existing_session.output_stream and existing_session.pa:
                            try:
                                close_stream(existing_session.pa, existing_session.output_stream)
                            except Exception:
                                pass
                        existing_session.output_stream = None
                        existing_session.pa = None
                        # Continue below to recreate the stream for this existing session
                    else:
                        # Stream is valid, just extend duration
                        print(
                            f"[PASSTHROUGH] Session already active for {source_channel_id}, "
                            f"remaining={remaining_time_ms} ms, new={duration_ms} ms, "
                            f"using longer={new_duration_ms} ms"
                        )
                        return True
                else:
                    self._stop_session(source_channel_id)
                    existing_session = None

            target_index = self._get_target_channel_index(target_channel_id)
            if target_index is None:
                print(
                    f"[PASSTHROUGH] ERROR: Cannot find target channel index for '{target_channel_id}'"
                )
                print(
                    f"[PASSTHROUGH] Available channel mappings: {list(self.channel_id_to_index.keys())}"
                )
                print(
                    f"[PASSTHROUGH] Supported channel names: channel_one, channel_two, channel_three, channel_four"
                )
                return False

            # Use existing session if available, otherwise create new one
            if existing_session:
                session = existing_session
                # Update duration (already done above, but ensure it's set)
                session.end_time_ms = current_time_ms + duration_ms
                session.duration_ms = duration_ms
            else:
                session = PassthroughSession(
                    source_channel_id, target_channel_id, duration_ms
                )
                session.target_channel_index = target_index
                self.active_sessions[source_channel_id] = session
            
            # Ensure target_index is set
            session.target_channel_index = target_index

            # Only create/recreate stream if it doesn't exist or is invalid
            if not session.output_stream or not session.pa:
                if HAS_AUDIO_DEVICES:
                    print(
                        f"[PASSTHROUGH] Creating output stream for target channel {target_channel_id} (index {target_index})"
                    )
                    device_index = select_output_device_for_channel(target_index)
                    if device_index is not None:
                        print(
                            f"[PASSTHROUGH] Selected device {device_index} for channel index {target_index}"
                        )
                        # Use 1920 frames to match incoming chunk size exactly
                        # This eliminates fragmentation and ensures smooth writes
                        pa, stream = open_output_stream(
                            device_index, frames_per_buffer=1920
                        )
                        if pa and stream:
                            try:
                                # Pre-fill buffer with silence to prevent initial choppiness
                                # Write 2 chunks of silence (1920 samples * 2 bytes * 2 = 7680 bytes)
                                silence_chunk = bytes(1920 * 2 * 2)  # 2 chunks of silence
                                try:
                                    stream.write(silence_chunk, exception_on_underflow=False)
                                except Exception:
                                    pass  # Ignore errors during pre-fill
                                
                                stream.start_stream()
                                session.pa = pa
                                session.output_stream = stream
                                print(
                                    f"[PASSTHROUGH] ✓ Opened and started output stream for target channel {target_channel_id} (index {target_index}, device {device_index})"
                                )
                            except Exception as e:
                                print(
                                    f"[PASSTHROUGH] ERROR: Failed to start output stream: {e}"
                                )
                                import traceback
                                traceback.print_exc()
                                close_stream(pa, stream)
                                # Don't create session if stream can't start
                                print(
                                    f"[PASSTHROUGH] Aborting passthrough start - output stream failed"
                                )
                                # Clean up session properly
                                self._stop_session(source_channel_id)
                                return False
                        else:
                            print(
                                f"[PASSTHROUGH] ERROR: open_output_stream returned None (pa={pa}, stream={stream})"
                            )
                            print(
                                f"[PASSTHROUGH] Aborting passthrough start - stream creation failed"
                            )
                            # Clean up session properly
                            self._stop_session(source_channel_id)
                            return False
                    else:
                        print(
                            f"[PASSTHROUGH] ERROR: No audio device available for channel index {target_index} (target: {target_channel_id})"
                        )
                        print(
                            f"[PASSTHROUGH] Aborting passthrough start - no audio device"
                        )
                        # Clean up session properly
                        self._stop_session(source_channel_id)
                        return False
                else:
                    print(
                        f"[PASSTHROUGH] ERROR: Audio devices module not available (HAS_AUDIO_DEVICES=False)"
                    )
                    print(
                        f"[PASSTHROUGH] Aborting passthrough start - audio devices unavailable"
                    )
                    # Clean up session properly
                    self._stop_session(source_channel_id)
                    return False
            else:
                print(
                    f"[PASSTHROUGH] Stream already exists for {source_channel_id}, reusing existing stream"
                )

            # Session already added to active_sessions above if new, or already exists if existing
            if source_channel_id not in self.active_sessions:
                self.active_sessions[source_channel_id] = session

            if source_channel_id not in self.audio_queues:
                # Use thread-safe queue with larger size to prevent dropping chunks
                # 50 chunks = ~2 seconds of audio at 1920 samples/chunk, 48kHz
                self.audio_queues[source_channel_id] = queue.Queue(maxsize=50)
                print(f"[PASSTHROUGH] Created audio queue for {source_channel_id}")

            # Ensure write thread is running
            if source_channel_id not in self.write_threads_running:
                self.write_threads_running[source_channel_id] = True
                thread = threading.Thread(
                    target=self._write_worker, args=(source_channel_id,), daemon=True
                )
                self.write_threads[source_channel_id] = thread
                thread.start()
                print(
                    f"[PASSTHROUGH] Started write thread for channel {source_channel_id}"
                )
            else:
                # Check if thread is still alive
                existing_thread = self.write_threads.get(source_channel_id)
                if existing_thread and not existing_thread.is_alive():
                    print(
                        f"[PASSTHROUGH] WARNING: Write thread died for {source_channel_id}, restarting..."
                    )
                    self.write_threads_running[source_channel_id] = True
                    thread = threading.Thread(
                        target=self._write_worker, args=(source_channel_id,), daemon=True
                    )
                    self.write_threads[source_channel_id] = thread
                    thread.start()
                    print(
                        f"[PASSTHROUGH] Restarted write thread for channel {source_channel_id}"
                    )

            print(
                f"[PASSTHROUGH] Started: {source_channel_id} -> {target_channel_id}, duration={duration_ms} ms"
            )
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
        # Only drop chunks if queue is critically full (above 40 chunks)
        # This gives more headroom and reduces audio dropouts
        max_queue_size = 40
        write_count = 0
        print(f"[PASSTHROUGH] Write worker started for channel {source_channel_id}")

        SAMPLE_RATE = 48000
        samples_per_write = 1920  # Match frames_per_buffer and incoming chunk size
        bytes_per_sample = 2  # int16 = 2 bytes
        bytes_per_write = samples_per_write * bytes_per_sample  # 3840 bytes
        
        # CRITICAL: Calculate time per chunk to match hardware playback rate
        # At 48kHz, 1920 samples = 40ms of audio
        # We must write at exactly this rate to prevent choppiness
        TIME_PER_CHUNK = samples_per_write / SAMPLE_RATE  # 0.04 seconds = 40ms
        last_write_time = time.time()
        
        # Incoming chunks are 1920 samples, write them directly without splitting
        # This matches the buffer size and eliminates fragmentation

        while self.write_threads_running.get(source_channel_id, False):
            try:
                # Get session and queue with minimal mutex hold time
                session = None
                queue = None
                with self.mutex:
                    if source_channel_id not in self.active_sessions:
                        break
                    session = self.active_sessions[source_channel_id]
                    queue = self.audio_queues.get(source_channel_id)

                if not session or not session.output_stream or not session.pa:
                    time.sleep(0.001)  # Reduced sleep to check more frequently
                    continue

                if not queue:
                    time.sleep(0.001)  # Reduced sleep to check more frequently
                    continue

                # Check queue length and drop if needed (thread-safe queue)
                queue_len = queue.qsize()
                if queue_len > max_queue_size:
                    dropped = 0
                    while queue.qsize() > max_queue_size:
                        try:
                            queue.get_nowait()
                            dropped += 1
                        except queue.Empty:
                            break
                    if dropped > 0 and write_count % 100 == 0:
                        print(
                            f"[PASSTHROUGH] Dropped {dropped} audio chunks from queue for {source_channel_id}"
                        )

                # CRITICAL FIX: Write only ONE chunk per iteration to maintain smooth playback
                # Writing multiple chunks in a loop causes bursts that create choppy "phone ringing" audio
                try:
                    pcm_bytes = queue.get_nowait()
                    
                    if not session.output_stream.is_active():
                        try:
                            session.output_stream.start_stream()
                            print(
                                f"[PASSTHROUGH] Started output stream for {source_channel_id}"
                            )
                        except Exception as e:
                            if write_count % 100 == 0:
                                print(
                                    f"[PASSTHROUGH] ERROR: Failed to start stream: {e}"
                                )
                            time.sleep(0.01)
                            continue

                    # Write entire chunk (1920 samples = 3840 bytes) in one go
                    # This matches the buffer size exactly and eliminates fragmentation
                    pcm_len = len(pcm_bytes)
                    
                    # Ensure we have a complete chunk (1920 samples = 3840 bytes)
                    if pcm_len != bytes_per_write:
                        # If chunk size doesn't match, log warning but still try to write
                        if write_count % 100 == 0:
                            print(
                                f"[PASSTHROUGH] WARNING: Unexpected chunk size {pcm_len} bytes (expected {bytes_per_write}) for {source_channel_id}"
                            )
                    
                    try:
                        # CRITICAL: Rate-limit writes to match hardware playback rate
                        # At 48kHz, 1920 samples = 40ms - we must write at this exact rate
                        current_time = time.time()
                        elapsed = current_time - last_write_time
                        
                        # If we're ahead of schedule, wait to match hardware rate
                        # This prevents writing chunks too quickly and causing choppy audio
                        if elapsed < TIME_PER_CHUNK:
                            sleep_time = TIME_PER_CHUNK - elapsed
                            if sleep_time > 0.0001:  # Only sleep if > 0.1ms
                                time.sleep(sleep_time)
                                current_time = time.time()  # Update time after sleep
                        # If elapsed >= TIME_PER_CHUNK, we're at or behind schedule
                        # Write immediately to catch up
                        
                        # Write the entire chunk at once for optimal timing
                        session.output_stream.write(
                            pcm_bytes, exception_on_underflow=False
                        )
                        write_count += 1
                        last_write_time = current_time
                        
                        if write_count <= 10 or write_count % 500 == 0:
                            print(
                                f"[PASSTHROUGH] Wrote audio chunk #{write_count} for {source_channel_id} ({pcm_len} bytes, queue={queue.qsize()}, elapsed={elapsed*1000:.2f}ms)"
                            )
                    except Exception as e:
                        # Underflow or other write error - log and continue
                        if write_count % 50 == 0:  # Log more frequently for errors
                            print(
                                f"[PASSTHROUGH] ERROR: Failed to write audio: {e}"
                            )
                        # Update time even on error to maintain timing
                        last_write_time = time.time()
                        # Sleep briefly before retrying to avoid tight error loop
                        time.sleep(0.001)
                except queue.Empty:
                    # No data available - sleep briefly to avoid busy-waiting
                    # But don't sleep too long or we'll miss timing
                    time.sleep(0.001)  # 1ms sleep when no data
                    # Don't update last_write_time here - maintain timing based on last actual write
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
                    # Log only first few times to avoid spam
                    if not hasattr(self, '_route_audio_miss_log_count'):
                        self._route_audio_miss_log_count = {}
                    count = self._route_audio_miss_log_count.get(source_channel_id, 0)
                    if count < 3:
                        print(f"[PASSTHROUGH DEBUG] route_audio called for {source_channel_id} but no active session (active_sessions={list(self.active_sessions.keys())})")
                        self._route_audio_miss_log_count[source_channel_id] = count + 1
                    return False

                session = self.active_sessions[source_channel_id]
                current_time_ms = int(time.time() * 1000)

                if current_time_ms >= session.end_time_ms:
                    self._stop_session(source_channel_id)
                    return False

                # Don't queue audio if there's no valid output stream
                if not session.output_stream or not session.pa:
                    # Session exists but no valid stream - stop the session
                    self._stop_session(source_channel_id)
                    return False

                if source_channel_id not in self.audio_queues:
                    # Use thread-safe queue with larger size to prevent dropping chunks
                    # 50 chunks = ~2 seconds of audio at 1920 samples/chunk, 48kHz
                    self.audio_queues[source_channel_id] = queue.Queue(maxsize=50)
                    print(
                        f"[PASSTHROUGH] Created audio queue for {source_channel_id} in route_audio"
                    )

                queue = self.audio_queues[source_channel_id]

            if queue is not None:
                try:
                    pcm = (np.clip(audio_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
                    pcm_bytes = pcm.tobytes()
                    queue_len_before = queue.qsize()
                    # Use non-blocking put - drop if queue is full to prevent blocking
                    try:
                        queue.put_nowait(pcm_bytes)
                        queue_len_after = queue.qsize()
                    except queue.Full:
                        # Queue is full, drop this chunk to prevent blocking
                        queue_len_after = queue.qsize()
                        if not hasattr(self, '_queue_full_drops'):
                            self._queue_full_drops = {}
                        drops = self._queue_full_drops.get(source_channel_id, 0) + 1
                        self._queue_full_drops[source_channel_id] = drops
                        if drops % 100 == 0:
                            print(f"[PASSTHROUGH] WARNING: Queue full, dropped chunk for {source_channel_id} (total drops: {drops})")
                        return True  # Return True even though we dropped it
                    # Log first chunk only once per session
                    if not session.first_chunk_logged:
                        print(f"[PASSTHROUGH] First audio chunk queued for {source_channel_id} ({len(pcm_bytes)} bytes)")
                        session.first_chunk_logged = True
                    # Log queue filling warning only occasionally to reduce spam
                    elif queue_len_after >= 40:  # Queue nearly full (maxsize=50)
                        # Use a simple counter to only log every 100th time
                        if not hasattr(self, '_queue_warning_counts'):
                            self._queue_warning_counts = {}
                        count = self._queue_warning_counts.get(source_channel_id, 0)
                        if count % 100 == 0:
                            print(f"[PASSTHROUGH] WARNING: Queue filling up for {source_channel_id} ({queue_len_after}/50 chunks)")
                        self._queue_warning_counts[source_channel_id] = count + 1
                    return True
                except Exception as e:
                    print(f"[PASSTHROUGH] ERROR: Failed to queue audio: {e}")
                    import traceback
                    traceback.print_exc()
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
            # Drain the queue before deleting
            q = self.audio_queues[source_channel_id]
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
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
            expired = [
                ch_id
                for ch_id, session in self.active_sessions.items()
                if current_time_ms >= session.end_time_ms
            ]
            for ch_id in expired:
                self._stop_session(ch_id)


global_passthrough_manager = PassthroughManager()
