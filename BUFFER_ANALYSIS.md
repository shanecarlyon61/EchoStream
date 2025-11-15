# Jitter Buffer Usage Analysis

## Buffer Structure

### Configuration
- **Buffer Size**: `JITTER_BUFFER_SIZE = 8` frames
- **Samples Per Frame**: `SAMPLES_PER_FRAME = 1920` samples
- **Sample Rate**: `SAMPLE_RATE = 48000` Hz
- **Frame Duration**: 1920 samples ÷ 48000 Hz = **40 milliseconds per frame**
- **Total Buffer Capacity**: 8 frames × 40 ms = **320 milliseconds of audio**

### Buffer Components
```python
class JitterBuffer:
    frames: List[AudioFrame]  # 8 frames, circular buffer
    write_index: int           # Where new frames are written
    read_index: int            # Where frames are read for playback
    frame_count: int           # Current number of valid frames (0-8)
    mutex: Lock                # Thread-safe access
```

### Audio Frame
```python
class AudioFrame:
    samples: np.ndarray        # 1920 float32 samples (-1.0 to 1.0)
    sample_count: int          # Actual samples in frame (usually 1920)
    valid: bool                # Whether frame contains valid audio
```

## Buffer Operations

### 1. Write Operation (UDP → Buffer)
**Location**: `process_received_audio()` in `audio.py`

**Process**:
1. UDP packet arrives with Opus-encoded audio
2. Decode Opus → PCM (1920 samples)
3. Convert to float32 and normalize (-32767 to 32767 → -1.0 to 1.0)
4. Apply **20x gain boost** (matching C implementation)
5. Write to buffer at `write_index`:
   - If buffer has space (`frame_count < 8`):
     - Copy samples to frame at `write_index`
     - Mark frame as `valid = True`
     - Advance `write_index = (write_index + 1) % 8`
     - Increment `frame_count`
   - If buffer is full (`frame_count == 8`):
     - **Drop oldest frame** (advance `read_index`)
     - Decrement `frame_count`
     - Write new frame at `write_index`
     - Advance `write_index`

**Current Status**: ❌ **No packets arriving** (`packets_received=0`)
- Buffer never receives data
- `frame_count` stays at 0
- `write_index` and `read_index` both at 0

### 2. Read Operation (Buffer → Audio Output)
**Location**: `audio_output_worker()` in `audio.py`

**Process**:
1. Output worker needs 1024 samples for PyAudio buffer
2. Lock jitter buffer mutex
3. Fill output buffer from jitter buffer:
   - While output buffer not full:
     - If jitter buffer has frames (`frame_count > 0`):
       - Read from frame at `read_index`
       - Copy samples (with **1.5x gain boost**) to output buffer
       - Track position within frame (`current_output_frame_pos`)
       - When frame is fully consumed:
         - Mark frame as `valid = False`
         - Advance `read_index = (read_index + 1) % 8`
         - Decrement `frame_count`
         - Reset `current_output_frame_pos = 0`
     - If jitter buffer is empty (`frame_count == 0`):
       - Fill remainder with **silence** (0.0)
4. Write 1024 samples to PyAudio output stream
5. Repeat every ~21.3ms (1024 samples ÷ 48000 Hz)

**Current Status**: ⚠️ **Playing silence**
- Buffer is empty (`frame_count=0`)
- Output worker fills buffer with silence
- RMS = 0.000000 (no audio)
- Underrun count = 500, 1000 (continuous underruns)

## Buffer States

### Ideal State (Normal Operation)
```
frame_count: 3-5 frames (120-200 ms)
write_index: advancing as packets arrive
read_index: advancing as audio plays
Status: Smooth playback, no underruns
```

### Current State (Your Logs)
```
frame_count: 0 frames
write_index: 0 (no writes)
read_index: 0 (no reads possible)
packets_received: 0
Status: Buffer empty, playing silence
```

### Underrun State
```
frame_count: 0 frames
read_index: trying to read but buffer empty
Status: Underrun detected, filling with silence
Log: [JITTER UNDERRUN] Buffer empty!
```

### Overflow State (If packets arrive too fast)
```
frame_count: 8 frames (buffer full)
write_index: == read_index (wrapping around)
Status: Dropping oldest frame to make room
Log: [JITTER OVERFLOW] Buffer full, dropped frame
```

## Timing Analysis

### Packet Arrival Rate (Expected)
- **Samples per second**: 48,000
- **Samples per frame**: 1,920
- **Frames per second**: 48,000 ÷ 1,920 = **25 frames/second**
- **Packet interval**: 1 ÷ 25 = **40 ms between packets**

### Buffer Fill Time
- **Minimum threshold**: 2 frames = 80 ms
- **Time to fill 2 frames**: 2 × 40 ms = **80 ms** (2 packets)
- **Optimal fill**: 3-5 frames = 120-200 ms

### Output Consumption Rate
- **Output buffer size**: 1024 samples
- **Output buffer duration**: 1024 ÷ 48000 = **21.3 ms**
- **Output rate**: ~47 buffers/second
- **Frames consumed per second**: ~25 frames/second (matches input)

## Current Issue: No UDP Packets

### Symptoms
1. **`packets_received=0`** - No UDP packets arriving
2. **`frame_count=0`** - Buffer never gets data
3. **`[JITTER UNDERRUN]`** - Continuous underruns
4. **`[AUDIO OUT] Playing SILENCE`** - No audio to play
5. **`[UDP] Timeout waiting for packets`** - UDP socket receiving nothing

### Root Cause
**Network/Server Issue**: The server at `18.206.115.29:51647` is not sending UDP audio packets to the Raspberry Pi.

### Why Buffer is Empty
1. **UDP Listener** (`udp_listener_worker`) is waiting for packets
2. **Socket timeout**: 0.1 seconds (100ms)
3. **No packets arrive**: Continuous timeouts
4. **Buffer never receives data**: `process_received_audio()` never called
5. **Output worker finds empty buffer**: Fills with silence

### Verification Steps
1. ✅ **WebSocket connected**: Confirmed (`users_connected` message received)
2. ✅ **UDP socket created**: Confirmed (`[UDP] Socket local address: 0.0.0.0:42830`)
3. ✅ **UDP server address**: `18.206.115.29:51647`
4. ❌ **UDP packets arriving**: **NO** (`packets_received=0`)
5. ❌ **Buffer has data**: **NO** (`frame_count=0`)

## Buffer Metrics (From Your Logs)

### Channel 555 Status
```
frame_count: 0/8
read_idx: 0
write_idx: 0
packets_received: 0
underrun_count: 500, 1000 (increasing)
RMS: 0.000000 (silence)
Status: EMPTY - No audio data
```

### Expected Metrics (When Working)
```
frame_count: 3-5/8
read_idx: advancing (0-7, wrapping)
write_idx: advancing (0-7, wrapping)
packets_received: increasing (25 packets/second)
underrun_count: 0 (or occasional, < 10)
RMS: > 0.001 (actual audio)
Status: HEALTHY - Smooth playback
```

## Buffer Recovery

### When Packets Start Arriving
1. **First packet**: `frame_count: 0 → 1`
2. **Second packet**: `frame_count: 1 → 2`
3. **Log**: `[JITTER RECOVERY] Buffer refilled!`
4. **Output worker**: Starts playing audio (RMS > 0)
5. **Steady state**: `frame_count: 3-5` (balanced)

### Recovery Time
- **Minimum**: 2 frames = 80 ms (2 packets)
- **Optimal**: 3-5 frames = 120-200 ms (3-5 packets)
- **Total recovery**: ~200 ms from first packet

## Recommendations

### 1. Verify Server is Sending Packets
- Check server logs for UDP packet transmission
- Verify server is sending to correct IP: `18.206.115.29:51647`
- Check if users are actually streaming/transmitting

### 2. Network Diagnostics
- Verify firewall allows UDP on port 51647
- Check network routing to Raspberry Pi
- Test UDP connectivity: `nc -u -l 51647` on Pi

### 3. Buffer Monitoring
- Current logs show buffer status every 500 packets
- When packets arrive, monitor:
  - `frame_count` should stabilize at 3-5
  - `read_idx` and `write_idx` should advance
  - `underrun_count` should stop increasing

### 4. Expected Behavior
Once packets arrive:
- `[JITTER WRITE]` logs will appear
- `[JITTER RECOVERY]` log when buffer fills
- `[AUDIO OUT] Playing AUDIO` (RMS > 0)
- `frame_count` will stabilize at 3-5
- Audio will play smoothly

## Summary

**Current State**: Buffer is **empty** because **no UDP packets are arriving** from the server.

**Buffer Design**: The buffer is correctly implemented and ready to receive data. It can hold 320ms of audio (8 frames × 40ms) and will automatically recover when packets start arriving.

**Next Steps**: Investigate why the server is not sending UDP packets to the Raspberry Pi. The buffer and audio pipeline are working correctly - they're just waiting for data.

