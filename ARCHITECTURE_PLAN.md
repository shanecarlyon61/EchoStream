# EchoStream - Complete Rebuild Architecture Plan

## System Overview
EchoStream is a multi-channel audio streaming system that:
- Captures audio from USB devices
- Streams audio over UDP using Opus codec
- Controls transmission via GPIO (PTT - Push To Talk)
- Detects DTMF-style tone sequences
- Manages channels via WebSocket signaling
- Supports MQTT notifications and S3 uploads

---

## Architecture: Step-by-Step Module Plan

### **PHASE 1: Foundation & Configuration** 🔧

#### Module 1.1: Constants & Shared Definitions (`echostream.py`)
**Purpose:** Central configuration and shared constants

**Components:**
- System constants (SAMPLE_RATE, MAX_CHANNELS, FFT_SIZE, etc.)
- Global state management (threading.Event for shutdown)
- Channel ID tracking
- Shared data structures

**Functions:**
- Constants definition
- Global state initialization
- Thread-safe shutdown mechanism

**Dependencies:** None (base module)

---

#### Module 1.2: Configuration Manager (`config.py`)
**Purpose:** Load and manage JSON configuration

**Components:**
- Configuration data classes
- JSON file loader
- Configuration validation
- Channel configuration
- Tone detection configuration
- Passthrough settings

**Functions:**
- `load_config(path)` - Load JSON config
- `get_channel_config(index)` - Get channel settings
- `get_tone_detect_config()` - Get tone detection settings
- `validate_config()` - Validate loaded config

**Dependencies:** `echostream.py`

---

### **PHASE 2: Hardware Interface Layer** 🎛️

#### Module 2.1: GPIO Controller (`gpio.py`)
**Purpose:** Manage GPIO pins for PTT control

**Components:**
- GPIO pin mapping (GPIO 20, 21, 23, 24)
- Pin state monitoring
- State change callbacks
- Thread-safe GPIO access

**Functions:**
- `init_gpio()` - Initialize GPIO chip
- `read_pin(pin)` - Read pin state
- `monitor_gpio(callback)` - Monitor pin changes
- `cleanup_gpio()` - Release GPIO resources

**Dependencies:** `lgpio`, `echostream.py`

---

#### Module 2.2: Audio Hardware Manager (`audio_hardware.py`)
**Purpose:** Manage PyAudio/ALSA audio devices

**Components:**
- PortAudio instance management
- USB device detection and mapping
- Device enumeration
- Device assignment to channels

**Functions:**
- `init_portaudio()` - Initialize PyAudio
- `detect_usb_devices()` - Scan for USB audio devices
- `assign_device_to_channel(channel_id, device_index)` - Map device to channel
- `get_device_info(device_index)` - Get device details
- `cleanup_portaudio()` - Terminate PyAudio

**Dependencies:** `pyaudio`, `echostream.py`

---

### **PHASE 3: Audio Processing Core** 🎵

#### Module 3.1: Audio Stream Manager (`audio_stream.py`)
**Purpose:** Manage individual audio channel streams

**Components:**
- AudioStream class (per channel)
- Input/Output stream management
- Opus encoder/decoder
- Jitter buffer
- AES encryption key

**Data Structures:**
```python
class AudioStream:
    - channel_id: str
    - device_index: int
    - input_stream: pyaudio.Stream
    - output_stream: pyaudio.Stream
    - encoder: opuslib.Encoder
    - decoder: opuslib.Decoder
    - jitter_buffer: JitterBuffer
    - encryption_key: bytes
    - gpio_active: bool
    - transmitting: bool
```

**Functions:**
- `create_stream(channel_id, device_index)` - Create new stream
- `start_stream(stream)` - Start audio streams
- `stop_stream(stream)` - Stop audio streams
- `encode_audio(stream, samples)` - Encode to Opus
- `decode_audio(stream, opus_data)` - Decode from Opus
- `cleanup_stream(stream)` - Release stream resources

**Dependencies:** `audio_hardware.py`, `opuslib`, `numpy`

---

#### Module 3.2: Audio Workers (`audio_workers.py`)
**Purpose:** Background threads for audio I/O

**Components:**
- Input worker thread (capture from microphone)
- Output worker thread (play to speaker)
- Shared audio buffer for tone detection
- Thread synchronization

**Functions:**
- `audio_input_worker(stream)` - Capture audio loop
- `audio_output_worker(stream)` - Playback audio loop
- `write_to_shared_buffer(samples)` - Feed tone detection
- `read_from_jitter_buffer(stream)` - Read audio for playback

**Dependencies:** `audio_stream.py`, `threading`

---

### **PHASE 4: Network Communication** 🌐

#### Module 4.1: WebSocket Client (`websocket_client.py`)
**Purpose:** WebSocket signaling and control

**Components:**
- WebSocket connection manager
- Message parsing and routing
- Channel registration
- UDP connection info handler
- User connection tracking

**Functions:**
- `connect_to_server(url)` - Establish WebSocket connection
- `register_channel(channel_id)` - Register channel with server
- `send_transmit_event(channel_id, active)` - Send PTT state
- `parse_udp_config(message)` - Parse UDP connection info
- `handle_users_connected(message)` - Track connected users
- `disconnect()` - Close WebSocket

**Dependencies:** `websocket-client`, `json`, `echostream.py`

---

#### Module 4.2: UDP Manager (`udp_manager.py`)
**Purpose:** UDP audio packet transmission and reception

**Components:**
- UDP socket management
- Audio packet sender
- Audio packet receiver
- Heartbeat mechanism
- Packet encryption/decryption

**Functions:**
- `setup_udp(host, port)` - Create UDP socket
- `send_audio_packet(channel_id, opus_data)` - Send audio
- `receive_audio_packet()` - Receive audio
- `send_heartbeat()` - Send keepalive
- `encrypt_packet(data, key)` - Encrypt audio packet
- `decrypt_packet(encrypted_data, key)` - Decrypt audio packet
- `cleanup_udp()` - Close socket

**Dependencies:** `socket`, `crypto.py`, `threading`

---

#### Module 4.3: Crypto Module (`crypto.py`)
**Purpose:** Encryption/decryption utilities

**Functions:**
- `decode_base64(key_str)` - Decode base64 key
- `encrypt_aes(data, key)` - AES encryption
- `decrypt_aes(encrypted_data, key)` - AES decryption

**Dependencies:** `base64`, `cryptography` or `Crypto`

---

### **PHASE 5: Tone Detection System** 🔔

#### Module 5.1: Tone Detection Engine (`tone_detection.py`)
**Purpose:** Detect DTMF-style tone sequences

**Components:**
- Tone definition storage
- FFT-based frequency analysis
- Tone sequence matching
- Duration validation

**Data Structures:**
```python
class ToneDefinition:
    - tone_id: str
    - tone_a_freq: float
    - tone_b_freq: float
    - tone_a_length_ms: int
    - tone_b_length_ms: int
    - tone_a_range_hz: int
    - tone_b_range_hz: int
    - record_length_ms: int
    - alert_type: str
```

**Functions:**
- `init_tone_detection()` - Initialize system
- `add_tone_definition(def)` - Add tone pattern
- `process_audio_samples(samples)` - Analyze audio
- `detect_tones(samples)` - Detect tone sequences
- `reset_detection_state()` - Reset state
- `get_active_tone()` - Get currently detected tone

**Dependencies:** `numpy`, `scipy`, `audio_workers.py`

---

#### Module 5.2: Tone Passthrough (`tone_passthrough.py`)
**Purpose:** Route audio when tones are detected

**Components:**
- Passthrough state machine
- Recording timer
- Audio routing logic

**Functions:**
- `enable_passthrough(tone_def, target_channel)` - Start passthrough
- `disable_passthrough()` - Stop passthrough
- `is_passthrough_active()` - Check state
- `get_remaining_time_ms()` - Get recording time left

**Dependencies:** `tone_detection.py`, `audio_stream.py`

---

### **PHASE 6: Integration & Control** 🔄

#### Module 6.1: Channel Manager (`channel_manager.py`)
**Purpose:** Manage multiple audio channels

**Components:**
- Channel registry
- Channel lifecycle management
- GPIO to channel mapping
- Transmission control

**Functions:**
- `create_channel(channel_id, device_index)` - Create channel
- `start_channel(channel_id)` - Start channel
- `stop_channel(channel_id)` - Stop channel
- `handle_gpio_change(channel_id, gpio_active)` - Handle PTT
- `get_all_channels()` - List all channels
- `cleanup_all_channels()` - Cleanup all

**Dependencies:** `audio_stream.py`, `gpio.py`, `websocket_client.py`, `udp_manager.py`

---

#### Module 6.2: Event Coordinator (`event_coordinator.py`)
**Purpose:** Coordinate events between modules

**Components:**
- Event handlers registry
- Event routing
- State synchronization

**Functions:**
- `register_handler(event_type, handler)` - Register event handler
- `emit_event(event_type, data)` - Emit event
- `handle_gpio_event(channel_id, state)` - Route GPIO events
- `handle_tone_detected(tone_def)` - Route tone events
- `handle_udp_ready()` - Handle UDP ready event

**Dependencies:** All modules (coordinator)

---

### **PHASE 7: Optional Features** 🎁

#### Module 7.1: MQTT Publisher (`mqtt_publisher.py`)
**Purpose:** Publish events to MQTT broker

**Functions:**
- `init_mqtt(broker, port)` - Initialize MQTT
- `publish_event(topic, data)` - Publish message
- `publish_tone_detected(tone_id)` - Publish tone event
- `cleanup_mqtt()` - Disconnect

**Dependencies:** `paho-mqtt`, `config.py`

---

#### Module 7.2: S3 Uploader (`s3_uploader.py`)
**Purpose:** Upload recorded audio to S3

**Functions:**
- `init_s3(credentials)` - Initialize S3 client
- `upload_audio(audio_data, filename)` - Upload audio file
- `get_upload_url(filename)` - Get presigned URL

**Dependencies:** `boto3`, `config.py`

---

### **PHASE 8: Main Application** 🚀

#### Module 8.1: Application Entry Point (`main.py`)
**Purpose:** Application initialization and main loop

**Flow:**
1. Load configuration
2. Initialize GPIO
3. Initialize audio hardware
4. Create channels from config
5. Connect WebSocket
6. Start worker threads:
   - GPIO monitor
   - Audio input workers (per channel)
   - Audio output workers (per channel)
   - UDP receiver
   - Tone detection worker
   - WebSocket handler
7. Main loop (wait for shutdown)
8. Cleanup on exit

**Functions:**
- `main()` - Entry point
- `handle_shutdown()` - Signal handler
- `start_all_workers()` - Start all threads
- `cleanup_all()` - Cleanup all resources

**Dependencies:** All modules

---

## Implementation Order

### **Week 1: Foundation**
1. ✅ Module 1.1: echostream.py
2. ✅ Module 1.2: config.py

### **Week 2: Hardware**
3. ✅ Module 2.1: gpio.py
4. ✅ Module 2.2: audio_hardware.py

### **Week 3: Audio Core**
5. ✅ Module 3.1: audio_stream.py
6. ✅ Module 3.2: audio_workers.py

### **Week 4: Network**
7. ✅ Module 4.1: websocket_client.py
8. ✅ Module 4.2: udp_manager.py
9. ✅ Module 4.3: crypto.py

### **Week 5: Tone Detection**
10. ✅ Module 5.1: tone_detection.py
11. ✅ Module 5.2: tone_passthrough.py

### **Week 6: Integration**
12. ✅ Module 6.1: channel_manager.py
13. ✅ Module 6.2: event_coordinator.py

### **Week 7: Optional Features**
14. ✅ Module 7.1: mqtt_publisher.py (optional)
15. ✅ Module 7.2: s3_uploader.py (optional)

### **Week 8: Main App & Testing**
16. ✅ Module 8.1: main.py
17. ✅ Integration testing
18. ✅ Bug fixes and optimization

---

## Key Design Principles

1. **Separation of Concerns:** Each module has a single, well-defined responsibility
2. **Dependency Injection:** Modules receive dependencies, not import them globally
3. **Thread Safety:** All shared state uses locks/mutexes
4. **Error Handling:** Comprehensive try/except with logging
5. **Resource Management:** Proper cleanup in finally blocks
6. **Configuration-Driven:** Behavior controlled via config file
7. **Testability:** Modules can be tested independently

---

## Data Flow

```
GPIO Pin Change
    ↓
GPIO Monitor
    ↓
Channel Manager
    ↓
Audio Stream (Start/Stop)
    ↓
Audio Workers
    ↓
├─→ UDP Sender (Encrypt & Send)
└─→ Shared Buffer → Tone Detection
                        ↓
                    Tone Passthrough
                        ↓
                    Audio Routing
```

---

## Testing Strategy

1. **Unit Tests:** Each module tested independently
2. **Integration Tests:** Test module interactions
3. **Hardware Tests:** Test with actual GPIO and audio devices
4. **Network Tests:** Test WebSocket and UDP communication
5. **Stress Tests:** Test with multiple channels and high load

---

## File Structure

```
EchoStream/
├── echostream.py          # Constants & shared definitions
├── config.py              # Configuration manager
├── gpio.py                # GPIO controller
├── audio_hardware.py      # Audio device management
├── audio_stream.py        # Audio stream management
├── audio_workers.py       # Audio I/O workers
├── websocket_client.py    # WebSocket client
├── udp_manager.py         # UDP manager
├── crypto.py              # Encryption utilities
├── tone_detection.py      # Tone detection engine
├── tone_passthrough.py    # Tone passthrough routing
├── channel_manager.py     # Channel lifecycle manager
├── event_coordinator.py   # Event coordination
├── mqtt_publisher.py      # MQTT integration (optional)
├── s3_uploader.py         # S3 upload (optional)
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
├── config.json            # Configuration file
└── README.md              # Documentation
```

---

## Next Steps

1. Review and approve this architecture plan
2. Start with Module 1.1 (echostream.py)
3. Implement modules in order
4. Test each module as it's completed
5. Integrate and test full system

---

**Ready to begin implementation?**

