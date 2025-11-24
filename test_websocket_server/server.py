#!/usr/bin/env python3
"""
Test WebSocket Server for EchoStream
This server mimics the behavior of the production WebSocket server for local testing.
"""

import asyncio
import websockets
import json
import time
import socket
import base64
import os
import threading
from typing import Set, Dict, Any, Optional, Tuple
from pathlib import Path

# Audio processing imports
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("[SERVER] WARNING: pydub not available. Install with: pip install pydub")

try:
    import opuslib
    HAS_OPUS = True
except ImportError:
    HAS_OPUS = False
    print("[SERVER] WARNING: opuslib not available. Install with: pip install opuslib")

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_AES = True
except ImportError:
    HAS_AES = False
    print("[SERVER] WARNING: cryptography not available. Install with: pip install cryptography")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[SERVER] WARNING: numpy not available. Install with: pip install numpy")

# Server configuration
HOST = "localhost"
PORT = 8765
WS_URL = f"ws://{HOST}:{PORT}"

# Audio directory
AUDIO_DIR = Path(__file__).parent / "audio"

# Track connected clients and channels
connected_clients: Set[websockets.WebSocketServerProtocol] = set()
registered_channels: Dict[str, Dict[str, Any]] = {}  # channel_id -> client info
channel_to_client: Dict[str, websockets.WebSocketServerProtocol] = {}  # channel_id -> client

# Track UDP connections and audio streaming
channel_udp_info: Dict[str, Dict[str, Any]] = {}  # channel_id -> {udp_port, udp_host, aes_key, socket, etc}
audio_streaming_tasks: Dict[str, asyncio.Task] = {}  # channel_id -> streaming task
streaming_active: Dict[str, bool] = {}  # channel_id -> is streaming


async def handle_client(websocket: websockets.WebSocketServerProtocol, path: str):
    """Handle a new WebSocket client connection."""
    client_addr = websocket.remote_address
    print(f"[SERVER] New client connected from {client_addr}")
    connected_clients.add(websocket)
    
    try:
        # Send initial users_connected message
        await send_users_connected(websocket)
        
        # Handle messages from client
        async for message in websocket:
            try:
                # Decode message if bytes
                if isinstance(message, bytes):
                    message = message.decode('utf-8')
                
                # Parse JSON
                data = json.loads(message)
                print(f"[SERVER] Received from {client_addr}: {json.dumps(data, indent=2)}")
                
                # Handle different message types
                await handle_message(websocket, data, client_addr)
                
            except json.JSONDecodeError as e:
                print(f"[SERVER] ERROR: Failed to parse JSON from {client_addr}: {e}")
                print(f"[SERVER] Raw message: {message[:200]}")
            except Exception as e:
                print(f"[SERVER] ERROR: Exception handling message from {client_addr}: {e}")
                import traceback
                traceback.print_exc()
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[SERVER] Client {client_addr} disconnected")
    except Exception as e:
        print(f"[SERVER] ERROR: Exception in client handler for {client_addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        connected_clients.discard(websocket)
        # Remove from channel mappings and stop audio streaming
        channels_to_remove = [ch_id for ch_id, client in channel_to_client.items() if client == websocket]
        for ch_id in channels_to_remove:
            await stop_audio_streaming(ch_id)
            del channel_to_client[ch_id]
            if ch_id in registered_channels:
                del registered_channels[ch_id]
            if ch_id in channel_udp_info:
                del channel_udp_info[ch_id]
        print(f"[SERVER] Client {client_addr} cleaned up. Active clients: {len(connected_clients)}")


async def handle_message(websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any], client_addr: tuple):
    """Handle different types of messages from the client."""
    
    # Handle "register" message
    if "type" in data and data["type"] == "register":
        channel_id = data.get("channel_id")
        if channel_id:
            print(f"[SERVER] Channel registration: {channel_id}")
            registered_channels[channel_id] = {
                "channel_id": channel_id,
                "client": client_addr,
                "registered_at": time.time()
            }
            channel_to_client[channel_id] = websocket
            # Send UDP config for this channel
            await send_udp_config(websocket, channel_id)
            # Send updated users_connected
            await send_users_connected(websocket)
    
    # Handle "connect" message
    elif "connect" in data:
        connect_data = data["connect"]
        channel_id = connect_data.get("channel_id")
        if channel_id:
            print(f"[SERVER] Channel connect: {channel_id} from {connect_data.get('user_name', 'Unknown')}")
            registered_channels[channel_id] = {
                "channel_id": channel_id,
                "client": client_addr,
                "user_name": connect_data.get("user_name"),
                "affiliation_id": connect_data.get("affiliation_id"),
                "agency_name": connect_data.get("agency_name"),
                "connected_at": time.time()
            }
            channel_to_client[channel_id] = websocket
            # Send UDP config for this channel
            await send_udp_config(websocket, channel_id)
            # Send updated users_connected
            await send_users_connected(websocket)
    
    # Handle "transmit_started" message
    elif "transmit_started" in data:
        transmit_data = data["transmit_started"]
        channel_id = transmit_data.get("channel_id")
        print(f"[SERVER] Transmit STARTED on channel {channel_id}")
        
        # Start audio streaming for this channel
        await start_audio_streaming(channel_id)
        
        # Echo back or handle as needed
        response = {
            "type": "transmit_ack",
            "channel_id": channel_id,
            "status": "ok",
            "time": int(time.time())
        }
        await websocket.send(json.dumps(response))
    
    # Handle "transmit_ended" message
    elif "transmit_ended" in data:
        transmit_data = data["transmit_ended"]
        channel_id = transmit_data.get("channel_id")
        print(f"[SERVER] Transmit ENDED on channel {channel_id}")
        
        # Stop audio streaming for this channel
        await stop_audio_streaming(channel_id)
        
        # Echo back or handle as needed
        response = {
            "type": "transmit_ack",
            "channel_id": channel_id,
            "status": "ok",
            "time": int(time.time())
        }
        await websocket.send(json.dumps(response))
    
    else:
        print(f"[SERVER] Unhandled message type: {list(data.keys())}")


async def send_udp_config(websocket: websockets.WebSocketServerProtocol, channel_id: str):
    """Send UDP configuration to the client for a channel."""
    # Generate a unique UDP port for this channel (base port + hash of channel_id)
    base_port = 50000
    port_offset = hash(channel_id) % 1000
    udp_port = base_port + abs(port_offset)
    
    # Use localhost for testing
    udp_host = "127.0.0.1"
    
    # Generate a dummy AES key (base64 encoded) - use the same hardcoded key as client expects
    # Client uses: "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0=" if server sends N/A
    dummy_key = "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0="
    
    config = {
        "udp_port": udp_port,
        "udp_host": udp_host,
        "websocket_id": channel_id,
        "aes_key": dummy_key
    }
    
    # Store UDP info for this channel
    channel_udp_info[channel_id] = {
        "udp_port": udp_port,
        "udp_host": udp_host,
        "aes_key": dummy_key,
        "client_addr": None  # Will be set when we receive first UDP packet
    }
    
    print(f"[SERVER] Sending UDP config to channel {channel_id}: port={udp_port}, host={udp_host}")
    await websocket.send(json.dumps(config))
    
    # Start UDP listener for this channel
    asyncio.create_task(start_udp_listener(channel_id, udp_port))


async def send_users_connected(websocket: websockets.WebSocketServerProtocol):
    """Send users_connected message to the client."""
    # Build list of connected users/channels
    connected_users = []
    for channel_id, info in registered_channels.items():
        user_info = {
            "id": info.get("affiliation_id", f"user_{channel_id}"),
            "user_id": info.get("affiliation_id", f"user_{channel_id}"),
            "user_name": info.get("user_name", "EchoStream"),
            "agency_name": info.get("agency_name", "Python"),
            "channel_id": channel_id
        }
        connected_users.append(user_info)
    
    message = {
        "users_connected": {
            "total": len(connected_users),
            "connected": connected_users
        }
    }
    
    print(f"[SERVER] Sending users_connected: {len(connected_users)} user(s)")
    await websocket.send(json.dumps(message))


async def start_udp_listener(channel_id: str, udp_port: int):
    """Start UDP listener for a channel to receive audio from client."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    sock.setblocking(False)
    
    print(f"[UDP] Listening on port {udp_port} for channel {channel_id}")
    
    loop = asyncio.get_event_loop()
    
    while channel_id in channel_udp_info:
        try:
            data, addr = await loop.sock_recvfrom(sock, 8192)
            
            # Store client address for sending audio back
            if channel_udp_info[channel_id]["client_addr"] is None:
                channel_udp_info[channel_id]["client_addr"] = addr
                print(f"[UDP] Client connected from {addr} "
                      f"for channel {channel_id}")
            
            # Handle heartbeat messages
            try:
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                if msg.get("type") == "KEEP_ALIVE":
                    continue
            except Exception:
                pass
            
        except asyncio.CancelledError:
            break
        except Exception:
            if channel_id in channel_udp_info:
                await asyncio.sleep(0.1)
            else:
                break
    
    sock.close()
    print(f"[UDP] Stopped listening on port {udp_port} "
          f"for channel {channel_id}")


def load_audio_file(channel_id: str) -> Optional[AudioSegment]:
    """Load an audio file for a channel. Cycles through available files."""
    if not HAS_PYDUB:
        return None
    
    if not AUDIO_DIR.exists():
        print(f"[AUDIO] ERROR: Audio directory not found: {AUDIO_DIR}")
        return None
    
    # Get list of MP3 files
    audio_files = list(AUDIO_DIR.glob("*.mp3"))
    if not audio_files:
        print(f"[AUDIO] ERROR: No MP3 files found in {AUDIO_DIR}")
        return None
    
    # Select file based on channel_id hash
    file_index = hash(channel_id) % len(audio_files)
    selected_file = audio_files[file_index]
    
    try:
        print(f"[AUDIO] Loading audio file: {selected_file.name} for channel {channel_id}")
        audio = AudioSegment.from_mp3(str(selected_file))
        # Convert to mono, 48kHz (Opus standard)
        audio = audio.set_channels(1).set_frame_rate(48000)
        print(f"[AUDIO] Loaded {len(audio)}ms of audio, sample rate: {audio.frame_rate}Hz")
        return audio
    except Exception as e:
        print(f"[AUDIO] ERROR: Failed to load audio file {selected_file}: {e}")
        return None


async def stream_audio_to_channel(channel_id: str):
    """Stream audio from file to a channel over UDP."""
    if not HAS_PYDUB or not HAS_OPUS or not HAS_AES or not HAS_NUMPY:
        print(f"[AUDIO] ERROR: Missing required libraries for audio streaming")
        return
    
    if channel_id not in channel_udp_info:
        print(f"[AUDIO] ERROR: No UDP info for channel {channel_id}")
        return
    
    udp_info = channel_udp_info[channel_id]
    client_addr = udp_info.get("client_addr")
    
    if client_addr is None:
        print(f"[AUDIO] Waiting for client to connect on UDP for channel {channel_id}")
        # Wait for client to send first packet
        max_wait = 30  # Wait up to 30 seconds
        waited = 0
        while client_addr is None and waited < max_wait and channel_id in channel_udp_info:
            await asyncio.sleep(0.5)
            client_addr = channel_udp_info[channel_id].get("client_addr")
            waited += 0.5
        
        if client_addr is None:
            print(f"[AUDIO] ERROR: Client never connected on UDP for channel {channel_id}")
            return
    
    # Load audio file
    audio_segment = load_audio_file(channel_id)
    if audio_segment is None:
        return
    
    # Initialize Opus encoder
    try:
        encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_VOIP)
        encoder.bitrate = 64000
        encoder.vbr = True
    except Exception as e:
        print(f"[AUDIO] ERROR: Failed to create Opus encoder: {e}")
        return
    
    # Initialize AES encryption
    try:
        aes_key = base64.b64decode(udp_info["aes_key"])
        if len(aes_key) != 32:
            print(f"[AUDIO] ERROR: Invalid AES key length: {len(aes_key)}")
            return
        aesgcm = AESGCM(aes_key)
    except Exception as e:
        print(f"[AUDIO] ERROR: Failed to initialize AES: {e}")
        return
    
    # Create UDP socket for sending
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Convert audio to numpy array
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    # Normalize to [-1.0, 1.0]
    if audio_segment.sample_width == 2:  # 16-bit
        samples = samples / 32768.0
    elif audio_segment.sample_width == 4:  # 32-bit
        samples = samples / 2147483648.0
    
    # Frame size for Opus: 1920 samples (40ms at 48kHz)
    frame_size = 1920
    sample_rate = 48000
    frame_duration_ms = (frame_size / sample_rate) * 1000  # ~40ms
    
    print(f"[AUDIO] Starting audio stream for channel {channel_id}")
    print(f"[AUDIO] Total samples: {len(samples)}, frame size: {frame_size}")
    
    packet_count = 0
    streaming_active[channel_id] = True
    
    try:
        # Stream audio in frames
        for i in range(0, len(samples), frame_size):
            if not streaming_active.get(channel_id, False) or channel_id not in channel_udp_info:
                break
            
            frame_samples = samples[i:i+frame_size]
            
            # Pad if last frame is short
            if len(frame_samples) < frame_size:
                padding = np.zeros(frame_size - len(frame_samples), dtype=np.float32)
                frame_samples = np.concatenate([frame_samples, padding])
            
            # Convert to int16 PCM
            pcm = (np.clip(frame_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
            pcm_bytes = pcm.tobytes()
            
            # Encode with Opus
            try:
                opus_data = encoder.encode(pcm_bytes, frame_size)
            except Exception as e:
                print(f"[AUDIO] ERROR: Opus encode failed: {e}")
                continue
            
            # Encrypt with AES
            try:
                iv = os.urandom(12)
                encrypted = aesgcm.encrypt(iv, opus_data, None)
                encrypted_with_iv = iv + encrypted
                b64_data = base64.b64encode(encrypted_with_iv).decode("utf-8")
            except Exception as e:
                print(f"[AUDIO] ERROR: AES encrypt failed: {e}")
                continue
            
            # Send over UDP
            message = json.dumps({
                "channel_id": channel_id,
                "type": "audio",
                "data": b64_data
            })
            
            try:
                sock.sendto(message.encode("utf-8"), client_addr)
                packet_count += 1
                
                if packet_count <= 5 or packet_count % 100 == 0:
                    print(f"[AUDIO] Sent packet #{packet_count} to {client_addr} for channel {channel_id}")
            except Exception as e:
                print(f"[AUDIO] ERROR: UDP send failed: {e}")
            
            # Wait for next frame (40ms)
            await asyncio.sleep(frame_duration_ms / 1000.0)
        
        print(f"[AUDIO] Finished streaming {packet_count} packets for channel {channel_id}")
        
        # Loop the audio if still active
        if streaming_active.get(channel_id, False) and channel_id in channel_udp_info:
            print(f"[AUDIO] Looping audio for channel {channel_id}")
            await stream_audio_to_channel(channel_id)  # Recursive call to loop
        
    except Exception as e:
        print(f"[AUDIO] ERROR: Exception in audio streaming: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        streaming_active[channel_id] = False
        print(f"[AUDIO] Stopped streaming for channel {channel_id}")


async def start_audio_streaming(channel_id: str):
    """Start audio streaming task for a channel."""
    if channel_id in audio_streaming_tasks:
        # Already streaming
        return
    
    if channel_id not in channel_udp_info:
        print(f"[AUDIO] ERROR: No UDP config for channel {channel_id}")
        return
    
    streaming_active[channel_id] = True
    task = asyncio.create_task(stream_audio_to_channel(channel_id))
    audio_streaming_tasks[channel_id] = task
    print(f"[AUDIO] Started audio streaming task for channel {channel_id}")


async def stop_audio_streaming(channel_id: str):
    """Stop audio streaming for a channel."""
    streaming_active[channel_id] = False
    
    if channel_id in audio_streaming_tasks:
        task = audio_streaming_tasks[channel_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        del audio_streaming_tasks[channel_id]
        print(f"[AUDIO] Stopped audio streaming for channel {channel_id}")


async def periodic_updates():
    """Send periodic updates to all connected clients."""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        if connected_clients:
            print(f"[SERVER] Sending periodic users_connected to {len(connected_clients)} client(s)")
            for client in list(connected_clients):
                try:
                    await send_users_connected(client)
                except Exception as e:
                    print(f"[SERVER] ERROR: Failed to send periodic update: {e}")


async def main():
    """Start the WebSocket server."""
    print("=" * 60)
    print("EchoStream Test WebSocket Server")
    print("=" * 60)
    print(f"Starting server on {HOST}:{PORT}")
    print(f"Connect your client to: {WS_URL}")
    print("=" * 60)
    
    # Start periodic updates task
    asyncio.create_task(periodic_updates())
    
    # Start WebSocket server
    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        subprotocols=["audio-protocol"],
        ping_interval=20,
        ping_timeout=20
    ):
        print(f"[SERVER] Server started and listening on {WS_URL}")
        print("[SERVER] Press Ctrl+C to stop")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    except Exception as e:
        print(f"[SERVER] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

