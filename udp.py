"""
UDP communication - handles UDP socket setup, heartbeat, and audio packet reception
"""
import socket
import threading
import json
import os
import time
from typing import Optional
from echostream import global_interrupted, MAX_CHANNELS
import audio
import crypto

# Global UDP state
global_udp_socket: Optional[socket.socket] = None
global_server_addr: Optional[tuple] = None
heartbeat_thread: Optional[threading.Thread] = None
udp_listener_thread: Optional[threading.Thread] = None

# Statistics
zero_key_warned = [False] * MAX_CHANNELS
jitter_drop_count = [0] * MAX_CHANNELS
decrypt_fail_count = [0] * MAX_CHANNELS

def udp_debug_enabled() -> bool:
    """Check if UDP debug is enabled via environment variable"""
    env = os.getenv("UDP_DEBUG")
    return env is not None and env != "0"

def setup_global_udp(config: dict) -> bool:
    """Setup global UDP socket"""
    global global_udp_socket, global_server_addr, heartbeat_thread, udp_listener_thread
    
    if global_udp_socket is not None:
        print(f"UDP socket already configured for {config['udp_host']}:{config['udp_port']}")
        return True
    
    try:
        global_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Enable socket reuse (allows binding to same port if needed)
        global_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Get local port info (socket gets assigned a port when we send)
        global_server_addr = (config['udp_host'], config['udp_port'])
        
        print(f"Global UDP socket configured for {config['udp_host']}:{config['udp_port']}")
        
        # Send immediate heartbeat to establish connection (this assigns a local port)
        heartbeat_msg = b'{"type":"KEEP_ALIVE"}'
        try:
            global_udp_socket.sendto(heartbeat_msg, global_server_addr)
            print("Initial heartbeat sent immediately upon UDP connection")
            
            # Get local port info after sending (like C implementation)
            try:
                local_addr = global_udp_socket.getsockname()
                print(f"[UDP] Socket local address: {local_addr[0]}:{local_addr[1]}")
            except Exception as e:
                print(f"[UDP] Could not get local socket info: {e}")
        except Exception as e:
            print(f"Initial heartbeat error: {e}")
        
        # Start heartbeat thread
        if heartbeat_thread is None or not heartbeat_thread.is_alive():
            heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
            heartbeat_thread.start()
        
        # Start UDP listener thread
        if udp_listener_thread is None or not udp_listener_thread.is_alive():
            udp_listener_thread = threading.Thread(target=udp_listener_worker, daemon=True)
            udp_listener_thread.start()
        
        return True
    except Exception as e:
        print(f"UDP setup failed: {e}")
        return False

def heartbeat_worker(arg=None):
    """Heartbeat worker thread - sends keep-alive messages"""
    global global_udp_socket, global_server_addr
    
    print("Heartbeat worker started")
    heartbeat_count = 0
    
    while not global_interrupted.is_set():
        if global_udp_socket is not None and global_server_addr is not None:
            heartbeat_msg = b'{"type":"KEEP_ALIVE"}'
            try:
                global_udp_socket.sendto(heartbeat_msg, global_server_addr)
                heartbeat_count += 1
                # Only log every 60th heartbeat (about every 10 minutes)
                if heartbeat_count % 60 == 0:
                    print(f"Heartbeat sent to keep NAT mapping active (count: {heartbeat_count})")
            except Exception as e:
                print(f"Heartbeat error: {e}")
        
        # Sleep for 10 seconds (100 iterations * 0.1s)
        for _ in range(100):
            if global_interrupted.is_set():
                break
            time.sleep(0.1)
    
    print("Heartbeat worker stopped")
    return None

def udp_listener_worker(arg=None):
    """UDP listener worker thread - receives and processes audio packets"""
    global global_udp_socket
    
    print("UDP listener worker started")
    
    if global_udp_socket is None:
        print("UDP Listener: ERROR - Invalid socket")
        return None
    
    print("Listening on UDP socket (blocking mode, 10s timeout)...")
    print(f"[UDP] Socket details: {global_udp_socket}")
    if global_server_addr:
        print(f"[UDP] Server address: {global_server_addr}")
    
    # Get socket info for debugging
    try:
        local_addr = global_udp_socket.getsockname()
        print(f"[UDP] Socket bound to: {local_addr[0]}:{local_addr[1]}")
    except Exception as e:
        print(f"[UDP] Could not get socket info: {e}")
    
    packet_count = 0
    timeout_count = 0
    last_packet_time = time.time()
    startup_time = time.time()
    
    # Log initial state
    print("[UDP] Listener started, waiting for packets...")
    print("[UDP] NOTE: Server may only send audio when client is transmitting!")
    print("[UDP] Make sure GPIO is active and input audio is being captured.")
    
    while not global_interrupted.is_set():
        try:
            # Match C implementation: Use blocking recvfrom() with no timeout
            # C uses blocking recvfrom() which waits indefinitely for packets
            # Use select() to make it truly blocking while still checking global_interrupted
            import select
            
            # Use select() to wait for data with a timeout, allowing interrupt checking
            # This matches C's blocking behavior more closely than settimeout()
            ready, _, _ = select.select([global_udp_socket], [], [], 1.0)  # 1s timeout for interrupt checking
            
            if not ready:
                # Timeout - check if we should continue
                timeout_count += 1
                time_since_last = time.time() - last_packet_time if packet_count > 0 else float('inf')
                elapsed_since_startup = time.time() - startup_time
                
                # Log timeouts occasionally (with 1s timeout, timeouts are frequent)
                if timeout_count % 10 == 0:  # Log every 10th timeout (10 seconds)
                    print(f"[UDP] Timeout #{timeout_count} waiting for packets (total_packets={packet_count}, "
                          f"time_since_last_packet={time_since_last:.2f}s, elapsed={elapsed_since_startup:.1f}s)")
                    
                    # Warn if we've been waiting a long time without packets
                    if timeout_count == 10:  # 10 seconds
                        print(f"[UDP WARNING] 10 seconds elapsed - no packets received!")
                        print(f"[UDP WARNING] Possible causes:")
                        print(f"[UDP WARNING]   1. Server only sends audio when client is transmitting")
                        print(f"[UDP WARNING]   2. Check GPIO is active and input audio is being captured")
                        print(f"[UDP WARNING]   3. Network/firewall issue blocking UDP packets")
                        print(f"[UDP WARNING]   4. Server not sending packets yet")
                        if packet_count == 0:
                            print(f"[UDP WARNING] NO packets received since startup - verify client is transmitting audio!")
                        else:
                            print(f"[UDP WARNING] Last packet was {time_since_last:.1f}s ago - packets may have stopped arriving")
                continue
            
            # Data is ready - receive it (this should not block now)
            buffer, client_addr = global_udp_socket.recvfrom(8192)
            
            packet_count += 1
            last_packet_time = time.time()
            
            # Log when we receive packets after timeouts (important for diagnosing)
            # Check BEFORE resetting timeout_count
            if timeout_count > 0:
                print(f"[UDP RX] Packet received after {timeout_count} timeouts! Packet #{packet_count} from {client_addr} "
                      f"(was waiting {timeout_count*0.1:.1f}s)")
            
            # Reset timeout count on successful receive
            timeout_count = 0
            
            # Log ALL packets initially (first 20), then occasionally
            if packet_count <= 20:
                print(f"[UDP RX] Received packet #{packet_count} from {client_addr} ({len(buffer)} bytes)")
            elif packet_count % 500 == 0:
                print(f"[UDP RX] Received packet #{packet_count} from {client_addr} ({len(buffer)} bytes)")
            
            if buffer:
                try:
                    data_str = buffer.decode('utf-8')
                    
                    # Parse JSON message
                    json_data = json.loads(data_str)
                    
                    channel_id = json_data.get('channel_id', '')
                    msg_type = json_data.get('type', '')
                    data = json_data.get('data', '')
                    
                    # Log audio messages occasionally
                    if msg_type == 'audio':
                        static_audio_count = getattr(udp_listener_worker, '_audio_count', {})
                        if channel_id not in static_audio_count:
                            static_audio_count[channel_id] = 0
                        static_audio_count[channel_id] += 1
                        udp_listener_worker._audio_count = static_audio_count
                        
                        # Log first few audio messages and then occasionally
                        if static_audio_count[channel_id] <= 5 or static_audio_count[channel_id] % 500 == 0:
                            print(f"[UDP AUDIO] Channel {channel_id}: Received audio packet #{static_audio_count[channel_id]} "
                                  f"({len(buffer)} bytes, data_length={len(data)})")
                    
                    # Log non-audio messages
                    elif msg_type:
                        print(f"[UDP] Received {msg_type} message: channel_id={channel_id}, data_length={len(data)}")
                    
                    if msg_type == 'audio':
                        # Find the channel
                        target_stream = None
                        target_index = -1
                        
                        for i in range(MAX_CHANNELS):
                            if audio.channels[i].active and audio.channels[i].audio.channel_id == channel_id:
                                target_stream = audio.channels[i].audio
                                target_index = i
                                break
                        
                        if not target_stream:
                            # Log missing channel occasionally to avoid spam
                            if static_audio_count[channel_id] <= 5 or static_audio_count[channel_id] % 500 == 0:
                                active_channels = [audio.channels[i].audio.channel_id 
                                                 for i in range(MAX_CHANNELS) if audio.channels[i].active]
                                print(f"[UDP ERROR] Channel {channel_id} not found! Active channels: {active_channels}")
                            continue
                        
                        # Decode base64 data
                        encrypted_data = crypto.decode_base64(data)
                        
                        if len(encrypted_data) > 0:
                            if udp_debug_enabled():
                                print(f"UDP Listener: Base64 decoded successfully ({len(encrypted_data)} bytes)")
                            
                            # Check if key is zero
                            key_is_zero = all(b == 0 for b in target_stream.key)
                            
                            if not key_is_zero:
                                zero_key_warned[target_index] = False
                            
                            # Decrypt the data
                            decrypted = crypto.decrypt_data(encrypted_data, bytes(target_stream.key))
                            
                            if decrypted:
                                if udp_debug_enabled():
                                    print(f"UDP Listener: Data decrypted successfully ({len(decrypted)} bytes)")
                                
                                # Decode Opus audio and add to jitter buffer
                                audio.process_received_audio(target_stream, decrypted, channel_id, target_index)
                            else:
                                if key_is_zero and target_index >= 0:
                                    if not zero_key_warned[target_index]:
                                        print(f"UDP Listener: AES key not set for channel {channel_id}; dropping encrypted audio until key is provisioned")
                                        zero_key_warned[target_index] = True
                                else:
                                    if target_index >= 0:
                                        decrypt_fail_count[target_index] += 1
                                        if udp_debug_enabled() or decrypt_fail_count[target_index] == 1 or decrypt_fail_count[target_index] % 50 == 0:
                                            print(f"UDP Listener: Decryption failed for channel {channel_id}")
                                    else:
                                        print("UDP Listener: Decryption failed")
                        else:
                            if udp_debug_enabled():
                                print("UDP Listener: Base64 decode failed")
                    else:
                        if udp_debug_enabled():
                            print(f"UDP Listener: Non-audio message type '{msg_type}', ignoring")
                            
                except json.JSONDecodeError:
                    if udp_debug_enabled():
                        print("UDP Listener: Failed to parse JSON")
                except Exception as e:
                    if udp_debug_enabled():
                        print(f"UDP Listener: Error processing message: {e}")
        
        except socket.timeout:
            # This shouldn't happen with select(), but handle it just in case
            timeout_count += 1
            continue
        except Exception as e:
            if not global_interrupted.is_set():
                print(f"[UDP ERROR] Receive error: {e}")
                import traceback
                traceback.print_exc()
            time.sleep(0.1)
            continue
    
    print("UDP listener worker stopped")
    return None

