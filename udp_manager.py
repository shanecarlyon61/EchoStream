"""
UDP Manager - UDP audio packet transmission and reception

This module handles UDP socket management, audio packet sending/receiving,
heartbeat mechanism, and packet encryption/decryption.
"""
import socket
import select
import threading
import time
import json
from typing import Optional, Tuple, Callable
from echostream import global_interrupted
import crypto


# ============================================================================
# Global State
# ============================================================================

# UDP socket instance
global_udp_socket: Optional[socket.socket] = None

# Server address
global_server_addr: Optional[Tuple[str, int]] = None

# Heartbeat worker thread
heartbeat_thread: Optional[threading.Thread] = None

# UDP receiver worker thread
udp_listener_thread: Optional[threading.Thread] = None

# Packet receive callback
packet_receive_callback: Optional[Callable[[str, bytes], None]] = None

# Heartbeat state
heartbeat_enabled = False
heartbeat_interval = 5.0  # seconds


# ============================================================================
# UDP Socket Management
# ============================================================================

def setup_udp(host: str, port: int, bind_port: int = 0) -> bool:
    """
    Create and configure UDP socket.
    
    Args:
        host: Server hostname or IP address
        port: Server UDP port
        bind_port: Local port to bind to (0 = auto-assign)
        
    Returns:
        True if setup successful, False otherwise
    """
    global global_udp_socket, global_server_addr
    
    if global_udp_socket is not None:
        print("[UDP] UDP socket already configured")
        return True
    
    try:
        # Create UDP socket
        global_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        global_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Set socket to non-blocking for use with select
        global_udp_socket.setblocking(False)
        
        # Bind to local port
        global_udp_socket.bind(('0.0.0.0', bind_port))
        
        # Get local address
        local_addr = global_udp_socket.getsockname()
        print(f"[UDP] Socket bound to: {local_addr[0]}:{local_addr[1]}")
        
        # Set server address
        global_server_addr = (host, port)
        print(f"[UDP] Server address: {global_server_addr}")
        
        # Send initial heartbeat immediately
        send_heartbeat()
        print("[UDP] Initial heartbeat sent")
        
        print(f"[UDP] UDP socket configured for {host}:{port}")
        return True
        
    except Exception as e:
        print(f"[UDP] ERROR: Failed to setup UDP socket: {e}")
        cleanup_udp()
        return False


def cleanup_udp():
    """Close UDP socket and cleanup."""
    global global_udp_socket, global_server_addr, heartbeat_enabled
    
    heartbeat_enabled = False
    
    if global_udp_socket is not None:
        try:
            global_udp_socket.close()
            print("[UDP] UDP socket closed")
        except Exception as e:
            print(f"[UDP] ERROR: Exception closing UDP socket: {e}")
        finally:
            global_udp_socket = None
            global_server_addr = None


def get_server_address() -> Optional[Tuple[str, int]]:
    """
    Get server address.
    
    Returns:
        Tuple of (host, port) or None if not configured
    """
    return global_server_addr


def is_udp_ready() -> bool:
    """
    Check if UDP is ready (socket configured and server address set).
    
    Returns:
        True if UDP is ready, False otherwise
    """
    return global_udp_socket is not None and global_server_addr is not None


# ============================================================================
# Audio Packet Transmission
# ============================================================================

def send_audio_packet(channel_id: str, opus_data: bytes, encryption_key: bytes) -> bool:
    """
    Send audio packet via UDP.
    
    Args:
        channel_id: Channel ID string
        opus_data: Opus-encoded audio data bytes
        encryption_key: Encryption key bytes (32 bytes)
        
    Returns:
        True if packet sent successfully, False otherwise
    """
    global global_udp_socket, global_server_addr
    
    if not is_udp_ready():
        return False
    
    try:
        # Encrypt audio data
        encrypted_data = crypto.encrypt_aes(opus_data, encryption_key)
        if encrypted_data is None:
            print(f"[UDP] ERROR: Failed to encrypt audio for channel {channel_id}")
            return False
        
        # Encode to base64
        b64_data = crypto.encode_base64(encrypted_data)
        
        # Create JSON message
        message = {
            "channel_id": channel_id,
            "type": "audio",
            "data": b64_data
        }
        
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        # Send via UDP
        global_udp_socket.sendto(message_bytes, global_server_addr)
        return True
        
    except Exception as e:
        print(f"[UDP] ERROR: Failed to send audio packet for channel {channel_id}: {e}")
        return False


def encrypt_packet(data: bytes, key: bytes) -> Optional[bytes]:
    """
    Encrypt a packet using AES.
    
    Args:
        data: Packet data bytes
        key: Encryption key bytes (32 bytes)
        
    Returns:
        Encrypted packet bytes or None on error
    """
    return crypto.encrypt_aes(data, key)


def decrypt_packet(encrypted_data: bytes, key: bytes) -> Optional[bytes]:
    """
    Decrypt a packet using AES.
    
    Args:
        encrypted_data: Encrypted packet bytes
        key: Decryption key bytes (32 bytes)
        
    Returns:
        Decrypted packet bytes or None on error
    """
    return crypto.decrypt_aes(encrypted_data, key)


# ============================================================================
# Audio Packet Reception
# ============================================================================

def receive_audio_packet() -> Optional[Tuple[str, bytes]]:
    """
    Receive audio packet from UDP socket (non-blocking).
    
    Returns:
        Tuple of (channel_id, encrypted_data) or None if no packet received
    """
    global global_udp_socket
    
    if global_udp_socket is None:
        return None
    
    try:
        # Receive packet (non-blocking due to setblocking(False))
        data, addr = global_udp_socket.recvfrom(8192)
        
        # Parse JSON message
        try:
            message = json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            # Log first few decode errors for debugging
            static_decode_errors = getattr(receive_audio_packet, '_decode_errors', 0)
            receive_audio_packet._decode_errors = static_decode_errors + 1
            if static_decode_errors < 3:
                print(f"[UDP] ERROR: Failed to parse received packet as JSON (error #{static_decode_errors + 1}): {e}")
                print(f"[UDP] DEBUG: Received {len(data)} bytes from {addr}")
                if len(data) < 100:
                    print(f"[UDP] DEBUG: First 50 bytes: {data[:50]}")
            return None
        
        # Extract channel ID and data
        channel_id = message.get('channel_id', '')
        b64_data = message.get('data', '')
        
        if not channel_id or not b64_data:
            return None
        
        # Decode base64
        encrypted_data = crypto.decode_base64(b64_data)
        if encrypted_data is None:
            print(f"[UDP] ERROR: Failed to decode base64 for channel {channel_id}")
            return None
        
        # Return channel ID and encrypted data (decryption happens elsewhere with correct key)
        return channel_id, encrypted_data
        
    except socket.error as e:
        # Non-blocking socket will raise socket.error (errno.EAGAIN) when no data available
        if e.errno != 11:  # EAGAIN = 11 on Linux (resource temporarily unavailable)
            if not global_interrupted.is_set():
                static_errors = getattr(receive_audio_packet, '_socket_errors', 0)
                receive_audio_packet._socket_errors = static_errors + 1
                if static_errors < 3:
                    print(f"[UDP] ERROR: Socket error receiving packet (error #{static_errors + 1}): {e}")
        return None
    except Exception as e:
        if not global_interrupted.is_set():
            static_errors = getattr(receive_audio_packet, '_general_errors', 0)
            receive_audio_packet._general_errors = static_errors + 1
            if static_errors < 3:
                print(f"[UDP] ERROR: Exception receiving packet (error #{static_errors + 1}): {e}")
        return None


def set_packet_receive_callback(callback: Callable[[str, bytes], None]):
    """
    Set callback function to be called when audio packet is received.
    
    Args:
        callback: Callback function(channel_id, encrypted_data) -> None
    """
    global packet_receive_callback
    packet_receive_callback = callback


# ============================================================================
# Heartbeat Mechanism
# ============================================================================

def send_heartbeat() -> bool:
    """
    Send heartbeat packet to server.
    
    Returns:
        True if heartbeat sent successfully, False otherwise
    """
    global global_udp_socket, global_server_addr
    
    if not is_udp_ready():
        return False
    
    try:
        heartbeat_msg = json.dumps({
            "type": "heartbeat",
            "timestamp": int(time.time())
        })
        
        # Use sendto (socket is non-blocking but sendto should work)
        global_udp_socket.sendto(heartbeat_msg.encode('utf-8'), global_server_addr)
        return True
        
    except socket.error as e:
        # Non-blocking socket might raise error on send if buffer is full
        if e.errno != 11:  # EAGAIN
            print(f"[UDP] ERROR: Failed to send heartbeat: {e}")
        return False
    except Exception as e:
        print(f"[UDP] ERROR: Failed to send heartbeat: {e}")
        return False


def heartbeat_worker():
    """Heartbeat worker thread - sends periodic heartbeat packets."""
    global heartbeat_enabled
    
    print("[UDP] Heartbeat worker started")
    heartbeat_enabled = True
    
    while not global_interrupted.is_set() and heartbeat_enabled:
        if is_udp_ready():
            send_heartbeat()
        time.sleep(heartbeat_interval)
    
    print("[UDP] Heartbeat worker stopped")


def start_heartbeat():
    """Start heartbeat worker thread."""
    global heartbeat_thread, heartbeat_enabled
    
    if heartbeat_thread is not None and heartbeat_thread.is_alive():
        return
    
    heartbeat_enabled = True
    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()
    print("[UDP] Heartbeat worker started")


def stop_heartbeat():
    """Stop heartbeat worker thread."""
    global heartbeat_enabled
    
    heartbeat_enabled = False


# ============================================================================
# UDP Listener Worker Thread
# ============================================================================

def udp_listener_worker():
    """UDP listener worker thread - receives audio packets."""
    global global_udp_socket, packet_receive_callback
    
    print("[UDP] UDP listener worker started")
    
    packet_count = 0
    timeout_count = 0
    last_log_time = time.time()
    
    while not global_interrupted.is_set():
        if global_udp_socket is None:
            time.sleep(0.1)
            continue
        
        try:
            # Check if socket is ready for reading (use select with 0.1s timeout)
            ready, _, _ = select.select([global_udp_socket], [], [], 0.1)
            
            if ready:
                # Socket is ready, try to receive packet(s)
                # Receive multiple packets if available
                packets_received = 0
                while True:
                    result = receive_audio_packet()
                    
                    if result:
                        channel_id, encrypted_data = result
                        packet_count += 1
                        packets_received += 1
                        
                        # Call callback if registered
                        if packet_receive_callback:
                            try:
                                packet_receive_callback(channel_id, encrypted_data)
                            except Exception as e:
                                print(f"[UDP] ERROR: Exception in packet receive callback: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            # Log warning if callback not set
                            static_callback_warn = getattr(udp_listener_worker, '_callback_warned', False)
                            if not static_callback_warn:
                                print("[UDP] WARNING: Packet received but callback not set!")
                                udp_listener_worker._callback_warned = True
                    else:
                        # No more packets available
                        break
                
                # Log when packets are received
                if packets_received > 0:
                    # Reset timeout counter
                    timeout_count = 0
                    
                    # Log first packet and periodically
                    if packet_count == 1:
                        print(f"[UDP] ✅ First packet received! (channel: {result[0] if result else 'unknown'})")
                    elif packet_count % 500 == 0:  # Every ~10 seconds at 50 packets/sec
                        print(f"[UDP] Received {packet_count} packets total")
            else:
                # No data available
                timeout_count += 1
                
                # Log periodically (every 10 seconds)
                current_time = time.time()
                if current_time - last_log_time >= 10.0:
                    last_log_time = current_time
                    if packet_count == 0:
                        print(f"[UDP] Still waiting for packets (elapsed: {timeout_count * 0.1:.1f}s)")
                        if timeout_count >= 100:  # 10 seconds
                            print("[UDP] DEBUG: Checking UDP socket state...")
                            if global_udp_socket:
                                try:
                                    sockname = global_udp_socket.getsockname()
                                    print(f"[UDP] DEBUG: Socket bound to {sockname}")
                                    print(f"[UDP] DEBUG: Server address: {global_server_addr}")
                                    print(f"[UDP] DEBUG: Callback registered: {packet_receive_callback is not None}")
                                except Exception as e:
                                    print(f"[UDP] DEBUG: Error checking socket: {e}")
                    else:
                        # We've received packets before, just log that we're waiting
                        if timeout_count % 100 == 0:  # Every 10 seconds
                            print(f"[UDP] Waiting for more packets (received {packet_count} so far)")
                
        except Exception as e:
            if not global_interrupted.is_set():
                static_errors = getattr(udp_listener_worker, '_loop_errors', 0)
                udp_listener_worker._loop_errors = static_errors + 1
                if static_errors < 5:
                    print(f"[UDP] ERROR: Exception in UDP listener (error #{static_errors + 1}): {e}")
                    import traceback
                    traceback.print_exc()
            time.sleep(0.1)
    
    print(f"[UDP] UDP listener worker stopped (received {packet_count} packets total)")


def start_udp_listener():
    """Start UDP listener worker thread."""
    global udp_listener_thread
    
    if udp_listener_thread is not None and udp_listener_thread.is_alive():
        return
    
    udp_listener_thread = threading.Thread(target=udp_listener_worker, daemon=True)
    udp_listener_thread.start()
    print("[UDP] UDP listener worker started")


def stop_udp_listener():
    """Stop UDP listener worker thread."""
    global udp_listener_thread
    
    # Thread will stop when global_interrupted is set
    pass

