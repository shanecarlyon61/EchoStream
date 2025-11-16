import socket
import threading
import time
import json
from typing import Optional, Tuple, Callable
from echostream import global_interrupted
import crypto

global_udp_socket: Optional[socket.socket] = None
global_server_addr: Optional[Tuple[str, int]] = None
heartbeat_thread: Optional[threading.Thread] = None
udp_listener_thread: Optional[threading.Thread] = None
packet_receive_callback: Optional[Callable[[str, bytes], None]] = None
heartbeat_enabled = False
heartbeat_interval = 10.0

def setup_udp(host: str, port: int, bind_port: int = 0) -> bool:
    global global_udp_socket, global_server_addr
    
    if global_udp_socket is not None:
        print("[UDP] UDP socket already configured")
        return True
    
    try:
        global_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        global_udp_socket.settimeout(None)
        global_server_addr = (host, port)
        print(f"[UDP] Server address: {global_server_addr}")
        
        send_heartbeat()
        
        try:
            local_addr = global_udp_socket.getsockname()
            print(f"[UDP] Socket bound to: {local_addr[0]}:{local_addr[1]}")
        except Exception as e:
            print(f"[UDP] Could not get local socket info: {e}")
        
        print(f"[UDP] UDP socket configured for {host}:{port}")
        
        start_udp_listener()
        start_heartbeat()
        
        return True
        
    except Exception as e:
        print(f"[UDP] ERROR: Failed to setup UDP socket: {e}")
        cleanup_udp()
        return False

def cleanup_udp():
    global global_udp_socket, global_server_addr, heartbeat_enabled
    
    heartbeat_enabled = False
    
    socket_to_close = global_udp_socket
    global_udp_socket = None
    global_server_addr = None
    
    if socket_to_close is not None:
        try:
            socket_to_close.close()
            print("[UDP] UDP socket closed")
        except Exception as e:
            print(f"[UDP] ERROR: Exception closing UDP socket: {e}")

def get_server_address() -> Optional[Tuple[str, int]]:
    return global_server_addr

def is_udp_ready() -> bool:
    return global_udp_socket is not None and global_server_addr is not None

def send_audio_packet(channel_id: str, opus_data: bytes, encryption_key: bytes) -> bool:
    global global_udp_socket, global_server_addr
    
    if not is_udp_ready():
        return False
    
    try:
        encrypted_data = crypto.encrypt_aes(opus_data, encryption_key)
        if encrypted_data is None:
            print(f"[UDP] ERROR: Failed to encrypt audio for channel {channel_id}")
            return False
        
        b64_data = crypto.encode_base64(encrypted_data)
        
        message = {
            "channel_id": channel_id,
            "type": "audio",
            "data": b64_data
        }
        
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        global_udp_socket.sendto(message_bytes, global_server_addr)
        return True
        
    except Exception as e:
        print(f"[UDP] ERROR: Failed to send audio packet for channel {channel_id}: {e}")
        return False

def encrypt_packet(data: bytes, key: bytes) -> Optional[bytes]:
    return crypto.encrypt_aes(data, key)

def decrypt_packet(encrypted_data: bytes, key: bytes) -> Optional[bytes]:
    return crypto.decrypt_aes(encrypted_data, key)

def receive_audio_packet() -> Optional[Tuple[str, bytes]]:
    global global_udp_socket
    
    socket_to_use = global_udp_socket
    if socket_to_use is None:
        return None
    
    try:
        data, addr = socket_to_use.recvfrom(8192)
        
        if not hasattr(receive_audio_packet, '_first_receive_logged'):
            print(f"[UDP] ✅ Socket receiving data! Got {len(data)} bytes from {addr[0]}:{addr[1]}")
            receive_audio_packet._first_receive_logged = True
        
        if len(data) == 0:
            return None
        
        try:
            message = json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            static_decode_errors = getattr(receive_audio_packet, '_decode_errors', 0)
            receive_audio_packet._decode_errors = static_decode_errors + 1
            if static_decode_errors < 3:
                print(f"[UDP] ERROR: Failed to parse JSON (error #{static_decode_errors + 1}): {e}")
                print(f"[UDP] DEBUG: Received {len(data)} bytes from {addr[0]}:{addr[1]}")
                if len(data) < 200:
                    print(f"[UDP] DEBUG: Data preview: {data[:100]}")
            return None
        
        channel_id = message.get('channel_id', '')
        message_type = message.get('type', '')
        b64_data = message.get('data', '')
        
        if message_type != 'audio':
            return None
        
        if not channel_id or not b64_data:
            return None
        
        encrypted_data = crypto.decode_base64(b64_data)
        if encrypted_data is None or len(encrypted_data) == 0:
            static_decode_failures = getattr(receive_audio_packet, '_base64_failures', 0)
            receive_audio_packet._base64_failures = static_decode_failures + 1
            if static_decode_failures < 3:
                print(f"[UDP] ERROR: Failed to decode base64 for channel {channel_id} (error #{static_decode_failures + 1})")
            return None
        
        return channel_id, encrypted_data
        
    except socket.timeout:
        return None
    except (socket.error, OSError) as e:
        errno = getattr(e, 'errno', None)
        if errno in (9, 107, 10035):
            return None
        if errno == 11:
            return None
        if global_interrupted.is_set():
            return None
        static_errors = getattr(receive_audio_packet, '_socket_errors', 0)
        receive_audio_packet._socket_errors = static_errors + 1
        if static_errors < 3:
            print(f"[UDP] ERROR: Socket error receiving packet (error #{static_errors + 1}): {e}")
            import traceback
            traceback.print_exc()
        return None
    except Exception as e:
        if global_interrupted.is_set():
            return None
        static_errors = getattr(receive_audio_packet, '_general_errors', 0)
        receive_audio_packet._general_errors = static_errors + 1
        if static_errors < 3:
            print(f"[UDP] ERROR: Exception receiving packet (error #{static_errors + 1}): {e}")
            import traceback
            traceback.print_exc()
        return None

def set_packet_receive_callback(callback: Callable[[str, bytes], None]):
    global packet_receive_callback
    packet_receive_callback = callback

def send_heartbeat() -> bool:
    global global_udp_socket, global_server_addr
    
    if not is_udp_ready():
        return False
    
    try:
        heartbeat_msg = '{"type":"KEEP_ALIVE"}'
        result = global_udp_socket.sendto(heartbeat_msg.encode('utf-8'), global_server_addr)
        
        if result >= 0:
            return True
        else:
            print(f"[UDP] ERROR: Failed to send heartbeat (result={result})")
            return False
        
    except socket.error as e:
        print(f"[UDP] ERROR: Failed to send heartbeat: {e}")
        return False
    except Exception as e:
        print(f"[UDP] ERROR: Failed to send heartbeat: {e}")
        return False

def heartbeat_worker():
    global heartbeat_enabled
    
    print("[UDP] Heartbeat worker started")
    heartbeat_enabled = True
    
    while not global_interrupted.is_set() and heartbeat_enabled:
        if is_udp_ready():
            send_heartbeat()
        time.sleep(heartbeat_interval)
    
    print("[UDP] Heartbeat worker stopped")

def start_heartbeat():
    global heartbeat_thread, heartbeat_enabled
    
    if heartbeat_thread is not None and heartbeat_thread.is_alive():
        return
    
    heartbeat_enabled = True
    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()
    print("[UDP] Heartbeat worker started")

def stop_heartbeat():
    global heartbeat_enabled
    heartbeat_enabled = False

def udp_listener_worker():
    global global_udp_socket, packet_receive_callback
    
    print("[UDP] UDP listener worker started")
    
    if global_udp_socket is None:
        print("[UDP] ERROR: UDP socket not initialized")
        return
    
    print(f"[UDP] Listening on UDP socket {global_udp_socket.fileno()}...")
    
    try:
        sockname = global_udp_socket.getsockname()
        print(f"[UDP] Socket bound to {sockname[0]}:{sockname[1]}")
        print(f"[UDP] Socket blocking: {global_udp_socket.gettimeout() is None}")
        print(f"[UDP] Ready to receive packets...")
    except Exception as e:
        print(f"[UDP] WARNING: Could not get socket info: {e}")
    
    packet_count = 0
    
    while not global_interrupted.is_set():
        if global_udp_socket is None:
            time.sleep(0.1)
            continue
        
        try:
            result = receive_audio_packet()
            
            if result:
                channel_id, encrypted_data = result
                packet_count += 1
                
                if packet_count == 1:
                    print(f"[UDP] ✅ First packet received! (channel: {channel_id}, {len(encrypted_data)} bytes)")
                
                if packet_count % 1000 == 0:
                    print(f"[UDP] Received {packet_count} packets total")
                
                if packet_receive_callback:
                    try:
                        packet_receive_callback(channel_id, encrypted_data)
                    except Exception as e:
                        print(f"[UDP] ERROR: Exception in callback: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    static_callback_warn = getattr(udp_listener_worker, '_callback_warned', False)
                    if not static_callback_warn:
                        print("[UDP] WARNING: Packet received but callback not set!")
                        udp_listener_worker._callback_warned = True
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            if not global_interrupted.is_set():
                static_errors = getattr(udp_listener_worker, '_loop_errors', 0)
                udp_listener_worker._loop_errors = static_errors + 1
                if static_errors < 10:
                    print(f"[UDP] ERROR: Exception in listener (error #{static_errors + 1}): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                break
    
    print(f"[UDP] UDP listener worker stopped (received {packet_count} packets total)")

def start_udp_listener():
    global udp_listener_thread
    
    if udp_listener_thread is not None and udp_listener_thread.is_alive():
        return
    
    udp_listener_thread = threading.Thread(target=udp_listener_worker, daemon=True)
    udp_listener_thread.start()
    print("[UDP] UDP listener worker started")

def stop_udp_listener():
    pass
