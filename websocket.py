"""
WebSocket client - handles WebSocket connection and message processing
"""
import json
import time
import threading
import websockets
import asyncio
from typing import Optional
from echostream import global_interrupted
import audio
import udp
import crypto

# Global WebSocket state
global_ws_client: Optional[websockets.WebSocketClientProtocol] = None
global_ws_context: Optional[asyncio.AbstractEventLoop] = None
global_config = {'udp_port': 0, 'udp_host': '', 'websocket_id': 0}
global_config_initialized = False

def parse_websocket_config(json_str: str, cfg: dict) -> bool:
    """Parse WebSocket configuration from JSON"""
    try:
        data = json.loads(json_str)
        cfg['udp_port'] = data.get('udp_port', 0)
        cfg['udp_host'] = data.get('udp_host', '')
        cfg['websocket_id'] = data.get('websocket_id', 0)
        
        print(f"UDP Port: {cfg['udp_port']}")
        print(f"UDP Host: {cfg['udp_host']}")
        print(f"WebSocket ID: {cfg['websocket_id']}")
        
        return True
    except Exception as e:
        print(f"Failed to parse WebSocket config: {e}")
        return False

def send_websocket_transmit_event(channel_id: str, is_started: int):
    """Send transmit event via WebSocket"""
    global global_ws_client
    
    if global_ws_client is None:
        print(f"[WARNING] WebSocket not connected, cannot send transmit event for channel {channel_id}")
        return
    
    try:
        event_type = "transmit_started" if is_started else "transmit_ended"
        now = int(time.time())
        
        transmit_msg = {
            event_type: {
                "affiliation_id": "12345",
                "user_name": "EchoStream",
                "agency_name": "TestAgency",
                "channel_id": channel_id,
                "time": now
            }
        }
        
        # Only log occasionally to reduce spam
        static_log_count = getattr(send_websocket_transmit_event, '_log_count', 0)
        send_websocket_transmit_event._log_count = static_log_count + 1
        if static_log_count % 100 == 0:  # Log every 100th call
            print(f"[INFO] Sending {event_type} for channel {channel_id}")
        
        # Send via asyncio if we're in an async context
        if global_ws_context and global_ws_context.is_running():
            asyncio.run_coroutine_threadsafe(
                global_ws_client.send(json.dumps(transmit_msg)),
                global_ws_context
            )
        else:
            # Fallback: try to send synchronously (may not work)
            if static_log_count % 100 == 0:
                print(f"[WARNING] Cannot send WebSocket message - no active event loop")
    except Exception as e:
        print(f"[ERROR] Failed to send WebSocket message for channel {channel_id}: {e}")

async def websocket_handler():
    """WebSocket connection handler"""
    global global_ws_client, global_config, global_config_initialized
    
    ws_url = "wss://audio.redenes.org/ws/"
    print(f"Connecting to: {ws_url} for all channels")
    
    try:
        async with websockets.connect(ws_url) as ws:
            global_ws_client = ws
            print("[INFO] WebSocket connection established for all channels")
            
            # Register all active channels
            print("[INFO] Registering all active channels with WebSocket")
            for i in range(4):
                if audio.channels[i].active:
                    print(f"[INFO] Registering channel {audio.channels[i].audio.channel_id}")
                    send_websocket_transmit_event(audio.channels[i].audio.channel_id, 1)
            
            # Send connect message for all active channels
            for i in range(4):
                if audio.channels[i].active:
                    now = int(time.time())
                    connect_msg = {
                        "connect": {
                            "affiliation_id": "12345",
                            "user_name": "EchoStream",
                            "agency_name": "TestAgency",
                            "channel_id": audio.channels[i].audio.channel_id,
                            "time": now
                        }
                    }
                    
                    print(f"[INFO] Sending connect message for channel {audio.channels[i].audio.channel_id}")
                    await ws.send(json.dumps(connect_msg))
                    print(f"[INFO] Connect message sent successfully for channel {audio.channels[i].audio.channel_id}")
            
            print("[INFO] Waiting for UDP connection info from WebSocket")
            
            # Listen for messages
            message_count = 0
            async for message in ws:
                if global_interrupted.is_set():
                    break
                
                message_count += 1
                
                # Log all WebSocket messages (occasionally to avoid spam)
                if message_count % 10 == 0 or len(str(message)) < 200:
                    print(f"[WEBSOCKET] Received message #{message_count}: {message[:200]}")
                
                try:
                    # Handle empty messages
                    if not message or len(message) == 0:
                        if message_count % 100 == 0:  # Log occasionally
                            print(f"[WEBSOCKET] Received empty message (#{message_count})")
                        continue
                    
                    data = json.loads(message)
                    
                    # Log full message content for important messages
                    if 'udp_host' in str(data) and 'udp_port' in str(data) and 'websocket_id' in str(data):
                        print("=" * 60)
                        print(f"[WEBSOCKET] UDP Connection Info Received:")
                        print(f"  Message: {message}")
                        print(f"  Parsed Data: {json.dumps(data, indent=2)}")
                        print("=" * 60)
                        
                        # Parse the WebSocket configuration
                        if parse_websocket_config(message, global_config):
                            print("Successfully parsed UDP connection info")
                            global_config_initialized = True
                            
                            # Setup UDP connection
                            if udp.setup_global_udp(global_config):
                                print("UDP connection established")
                                
                                # Start transmission for all active channels
                                for i in range(4):
                                    if audio.channels[i].active:
                                        key_b64 = "46dR4QR5KH7JhPyyjh/ZS4ki/3QBVwwOTkkQTdZQkC0="
                                        key_bytes = crypto.decode_base64(key_b64)
                                        if len(key_bytes) == 32:
                                            audio.channels[i].audio.key = list(key_bytes)
                                            print(f"AES key decoded for channel {audio.channels[i].audio.channel_id}")
                                            
                                            if audio.start_transmission_for_channel(audio.channels[i].audio):
                                                print(f"Audio transmission ready for channel {audio.channels[i].audio.channel_id} (waiting for GPIO activation)")
                                        else:
                                            print(f"Key decode failed for channel {audio.channels[i].audio.channel_id}")
                    elif 'users_connected' in str(data):
                        # Track users_connected messages to reduce spam
                        if not hasattr(websocket_handler, '_users_connected_count'):
                            websocket_handler._users_connected_count = 0
                        websocket_handler._users_connected_count += 1
                        count = websocket_handler._users_connected_count
                        
                        # Log full content first time only, then occasionally
                        if count == 1:
                            print("=" * 60)
                            print(f"[WEBSOCKET] Users Connected Message (first occurrence):")
                            print(f"  Message: {message}")
                            print(f"  Parsed Data: {json.dumps(data, indent=2)}")
                            print("=" * 60)
                        elif count % 10 == 0:
                            # Log every 10th message to avoid spam
                            print(f"[WEBSOCKET] Users connected message #{count} received")
                        
                        # This is just an informational message - UDP may or may not be configured yet
                        if udp.global_udp_socket and udp.global_server_addr:
                            if count == 1:
                                print("[WEBSOCKET] Users connected - UDP is configured and ready")
                        else:
                            if count == 1:
                                print("[WEBSOCKET] Users connected - UDP configuration pending")
                    else:
                        # Log other messages occasionally
                        if message_count % 50 == 0:
                            print(f"[WEBSOCKET] Other message (#{message_count}): {json.dumps(data, indent=2)[:300]}")
                            
                except json.JSONDecodeError as e:
                    print(f"[WEBSOCKET] JSON decode error: {e}")
                    print(f"[WEBSOCKET] Message content: {message[:200]}")
                except Exception as e:
                    print(f"[WEBSOCKET] Error processing message: {e}")
                    print(f"[WEBSOCKET] Message: {message[:200]}")
    
    except Exception as e:
        print(f"[ERROR] WebSocket connection error: {e}")
        global_ws_client = None
    finally:
        print("[WARNING] WebSocket closed for all channels")
        global_ws_client = None

def connect_global_websocket() -> bool:
    """Connect to global WebSocket"""
    global global_ws_context
    
    if global_ws_client is not None:
        print("WebSocket already connected")
        return True
    
    print("[INFO] Attempting WebSocket connection...")
    
    # Create new event loop for WebSocket
    global_ws_context = asyncio.new_event_loop()
    
    def run_websocket():
        global_ws_context.run_until_complete(websocket_handler())
    
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    
    # Give it a moment to connect
    time.sleep(1)
    
    return global_ws_client is not None

def global_websocket_thread(arg=None):
    """Global WebSocket thread wrapper"""
    print("Starting global WebSocket thread")
    
    while not global_interrupted.is_set():
        if global_ws_client is None:
            connect_global_websocket()
        time.sleep(1)
    
    print("[INFO] Global WebSocket thread terminated")
    return None

