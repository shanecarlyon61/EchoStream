
import websockets
import asyncio
import json
import threading
from typing import Optional, Dict, Any, Callable, List
from echostream import global_interrupted, MAX_CHANNELS

global_ws_client: Optional[websockets.WebSocketClientProtocol] = None

ws_connected = False
ws_url: Optional[str] = None

ws_event_loop: Optional[asyncio.AbstractEventLoop] = None

message_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

registered_channels: List[str] = []

users_connected_count = 0
connected_users: List[Dict[str, Any]] = []

udp_config_callback: Optional[Callable[[Dict[str, Any]], None]] = None

async def connect_to_server_async(url: str) -> bool:

    global global_ws_client, ws_connected, ws_url

    if global_ws_client is not None:
        print("[WEBSOCKET] Already connected to WebSocket server")
        return True

    try:
        print(f"[WEBSOCKET] Connecting to: {url}")
        global_ws_client = await websockets.connect(url)
        ws_connected = True
        ws_url = url
        print("[WEBSOCKET] WebSocket connection established")
        return True

    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Failed to connect to {url}: {e}")
        global_ws_client = None
        ws_connected = False
        return False

def connect_to_server(url: str) -> bool:

    global ws_event_loop

    try:

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:

            if ws_event_loop is not None:
                loop = ws_event_loop
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        if loop.is_running():

            future = asyncio.run_coroutine_threadsafe(connect_to_server_async(url), loop)
            result = future.result(timeout=10.0)
        else:

            result = loop.run_until_complete(connect_to_server_async(url))
        return result
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Exception in connect_to_server: {e}")
        return False

async def disconnect_async():
    global global_ws_client, ws_connected

    if global_ws_client is not None:
        try:
            await global_ws_client.close()
            print("[WEBSOCKET] WebSocket disconnected")
        except Exception as e:
            print(f"[WEBSOCKET] ERROR: Exception during disconnect: {e}")
        finally:
            global_ws_client = None
            ws_connected = False

def disconnect():
    global global_ws_client

    if global_ws_client is None:
        return

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(disconnect_async())
    except RuntimeError:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(disconnect_async())
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Exception in disconnect: {e}")
    finally:
        global_ws_client = None

async def send_message_async(message: str) -> bool:

    global global_ws_client

    if global_ws_client is None or not ws_connected:
        print("[WEBSOCKET] WARNING: Not connected, cannot send message")
        return False

    try:
        await global_ws_client.send(message)
        return True
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Failed to send message: {e}")
        return False

def send_message(message: str) -> bool:

    global global_ws_client, ws_event_loop

    if global_ws_client is None:
        return False

    try:

        if ws_event_loop is not None and ws_event_loop.is_running():

            future = asyncio.run_coroutine_threadsafe(send_message_async(message), ws_event_loop)
            return future.result(timeout=5.0)
        else:

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(send_message_async(message), loop)
                    return future.result(timeout=5.0)
                else:
                    return loop.run_until_complete(send_message_async(message))
            except RuntimeError:

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(send_message_async(message))
                finally:
                    loop.close()
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Exception in send_message: {e}")
        return False

def register_channel(channel_id: str) -> bool:

    if channel_id in registered_channels:
        print(f"[WEBSOCKET] Channel {channel_id} already registered")
        return True

    message = json.dumps({
        "type": "register",
        "channel_id": channel_id
    })

    if send_message(message):
        registered_channels.append(channel_id)
        print(f"[WEBSOCKET] Registered channel {channel_id}")
        return True

    return False

def send_transmit_event(channel_id: str, active: bool) -> bool:

    message = json.dumps({
        "type": "transmit_started" if active else "transmit_stopped",
        "channel_id": channel_id
    })

    if send_message(message):
        status = "PTT PRESSED" if active else "PTT RELEASED"
        print(f"[WEBSOCKET] Sent transmit event for channel {channel_id} ({status})")
        return True

    return False

def send_connect_message(channel_id: str) -> bool:

    message = json.dumps({
        "type": "connect",
        "channel_id": channel_id
    })

    if send_message(message):
        print(f"[WEBSOCKET] Sent connect message for channel {channel_id}")
        return True

    return False

def parse_udp_config(message: str) -> Optional[Dict[str, Any]]:

    try:
        data = json.loads(message)

        if 'udp_port' in data and 'udp_host' in data:
            config = {
                'udp_port': data.get('udp_port'),
                'udp_host': data.get('udp_host'),
                'websocket_id': data.get('websocket_id', data.get('udp_port')),
                'aes_key': data.get('aes_key', 'N/A')
            }
            print(f"[WEBSOCKET] Parsed UDP config: {config}")
            return config

        return None

    except json.JSONDecodeError as e:
        print(f"[WEBSOCKET] ERROR: Failed to parse UDP config: {e}")
        return None
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Exception parsing UDP config: {e}")
        return None

def handle_users_connected(message: Dict[str, Any]) -> None:

    global users_connected_count, connected_users

    users_connected_count += 1

    try:
        users_data = message.get('users_connected', {})
        if isinstance(users_data, dict):
            total_users = users_data.get('total', 0)
            connected_users = users_data.get('connected', [])

            if users_connected_count == 1:
                print("=" * 60)
                print("[WEBSOCKET] Users Connected Event (first occurrence):")
                print(f"  Total users: {total_users}")
                if connected_users:
                    print(f"  Connected users: {len(connected_users)}")
                    for idx, user in enumerate(connected_users[:5]):
                        user_id = user.get('id', user.get('user_id', 'Unknown'))
                        print(f"    User {idx + 1}: {user_id}")
                    if len(connected_users) > 5:
                        print(f"    ... and {len(connected_users) - 5} more")
                print("=" * 60)
            else:

                print(f"[WEBSOCKET] Users Connected Event #{users_connected_count}: {total_users} total user(s) online")
                if connected_users and len(connected_users) > 0:
                    print(f"  Active connections: {len(connected_users)}")
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Exception handling users_connected: {e}")

def register_message_handler(message_type: str, handler: Callable[[Dict[str, Any]], None]):

    message_handlers[message_type] = handler
    print(f"[WEBSOCKET] Registered handler for message type: {message_type}")

def set_udp_config_callback(callback: Callable[[Dict[str, Any]], None]):

    global udp_config_callback
    udp_config_callback = callback

async def websocket_handler_async():
    global global_ws_client, ws_connected

    message_count = 0

    try:
        while not global_interrupted.is_set() and ws_connected:
            if global_ws_client is None:
                break

            try:

                message = await asyncio.wait_for(
                    global_ws_client.recv(),
                    timeout=1.0
                )

                message_count += 1

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:

                    continue

                udp_config = parse_udp_config(message)
                if udp_config:
                    if udp_config_callback:
                        try:
                            udp_config_callback(udp_config)
                        except Exception as e:
                            print(f"[WEBSOCKET] ERROR: Exception in UDP config callback: {e}")
                    continue

                if 'users_connected' in str(data):
                    handle_users_connected(data)
                    continue

                message_type = data.get('type', 'unknown')
                if message_type in message_handlers:
                    try:
                        message_handlers[message_type](data)
                    except Exception as e:
                        print(f"[WEBSOCKET] ERROR: Exception in message handler for {message_type}: {e}")
                else:

                    if message_count % 50 == 0:
                        print(f"[WEBSOCKET] Unhandled message type: {message_type}")

            except asyncio.TimeoutError:

                continue
            except websockets.exceptions.ConnectionClosed:
                print("[WEBSOCKET] Connection closed by server")
                break
            except Exception as e:
                if not global_interrupted.is_set():
                    print(f"[WEBSOCKET] ERROR: Exception in message handler loop: {e}")
                    await asyncio.sleep(0.1)

    except Exception as e:
        if not global_interrupted.is_set():
            print(f"[WEBSOCKET] FATAL: WebSocket handler crashed: {e}")
            import traceback
            traceback.print_exc()

    finally:
        ws_connected = False
        print("[WEBSOCKET] WebSocket handler stopped")

def global_websocket_thread(url: str):

    global ws_event_loop

    print("[WEBSOCKET] Starting WebSocket thread")

    ws_event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_event_loop)

    try:

        try:
            ws_event_loop.run_until_complete(connect_to_server_async(url))
        except Exception as e:
            print(f"[WEBSOCKET] ERROR: Failed to connect: {e}")
            return

        if not ws_connected:
            print("[WEBSOCKET] ERROR: Connection not established, thread exiting")
            return

        try:
            import channel_manager
            channel_manager.register_channels_with_websocket()
        except Exception as e:
            print(f"[WEBSOCKET] WARNING: Failed to register channels: {e}")

        ws_event_loop.run_until_complete(websocket_handler_async())

    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Exception in WebSocket thread: {e}")
        import traceback
        traceback.print_exc()
    finally:

        try:
            if global_ws_client is not None:
                ws_event_loop.run_until_complete(disconnect_async())
        except Exception as e:
            print(f"[WEBSOCKET] ERROR: Exception during disconnect: {e}")
        finally:
            ws_event_loop.close()
            ws_event_loop = None
            print("[WEBSOCKET] WebSocket thread stopped")

