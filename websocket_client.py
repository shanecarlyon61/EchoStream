import threading
import asyncio
import websockets
import json
from typing import Optional, Dict, Any, Callable, List

global_interrupted = threading.Event()

_audio_streams_by_channel: Dict[int, Dict[str, object]] = {}
_channel_output_device_index: Dict[int, int] = {}

try:
    from audio_devices import select_output_device_for_channel, open_output_stream, close_stream
    from gpio_monitor import GPIO_PINS, gpio_states
    _AUDIO_OK = True
except Exception as _e:
    print(f"[WEBSOCKET] WARNING: Audio/GPIO modules not fully available: {_e}")
    _AUDIO_OK = False

global_ws_client: Optional[websockets.WebSocketClientProtocol] = None

ws_connected = False
ws_url: Optional[str] = None

ws_event_loop: Optional[asyncio.AbstractEventLoop] = None

message_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

registered_channels: List[str] = []
pending_register_ids: List[str] = []

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
        try:
            global_ws_client = await websockets.connect(url, subprotocols=["audio-protocol"])
        except TypeError:
            print("[WEBSOCKET] WARNING: websockets library does not support 'subprotocols' argument, trying without.")
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
        if channel_id in pending_register_ids:
            try:
                pending_register_ids.remove(channel_id)
                print(f"[WEBSOCKET] pending_register_ids updated: {len(pending_register_ids)} remaining")
            except ValueError:
                pass
        return True

    return False

def send_transmit_event(channel_id: str, active: bool) -> bool:
    try:
        import time as _time
        event_type = "transmit_started" if active else "transmit_ended"
        payload = {
            event_type: {
                "affiliation_id": "12345",
                "user_name": "EchoStream",
                "agency_name": "TestAgency",
                "channel_id": channel_id,
                "time": int(_time.time())
            }
        }
        message = json.dumps(payload)
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Failed to build transmit payload: {e}")
        return False

    if send_message(message):
        status = "PTT PRESSED" if active else "PTT RELEASED"
        print(f"[WEBSOCKET] Sent transmit event for channel {channel_id} ({status})")
        
        try:
            from udp_player import global_udp_player
            channel_index = None
            if global_udp_player._channel_ids:
                try:
                    channel_index = global_udp_player._channel_ids.index(channel_id)
                except ValueError:
                    pass
            
            if channel_index is not None:
                if active:
                    if global_udp_player.start_transmission_for_channel(channel_index):
                        print(f"[WEBSOCKET] Started audio transmission for channel {channel_id} (index {channel_index})")
                    else:
                        print(f"[WEBSOCKET] WARNING: Failed to start audio transmission for channel {channel_id}")
                else:
                    global_udp_player.stop_transmission_for_channel(channel_index)
                    print(f"[WEBSOCKET] Stopped audio transmission for channel {channel_id} (index {channel_index})")
        except Exception as e:
            print(f"[WEBSOCKET] WARNING: Failed to start/stop audio transmission: {e}")
        
        return True

    return False

def send_connect_message(channel_id: str) -> bool:
    try:
        import time as _time
        payload = {
            "connect": {
                "affiliation_id": "12345",
                "user_name": "EchoStream",
                "agency_name": "TestAgency",
                "channel_id": channel_id,
                "time": int(_time.time())
            }
        }
        message = json.dumps(payload)
    except Exception as e:
        print(f"[WEBSOCKET] ERROR: Failed to build connect payload: {e}")
        return False

    if send_message(message):
        print(f"[WEBSOCKET] Sent connect message for channel {channel_id}")
        return True

    return False

def register_channels(channel_ids: List[str]) -> None:
    for cid in channel_ids:
        if send_connect_message(cid):
            if cid not in registered_channels:
                registered_channels.append(cid)
            if cid in pending_register_ids:
                try:
                    pending_register_ids.remove(cid)
                    print(f"[WEBSOCKET] pending_register_ids updated: {len(pending_register_ids)} remaining")
                except ValueError:
                    pass

def request_register_channel(channel_id: str) -> None:
    if channel_id in registered_channels:
        return
    if not ws_connected or global_ws_client is None:
        if channel_id not in pending_register_ids:
            pending_register_ids.append(channel_id)
            print(f"[WEBSOCKET] Queued channel {channel_id} for registration (pending={len(pending_register_ids)})")
        return
    register_channels([channel_id])

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

def set_output_device_map(ch_idx_to_device: Dict[int, int]) -> None:
    try:
        _channel_output_device_index.clear()
        for k, v in ch_idx_to_device.items():
            _channel_output_device_index[int(k)] = int(v)
        print(f"[AUDIO] Output device map set for {len(_channel_output_device_index)} channel(s)")
    except Exception as e:
        print(f"[AUDIO] WARNING: Failed setting output device map: {e}")

def _ensure_output_stream_for_channel_index(channel_index: int) -> None:
    if not _AUDIO_OK:
        return
    if channel_index in _audio_streams_by_channel:
        return
    device_index = _channel_output_device_index.get(channel_index)
    if device_index is None:
        device_index = select_output_device_for_channel(channel_index)
    if device_index is None:
        print(f"[AUDIO] WARNING: No output device for channel index {channel_index}")
        return
    pa, stream = open_output_stream(device_index)
    if pa is None or stream is None:
        return
    _audio_streams_by_channel[channel_index] = {'pa': pa, 'stream': stream}
    print(f"[AUDIO] Output stream opened on device {device_index} for channel index {channel_index}")

def _close_all_output_streams() -> None:
    if not _AUDIO_OK:
        return
    for ch_idx, bundle in list(_audio_streams_by_channel.items()):
        pa = bundle.get('pa')
        stream = bundle.get('stream')
        try:
            if pa and stream:
                close_stream(pa, stream)  # type: ignore[arg-type]
        except Exception:
            pass
    _audio_streams_by_channel.clear()

def _active_channel_indices_from_gpio() -> List[int]:
    if not _AUDIO_OK:
        return []
    try:
        keys = list(GPIO_PINS.keys())
        actives: List[int] = []
        for g, val in gpio_states.items():
            if val == 0 and g in keys:
                actives.append(keys.index(g))
        return actives
    except Exception:
        return []

class _UDPProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print("[UDP] Async UDP listener started")
    def datagram_received(self, data, addr):
        try:
            for ch_index in _active_channel_indices_from_gpio():
                _ensure_output_stream_for_channel_index(ch_index)
                bundle = _audio_streams_by_channel.get(ch_index)
                if bundle:
                    try:
                        bundle['stream'].write(data)  # type: ignore[index]
                    except Exception as e:
                        print(f"[UDP] WARNING: stream write failed ch={ch_index}: {e}")
        except Exception as e:
            print(f"[UDP] ERROR: datagram handler exception: {e}")
    def error_received(self, exc):
        print(f"[UDP] WARNING: error received: {exc}")
    def connection_lost(self, exc):
        print("[UDP] Async UDP listener stopped")

async def _async_start_udp_listener(loop: asyncio.AbstractEventLoop, udp_port: int, host_hint: str = "") -> None:
    global _udp_transport, _udp_listening_port
    if _udp_transport is not None and _udp_listening_port == udp_port:
        return
    if _udp_transport is not None:
        try:
            _udp_transport.close()
        except Exception:
            pass
        _udp_transport = None
    transport, _ = await loop.create_datagram_endpoint(_UDPProtocol, local_addr=('0.0.0.0', int(udp_port)))
    _udp_transport = transport
    _udp_listening_port = udp_port
    print(f"[UDP] Listening for audio on 0.0.0.0:{udp_port} (server={host_hint})")

def _start_async_udp_listener(loop: asyncio.AbstractEventLoop, udp_port: int, host_hint: str = "") -> None:
    global _udp_task
    try:
        if udp_port <= 0:
            return
        # schedule on the running loop
        _udp_task = loop.create_task(_async_start_udp_listener(loop, udp_port, host_hint))
    except Exception as e:
        print(f"[UDP] ERROR: Failed to schedule async UDP listener on {udp_port}: {e}")

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

                message_text = (message.decode('utf-8', errors='replace')
                                if isinstance(message, (bytes, bytearray)) else message)
                message_text_stripped = message_text.strip()

                if "users_connected" in message_text:
                    print("=" * 60)
                    print("[WEBSOCKET] Raw users_connected message detected:")
                    print(f"  Raw message: {message_text[:500]}")
                    print("=" * 60)

                try:
                    if not message_text_stripped:
                        continue
                    data = json.loads(message_text_stripped)
                except json.JSONDecodeError as e:
                    if "users_connected" not in message_text:
                        print(f"[WEBSOCKET] JSON decode error: {e}")
                        if message_text_stripped:
                            print(f"[WEBSOCKET] Message content: {message_text_stripped[:200]}")
                    continue

                udp_config = parse_udp_config(message_text_stripped)
                if udp_config:
                    if udp_config_callback:
                        try:
                            udp_config_callback(udp_config)
                        except Exception as e:
                            print(f"[WEBSOCKET] ERROR: Exception in UDP config callback: {e}")
                    try:
                        from udp_player import global_udp_player
                        udp_port = int(udp_config.get('udp_port', 0) or 0)
                        udp_host = str(udp_config.get('udp_host', ''))
                        aes_key = str(udp_config.get('aes_key', '') or '')
                        
                        if pending_register_ids:
                            global_udp_player.set_channel_ids(pending_register_ids)
                            print(f"[WEBSOCKET] Set channel IDs for UDP player: {pending_register_ids}")
                        
                        if udp_port > 0:
                            print(f"[WEBSOCKET] Starting UDP player: port={udp_port}, host={udp_host}, aes_key={'SET' if aes_key and aes_key != 'N/A' else 'N/A (will use hardcoded)'}")
                            if global_udp_player.start(udp_port=udp_port, host_hint=udp_host, aes_key_b64=aes_key):
                                print("[WEBSOCKET] UDP ready - starting audio transmission for active channels")
                                if _AUDIO_OK:
                                    channel_ids_to_check = registered_channels if registered_channels else global_udp_player._channel_ids
                                    gpio_keys = list(GPIO_PINS.keys())
                                    for idx, channel_id in enumerate(channel_ids_to_check):
                                        if idx < len(gpio_keys):
                                            gpio_num = gpio_keys[idx]
                                            gpio_state = gpio_states.get(gpio_num, -1)
                                            if gpio_state == 0:
                                                print(f"[WEBSOCKET] Starting audio transmission for channel {channel_id} (GPIO {gpio_num} is ACTIVE, UDP now ready)")
                                                if global_udp_player.start_transmission_for_channel(idx):
                                                    print(f"[WEBSOCKET] ✓ Audio transmission started for channel {channel_id}")
                                                else:
                                                    print(f"[WEBSOCKET] ✗ Failed to start audio transmission for channel {channel_id}")
                                                send_transmit_event(channel_id, True)
                        else:
                            print(f"[WEBSOCKET] WARNING: Invalid UDP port: {udp_port}")
                    except Exception as e:
                        print(f"[WEBSOCKET] ERROR: Failed to start UDP player: {e}")
                        import traceback
                        traceback.print_exc()
                    continue

                if "users_connected" in str(data):
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

        print("[WEBSOCKET] WebSocket connection established - sending connect messages for all configured channels")
        
        try:
            if pending_register_ids:
                print(f"[WEBSOCKET] Sending connect messages for {len(pending_register_ids)} configured channel(s)")
                for ch_id in list(pending_register_ids):
                    if send_connect_message(ch_id):
                        if ch_id not in registered_channels:
                            registered_channels.append(ch_id)
                        print(f"[WEBSOCKET] ✓ Connect message sent for channel {ch_id}")
                    else:
                        print(f"[WEBSOCKET] ✗ Failed to send connect message for channel {ch_id}")
                pending_register_ids.clear()
                print(f"[WEBSOCKET] All connect messages sent. Registered channels: {registered_channels}")
            else:
                print("[WEBSOCKET] WARNING: No channels configured (pending_register_ids is empty)")
        except Exception as e:
            print(f"[WEBSOCKET] ERROR: Failed to send connect messages: {e}")
            import traceback
            traceback.print_exc()

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


def start_websocket(url: str = "wss://audio.redenes.org/ws/", channel_ids: Optional[List[str]] = None) -> threading.Thread:
    if isinstance(channel_ids, list) and channel_ids:
        pending_register_ids.clear()
        pending_register_ids.extend([str(c).strip() for c in channel_ids if str(c).strip()])
        try:
            from udp_player import global_udp_player
            global_udp_player.set_channel_ids(channel_ids)
        except Exception:
            pass
    t = threading.Thread(target=lambda: global_websocket_thread(url), daemon=True)
    t.start()
    return t
