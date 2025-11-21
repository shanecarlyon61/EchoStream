import json
import os
import threading
import time
import uuid
from typing import Optional, Dict, Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class MQTTState:
    def __init__(self):
        self.client: Optional[mqtt.Client] = None
        self.device_id = ""
        self.broker_host = ""
        self.broker_port = 8883
        self.connected = False
        self.initialized = False
        self.ca_cert_path = ""
        self.client_cert_path = ""
        self.client_key_path = ""


global_mqtt = MQTTState()
mqtt_mutex = threading.Lock()


def get_device_id_from_config() -> Optional[str]:
    config_path = os.path.expanduser("~/.an/config.json")
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        unique_id = cfg.get("unique_id")
        if unique_id:
            return unique_id

        unique_id = (
            cfg.get("shadow", {})
            .get("state", {})
            .get("desired", {})
            .get("unique_id", "")
        )
        if unique_id:
            return unique_id
    except Exception as e:
        print(f"[MQTT] Error loading device ID from config: {e}")
    return None


def find_certificates():
    cert_dir = os.path.expanduser("~/.an")
    ca_cert = os.path.join(cert_dir, "AmazonRootCA1.pem")
    client_cert = os.path.join(cert_dir, "certificate.pem.crt")
    client_key = os.path.join(cert_dir, "private.pem.key")

    ca_path = ca_cert if os.path.exists(ca_cert) else ""
    cert_path = client_cert if os.path.exists(client_cert) else ""
    key_path = client_key if os.path.exists(client_key) else ""

    return ca_path, cert_path, key_path


def _on_connect(client, userdata, flags, rc):
    global global_mqtt
    if rc == 0:
        global_mqtt.connected = True
        print(f"[MQTT] ✓ Connected to broker (rc={rc})")
    else:
        global_mqtt.connected = False
        print(f"[MQTT] ERROR: Connection failed with rc={rc}")


def _on_disconnect(client, userdata, rc):
    global global_mqtt
    global_mqtt.connected = False
    if rc == 0:
        print(f"[MQTT] Disconnected normally (rc={rc})")
    else:
        rc_messages = {
            1: "Protocol version",
            2: "Client identifier rejected",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized",
            7: "Network error"
        }
        rc_msg = rc_messages.get(rc, f"Unknown error")
        print(
            f"[MQTT] Unexpected disconnection (rc={rc}: {rc_msg}), will attempt to reconnect")


def _reconnect_mqtt(within_mutex: bool = False) -> bool:
    global global_mqtt

    if not global_mqtt.client:
        if not within_mutex:
            with mqtt_mutex:
                return _reconnect_mqtt(within_mutex=True)
        return False

    try:
        if global_mqtt.client.is_connected():
            return True

        print("[MQTT] Attempting to reconnect...")

        try:
            global_mqtt.client.reconnect()
            time.sleep(1.0)
            if global_mqtt.client.is_connected():
                print("[MQTT] ✓ Reconnected successfully")
                return True
        except Exception as e:
            print(
                f"[MQTT] reconnect() failed: {e}, re-initializing connection...")

        try:
            global_mqtt.client.loop_stop()
        except Exception:
            pass

        try:
            global_mqtt.client.disconnect()
        except Exception:
            pass

        global_mqtt.client = None
        global_mqtt.connected = False
        global_mqtt.initialized = False

        if within_mutex:
            device_id = global_mqtt.device_id
            broker_host = global_mqtt.broker_host
            broker_port = global_mqtt.broker_port
        else:
            with mqtt_mutex:
                device_id = global_mqtt.device_id
                broker_host = global_mqtt.broker_host
                broker_port = global_mqtt.broker_port

        if not within_mutex:
            return init_mqtt(device_id, broker_host, broker_port)
        else:
            return False
    except Exception as e:
        print(f"[MQTT] ERROR: Reconnection attempt failed: {e}")
        return False


def init_mqtt(
        device_id: Optional[str] = None,
        broker_host: Optional[str] = None,
        broker_port: int = 8883) -> bool:
    global global_mqtt

    if not MQTT_AVAILABLE:
        print("[MQTT] WARNING: paho-mqtt not available, MQTT functionality disabled")
        return False

    with mqtt_mutex:
        if global_mqtt.initialized and global_mqtt.client and global_mqtt.client.is_connected():
            return True

        if not device_id:
            device_id = get_device_id_from_config()
            if not device_id:
                print(
                    "[MQTT] ERROR: device_id not provided and not found in config.json")
                return False

        if not broker_host:
            broker_host = "a1d6e0zlehb0v9-ats.iot.us-west-2.amazonaws.com"

        try:
            if global_mqtt.client:
                try:
                    global_mqtt.client.loop_stop()
                    global_mqtt.client.disconnect()
                except Exception:
                    pass

            global_mqtt.device_id = device_id
            global_mqtt.broker_host = broker_host
            global_mqtt.broker_port = broker_port

            ca_path, cert_path, key_path = find_certificates()
            global_mqtt.ca_cert_path = ca_path
            global_mqtt.client_cert_path = cert_path
            global_mqtt.client_key_path = key_path

            global_mqtt.client = mqtt.Client(
                client_id=device_id,
                clean_session=True,
                protocol=mqtt.MQTTv311
            )
            global_mqtt.client.on_connect = _on_connect
            global_mqtt.client.on_disconnect = _on_disconnect

            if ca_path and cert_path and key_path:
                try:
                    global_mqtt.client.tls_set(
                        ca_certs=ca_path,
                        certfile=cert_path,
                        keyfile=key_path,
                        tls_version=2
                    )
                    print(f"[MQTT] TLS configured with certificates")
                except Exception as e:
                    print(f"[MQTT] ERROR: TLS configuration failed: {e}")
                    import traceback
                    traceback.print_exc()
                    global_mqtt.initialized = True
                    return False
            else:
                print(
                    f"[MQTT] WARNING: Certificates not found, attempting connection without TLS")

            global_mqtt.client.reconnect_delay_set(min_delay=1, max_delay=120)

            # Match C mqtt.c configuration: keepalive=60 (line 105)
            # The C code processes network I/O every 1 second via mqtt_keepalive()
            # In Python, loop_start() runs a background thread that does this
            # automatically
            print(
                f"[MQTT] Connecting to {broker_host}:{broker_port} (keepalive=60s)...")
            try:
                result = global_mqtt.client.connect(
                    broker_host, broker_port, keepalive=60)
                if result != mqtt.MQTT_ERR_SUCCESS:
                    print(
                        f"[MQTT] ERROR: connect() returned error code: {result}")
                    global_mqtt.initialized = True
                    return False
            except Exception as e:
                print(f"[MQTT] ERROR: connect() failed: {e}")
                import traceback
                traceback.print_exc()
                global_mqtt.initialized = True
                return False

            try:
                global_mqtt.client.loop_start()
            except Exception as e:
                print(f"[MQTT] ERROR: loop_start() failed: {e}")
                global_mqtt.initialized = True
                return False

            for i in range(20):
                time.sleep(0.25)
                if global_mqtt.client.is_connected():
                    global_mqtt.connected = True
                    global_mqtt.initialized = True
                    print(
                        f"[MQTT] ✓ Connected to broker at {broker_host}:{broker_port}")
                    time.sleep(0.5)
                    if global_mqtt.client.is_connected():
                        return True
                    else:
                        print(
                            f"[MQTT] WARNING: Connection lost immediately after connect")
                        break

            print(f"[MQTT] WARNING: Connection timeout after 5 seconds")
            if global_mqtt.client.is_connected():
                global_mqtt.connected = True
                global_mqtt.initialized = True
                return True
            else:
                print(
                    f"[MQTT] Connection not established, will retry on next publish")
                global_mqtt.initialized = True
                return False
        except Exception as e:
            print(f"[MQTT] ERROR: Failed to initialize MQTT: {e}")
            import traceback
            traceback.print_exc()
            global_mqtt.initialized = True
            return False


def mqtt_publish(topic: str, payload: str) -> bool:
    global global_mqtt

    if not MQTT_AVAILABLE:
        return False

    with mqtt_mutex:
        needs_reinit = False
        if not global_mqtt.initialized or not global_mqtt.client:
            if global_mqtt.device_id and global_mqtt.broker_host:
                needs_reinit = True
            else:
                return False
        elif global_mqtt.client and not global_mqtt.client.is_connected():
            reconnect_result = _reconnect_mqtt(within_mutex=True)
            if not reconnect_result:
                if not global_mqtt.initialized:
                    needs_reinit = True
                else:
                    return False
            elif not global_mqtt.client or not global_mqtt.client.is_connected():
                if not global_mqtt.initialized:
                    needs_reinit = True
                else:
                    return False

        if needs_reinit:
            device_id = global_mqtt.device_id
            broker_host = global_mqtt.broker_host
            broker_port = global_mqtt.broker_port

    if needs_reinit:
        if not init_mqtt(device_id, broker_host, broker_port):
            return False

    with mqtt_mutex:
        if not global_mqtt.client or not global_mqtt.client.is_connected():
            return False

        try:

            result = global_mqtt.client.publish(topic, payload, qos=1)
            result.wait_for_publish(timeout=2.0)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                print(
                    f"[MQTT] ERROR: Failed to publish to '{topic}' (rc={
                        result.rc})")
                return False
        except Exception as e:
            print(f"[MQTT] ERROR: Exception during publish: {e}")
            return False


def mqtt_keepalive():
    """
    Keep MQTT connection alive by checking connection and reconnecting if needed.
    In C (mqtt.c line 229-261), this is called every 1 second from GPIO monitor.
    It calls mosquitto_loop() to process network I/O.

    In Python with paho-mqtt, loop_start() already runs a background thread for network I/O,
    so we only need to check connection status and reconnect if needed.
    """
    global global_mqtt

    if not MQTT_AVAILABLE or not global_mqtt.client:
        return

    try:
        # Quick connection check
        if not global_mqtt.client.is_connected():
            # Attempt reconnection (matches C mqtt.c reconnection logic)
            try:
                global_mqtt.client.reconnect()
                time.sleep(0.5)  # Give it time to establish
                if global_mqtt.client.is_connected():
                    print("[MQTT] ✓ Keepalive reconnected successfully")
                    global_mqtt.connected = True
            except Exception as e:
                # Connection failed, will retry on next keepalive or publish
                global_mqtt.connected = False
    except Exception:
        pass


def cleanup_mqtt():
    global global_mqtt

    if not MQTT_AVAILABLE or not global_mqtt.client:
        return

    try:
        global_mqtt.client.loop_stop()
        global_mqtt.client.disconnect()
        global_mqtt.connected = False
        global_mqtt.initialized = False
        print("[MQTT] Cleaned up")
    except Exception as e:
        print(f"[MQTT] Cleanup error: {e}")


def generate_uuid() -> str:
    return str(uuid.uuid4())


def publish_known_tone_detection(
    tone_id: str,
    tone_a_hz: float,
    tone_b_hz: float,
    tone_a_duration_ms: int,
    tone_b_duration_ms: int,
    tone_a_range_hz: int,
    tone_b_range_hz: int,
    channel_id: str,
    record_length_ms: int = 0,
    detection_tone_alert: Optional[str] = None
) -> bool:
    if not global_mqtt.initialized or not global_mqtt.client:
        device_id = get_device_id_from_config()
        if device_id:
            broker = "a1d6e0zlehb0v9-ats.iot.us-west-2.amazonaws.com"
            port = 8883
            print(
                f"[MQTT] Attempting to connect to AWS IoT Core: {broker}:{port}")
            if not init_mqtt(device_id, broker, port):
                print(
                    "[MQTT] Failed to initialize MQTT connection - tone detection logged but not published")
                return False
        else:
            print(
                "[MQTT] Cannot publish: MQTT not initialized and device_id not available")
            print(
                "[MQTT] Check that 'unique_id' exists in config.json (shadow.state.desired.unique_id)")
            return False

    try:
        message_id = generate_uuid()
        timestamp = int(time.time() * 1000)

        device_id = get_device_id_from_config()
        if not device_id:
            device_id = global_mqtt.device_id
        if not device_id:
            print("[MQTT] ERROR: Cannot determine device_id (unique_id) for topic")
            return False

        payload = {
            "message_id": message_id,
            "timestamp": timestamp,
            "device_id": device_id,
            "event_type": "defined_tones_detected",
            "tone_details": {
                "tone_a": tone_a_hz,
                "tone_b": tone_b_hz
            }
        }

        topic = f"from/device/{device_id}/tone_detection"
        json_payload = json.dumps(payload, separators=(',', ':'))

        result = mqtt_publish(topic, json_payload)

        if result:
            print(f"[MQTT] ✓ Published known tone detection to '{topic}': "
                  f"A={tone_a_hz:.1f} Hz, B={tone_b_hz:.1f} Hz")
            print(f"[MQTT]   Message payload: {json_payload}")
        else:
            print(
                f"[MQTT] ✗ Failed to publish known tone detection to '{topic}'")

        return result
    except Exception as e:
        print(f"[MQTT] ERROR: Exception in publish_known_tone_detection: {e}")
        return False


def publish_new_tone_pair(tone_a_hz: float, tone_b_hz: float) -> bool:
    if not global_mqtt.initialized or not global_mqtt.client:
        device_id = get_device_id_from_config()
        if device_id:
            broker = "a1d6e0zlehb0v9-ats.iot.us-west-2.amazonaws.com"
            port = 8883
            print(
                f"[MQTT] Attempting to connect to AWS IoT Core: {broker}:{port}")
            if not init_mqtt(device_id, broker, port):
                print(
                    "[MQTT] Failed to initialize MQTT connection - tone pair logged but not published")
                return False
        else:
            print(
                "[MQTT] Cannot publish: MQTT not initialized and device_id not available")
            return False

    try:
        message_id = generate_uuid()
        timestamp = int(time.time() * 1000)

        device_id = get_device_id_from_config()
        if not device_id:
            device_id = global_mqtt.device_id
        if not device_id:
            print("[MQTT] ERROR: Cannot determine device_id (unique_id) for topic")
            return False

        payload = {
            "message_id": message_id,
            "timestamp": timestamp,
            "device_id": device_id,
            "event_type": "new_tone_detected",
            "tone_details": {
                "tone_a": tone_a_hz,
                "tone_b": tone_b_hz
            }
        }

        topic = f"from/device/{device_id}/tone_detection"
        json_payload = json.dumps(payload, separators=(',', ':'))

        result = mqtt_publish(topic, json_payload)

        if result:
            print(f"[MQTT] ✓ Published new tone pair to '{topic}': "
                  f"A={tone_a_hz:.1f} Hz, B={tone_b_hz:.1f} Hz")
            print(f"[MQTT]   Message payload: {json_payload}")
        else:
            print(f"[MQTT] ✗ Failed to publish new tone pair to '{topic}'")

        return result
    except Exception as e:
        print(f"[MQTT] ERROR: Exception in publish_new_tone_pair: {e}")
        return False
