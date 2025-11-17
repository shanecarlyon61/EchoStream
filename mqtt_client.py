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

def init_mqtt(device_id: Optional[str] = None, broker_host: Optional[str] = None, broker_port: int = 8883) -> bool:
    global global_mqtt
    
    if not MQTT_AVAILABLE:
        print("[MQTT] WARNING: paho-mqtt not available, MQTT functionality disabled")
        return False
    
    with mqtt_mutex:
        if global_mqtt.initialized:
            return True
        
        if not device_id:
            device_id = get_device_id_from_config()
            if not device_id:
                print("[MQTT] ERROR: device_id not provided and not found in config.json")
                return False
        
        if not broker_host:
            broker_host = "a1d6e0zlehb0v9-ats.iot.us-west-2.amazonaws.com"
        
        try:
            global_mqtt.device_id = device_id
            global_mqtt.broker_host = broker_host
            global_mqtt.broker_port = broker_port
            
            ca_path, cert_path, key_path = find_certificates()
            global_mqtt.ca_cert_path = ca_path
            global_mqtt.client_cert_path = cert_path
            global_mqtt.client_key_path = key_path
            
            global_mqtt.client = mqtt.Client(client_id=device_id)
            
            if ca_path and cert_path and key_path:
                global_mqtt.client.tls_set(
                    ca_certs=ca_path,
                    certfile=cert_path,
                    keyfile=key_path
                )
                print(f"[MQTT] TLS configured with certificates")
            else:
                print(f"[MQTT] WARNING: Certificates not found, attempting connection without TLS")
            
            print(f"[MQTT] Connecting to {broker_host}:{broker_port}...")
            global_mqtt.client.connect(broker_host, broker_port, 60)
            global_mqtt.client.loop_start()
            
            time.sleep(1)
            
            if global_mqtt.client.is_connected():
                global_mqtt.connected = True
                global_mqtt.initialized = True
                print(f"[MQTT] ✓ Connected to broker at {broker_host}:{broker_port}")
                return True
            else:
                print(f"[MQTT] WARNING: Connection timeout - broker may not be responding")
                global_mqtt.initialized = True
                return False
        except Exception as e:
            print(f"[MQTT] ERROR: Failed to initialize MQTT: {e}")
            global_mqtt.initialized = True
            return False

def mqtt_publish(topic: str, payload: str) -> bool:
    global global_mqtt
    
    if not MQTT_AVAILABLE:
        return False
    
    with mqtt_mutex:
        if not global_mqtt.initialized or not global_mqtt.client:
            return False
        
        try:
            if not global_mqtt.client.is_connected():
                print("[MQTT] Connection lost, attempting to reconnect...")
                global_mqtt.client.reconnect()
                time.sleep(0.5)
                if not global_mqtt.client.is_connected():
                    print("[MQTT] ERROR: Reconnection failed")
                    global_mqtt.connected = False
                    return False
                global_mqtt.connected = True
                print("[MQTT] Reconnected successfully")
            
            result = global_mqtt.client.publish(topic, payload, qos=1)
            result.wait_for_publish(timeout=2.0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                print(f"[MQTT] ERROR: Failed to publish to '{topic}' (rc={result.rc})")
                return False
        except Exception as e:
            print(f"[MQTT] ERROR: Exception during publish: {e}")
            return False

def mqtt_keepalive():
    global global_mqtt
    
    if not MQTT_AVAILABLE or not global_mqtt.client:
        return
    
    try:
        if not global_mqtt.client.is_connected():
            global_mqtt.client.reconnect()
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
            print(f"[MQTT] Attempting to connect to AWS IoT Core: {broker}:{port}")
            if not init_mqtt(device_id, broker, port):
                print("[MQTT] Failed to initialize MQTT connection - tone detection logged but not published")
                return False
        else:
            print("[MQTT] Cannot publish: MQTT not initialized and device_id not available")
            print("[MQTT] Check that 'unique_id' exists in config.json (shadow.state.desired.unique_id)")
            return False
    
    try:
        message_id = generate_uuid()
        timestamp = int(time.time())
        
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
            print(f"[MQTT] ✗ Failed to publish known tone detection to '{topic}'")
        
        return result
    except Exception as e:
        print(f"[MQTT] ERROR: Exception in publish_known_tone_detection: {e}")
        return False

