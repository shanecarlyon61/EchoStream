"""
MQTT client - handles MQTT connection and message publishing
"""
import json
import os
import threading
from typing import Optional
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARNING] paho-mqtt not available, MQTT functionality disabled")

# Global MQTT state
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

def get_device_id_from_config(device_id: str, device_id_size: int) -> bool:
    """Get device ID from config.json"""
    config_path = "/home/will/.an/config.json"
    
    try:
        with open(config_path, 'r') as file:
            json_data = json.load(file)
        
        # Navigate to unique_id
        shadow = json_data.get('shadow', {})
        state = shadow.get('state', {})
        desired = state.get('desired', {})
        unique_id = desired.get('unique_id', '')
        
        if unique_id and len(unique_id) < device_id_size:
            device_id = unique_id
            print(f"Loaded device ID from config: {device_id}")
            return True
    except Exception as e:
        print(f"Error loading device ID from config: {e}")
    
    return False

def init_mqtt(device_id: str, broker_host: str, broker_port: int) -> bool:
    """Initialize MQTT connection"""
    global global_mqtt
    
    if not MQTT_AVAILABLE:
        print("[WARNING] MQTT not available - paho-mqtt library not installed")
        return False
    
    with mqtt_mutex:
        if global_mqtt.initialized:
            print("MQTT already initialized")
            return True
        
        try:
            global_mqtt.device_id = device_id
            global_mqtt.broker_host = broker_host
            global_mqtt.broker_port = broker_port
            
            # Create MQTT client
            global_mqtt.client = mqtt.Client(client_id=device_id)
            
            # Setup TLS if certificates are available
            if global_mqtt.ca_cert_path and global_mqtt.client_cert_path and global_mqtt.client_key_path:
                global_mqtt.client.tls_set(
                    ca_certs=global_mqtt.ca_cert_path,
                    certfile=global_mqtt.client_cert_path,
                    keyfile=global_mqtt.client_key_path
                )
            
            # Connect to broker
            global_mqtt.client.connect(broker_host, broker_port, 60)
            global_mqtt.client.loop_start()
            
            global_mqtt.initialized = True
            global_mqtt.connected = True
            print(f"MQTT initialized and connected to {broker_host}:{broker_port}")
            return True
        except Exception as e:
            print(f"MQTT initialization failed: {e}")
            return False

def mqtt_publish(topic: str, payload: str) -> bool:
    """Publish MQTT message"""
    global global_mqtt
    
    if not MQTT_AVAILABLE or not global_mqtt.client:
        return False
    
    try:
        result = global_mqtt.client.publish(topic, payload)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    except Exception as e:
        print(f"MQTT publish failed: {e}")
        return False

def mqtt_keepalive():
    """Keep MQTT connection alive"""
    global global_mqtt
    
    if not MQTT_AVAILABLE or not global_mqtt.client:
        return
    
    try:
        if global_mqtt.client.is_connected():
            # Paho MQTT handles keepalive automatically
            pass
        else:
            # Try to reconnect
            global_mqtt.client.reconnect()
    except Exception:
        pass

def cleanup_mqtt():
    """Cleanup MQTT connection"""
    global global_mqtt
    
    if not MQTT_AVAILABLE or not global_mqtt.client:
        return
    
    try:
        global_mqtt.client.loop_stop()
        global_mqtt.client.disconnect()
        global_mqtt.connected = False
        global_mqtt.initialized = False
        print("MQTT cleaned up")
    except Exception as e:
        print(f"MQTT cleanup error: {e}")

def publish_new_tone_detection(frequency: float, duration_ms: int, range_hz: int) -> bool:
    """Publish new tone detection message"""
    import time
    if not global_mqtt.connected:
        return False
    
    try:
        device_id = global_mqtt.device_id or "echostream_device"
        topic = f"devices/{device_id}/tone_detection/new_tone"
        
        message = {
            "device_id": device_id,
            "timestamp": int(time.time()),
            "tone_details": {
                "frequency": frequency,
                "duration_ms": duration_ms,
                "range_hz": range_hz
            }
        }
        
        return mqtt_publish(topic, json.dumps(message))
    except Exception as e:
        print(f"Failed to publish new tone detection: {e}")
        return False

def publish_new_tone_pair(tone_a_hz: float, tone_b_hz: float) -> bool:
    """Publish a new unknown tone pair"""
    import time
    if not global_mqtt.connected:
        return False
    
    try:
        device_id = global_mqtt.device_id or "echostream_device"
        topic = f"devices/{device_id}/tone_detection/new_tone_pair"
        
        message = {
            "device_id": device_id,
            "timestamp": int(time.time()),
            "tone_details": {
                "tone_a_hz": tone_a_hz,
                "tone_b_hz": tone_b_hz
            }
        }
        
        return mqtt_publish(topic, json.dumps(message))
    except Exception as e:
        print(f"Failed to publish new tone pair: {e}")
        return False

