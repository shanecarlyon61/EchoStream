"""
MQTT Publisher - Publish events to MQTT broker

This module provides MQTT publishing functionality for events such as
tone detection, user connections, and system status updates.
"""
import paho.mqtt.client as mqtt
import json
import threading
import time
from typing import Optional, Dict, Any
from echostream import global_interrupted


# ============================================================================
# Global State
# ============================================================================

# MQTT client instance
mqtt_client: Optional[mqtt.Client] = None

# MQTT broker configuration
mqtt_broker: Optional[str] = None
mqtt_port: int = 1883
mqtt_connected = False

# MQTT mutex
mqtt_mutex = threading.Lock()


# ============================================================================
# MQTT Initialization
# ============================================================================

def init_mqtt(broker: str, port: int = 1883, client_id: Optional[str] = None) -> bool:
    """
    Initialize MQTT client.
    
    Args:
        broker: MQTT broker hostname or IP address
        port: MQTT broker port (default: 1883)
        client_id: Client ID string (optional, auto-generated if None)
        
    Returns:
        True if initialization successful, False otherwise
    """
    global mqtt_client, mqtt_broker, mqtt_port
    
    if mqtt_client is not None:
        print("[MQTT] MQTT client already initialized")
        return True
    
    try:
        mqtt_broker = broker
        mqtt_port = port
        
        # Create MQTT client
        mqtt_client = mqtt.Client(client_id=client_id)
        
        # Set callbacks
        mqtt_client.on_connect = _on_mqtt_connect
        mqtt_client.on_disconnect = _on_mqtt_disconnect
        mqtt_client.on_publish = _on_mqtt_publish
        
        # Connect to broker
        print(f"[MQTT] Connecting to MQTT broker: {broker}:{port}")
        mqtt_client.connect(broker, port, 60)
        
        # Start network loop in background thread
        mqtt_client.loop_start()
        
        print("[MQTT] MQTT client initialized")
        return True
        
    except Exception as e:
        print(f"[MQTT] ERROR: Failed to initialize MQTT: {e}")
        mqtt_client = None
        return False


def _on_mqtt_connect(client, userdata, flags, rc):
    """MQTT on_connect callback."""
    global mqtt_connected
    
    if rc == 0:
        mqtt_connected = True
        print("[MQTT] Connected to MQTT broker")
    else:
        mqtt_connected = False
        print(f"[MQTT] ERROR: Failed to connect to MQTT broker (rc={rc})")


def _on_mqtt_disconnect(client, userdata, rc):
    """MQTT on_disconnect callback."""
    global mqtt_connected
    
    mqtt_connected = False
    if rc != 0:
        print(f"[MQTT] Unexpected disconnection from MQTT broker (rc={rc})")
    else:
        print("[MQTT] Disconnected from MQTT broker")


def _on_mqtt_publish(client, userdata, mid):
    """MQTT on_publish callback."""
    # Message published successfully (optional logging)
    pass


# ============================================================================
# Event Publishing
# ============================================================================

def publish_event(topic: str, data: Dict[str, Any], qos: int = 0) -> bool:
    """
    Publish an event to MQTT broker.
    
    Args:
        topic: MQTT topic string
        data: Event data dictionary
        qos: Quality of Service level (0, 1, or 2)
        
    Returns:
        True if message published successfully, False otherwise
    """
    global mqtt_client, mqtt_connected
    
    if mqtt_client is None or not mqtt_connected:
        return False
    
    try:
        message = json.dumps(data)
        result = mqtt_client.publish(topic, message, qos=qos)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            return True
        else:
            print(f"[MQTT] ERROR: Failed to publish to {topic} (rc={result.rc})")
            return False
            
    except Exception as e:
        print(f"[MQTT] ERROR: Exception publishing to {topic}: {e}")
        return False


def publish_tone_detected(tone_id: str, tone_def: Optional[Any] = None) -> bool:
    """
    Publish tone detected event.
    
    Args:
        tone_id: Tone ID string
        tone_def: Tone definition object (optional)
        
    Returns:
        True if message published successfully
    """
    data = {
        "event": "tone_detected",
        "tone_id": tone_id,
        "timestamp": int(time.time())
    }
    
    if tone_def:
        data["tone_a_freq"] = getattr(tone_def, 'tone_a_freq', 0.0)
        data["tone_b_freq"] = getattr(tone_def, 'tone_b_freq', 0.0)
    
    return publish_event("echostream/tone_detected", data)


def cleanup_mqtt():
    """Disconnect from MQTT broker and cleanup."""
    global mqtt_client, mqtt_connected
    
    if mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("[MQTT] MQTT client disconnected")
        except Exception as e:
            print(f"[MQTT] ERROR: Exception during MQTT cleanup: {e}")
        finally:
            mqtt_client = None
            mqtt_connected = False

