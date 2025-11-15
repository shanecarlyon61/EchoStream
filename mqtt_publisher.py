
import paho.mqtt.client as mqtt
import json
import threading
import time
from typing import Optional, Dict, Any
from echostream import global_interrupted

mqtt_client: Optional[mqtt.Client] = None

mqtt_broker: Optional[str] = None
mqtt_port: int = 1883
mqtt_connected = False

mqtt_mutex = threading.Lock()

def init_mqtt(broker: str, port: int = 1883, client_id: Optional[str] = None) -> bool:

    global mqtt_client, mqtt_broker, mqtt_port

    if mqtt_client is not None:
        print("[MQTT] MQTT client already initialized")
        return True

    try:
        mqtt_broker = broker
        mqtt_port = port

        mqtt_client = mqtt.Client(client_id=client_id)

        mqtt_client.on_connect = _on_mqtt_connect
        mqtt_client.on_disconnect = _on_mqtt_disconnect
        mqtt_client.on_publish = _on_mqtt_publish

        print(f"[MQTT] Connecting to MQTT broker: {broker}:{port}")
        mqtt_client.connect(broker, port, 60)

        mqtt_client.loop_start()

        print("[MQTT] MQTT client initialized")
        return True

    except Exception as e:
        print(f"[MQTT] ERROR: Failed to initialize MQTT: {e}")
        mqtt_client = None
        return False

def _on_mqtt_connect(client, userdata, flags, rc):
    global mqtt_connected

    if rc == 0:
        mqtt_connected = True
        print("[MQTT] Connected to MQTT broker")
    else:
        mqtt_connected = False
        print(f"[MQTT] ERROR: Failed to connect to MQTT broker (rc={rc})")

def _on_mqtt_disconnect(client, userdata, rc):
    global mqtt_connected

    mqtt_connected = False
    if rc != 0:
        print(f"[MQTT] Unexpected disconnection from MQTT broker (rc={rc})")
    else:
        print("[MQTT] Disconnected from MQTT broker")

def _on_mqtt_publish(client, userdata, mid):

    pass

def publish_event(topic: str, data: Dict[str, Any], qos: int = 0) -> bool:

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

