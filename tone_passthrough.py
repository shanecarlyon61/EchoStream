
import threading
import time
from typing import Optional
from tone_detection import global_tone_detection, get_recording_time_remaining_ms
from config import ToneDefinition, MAX_CHANNELS

def enable_passthrough(tone_def: ToneDefinition, target_channel: Optional[str] = None) -> bool:

    import config

    tone_config = None
    source_channel_idx = -1

    for i in range(MAX_CHANNELS):
        channel_config = config.get_channel_config(i)
        if channel_config and channel_config.valid and channel_config.tone_detect:
            tone_config = channel_config.tone_config
            source_channel_idx = i
            break

    if not tone_config or not tone_config.tone_passthrough:
        print("[TONE_PASSTHROUGH] Passthrough not enabled in config")
        return False

    if target_channel is None:
        target_channel = tone_config.passthrough_channel

    if not target_channel:
        print("[TONE_PASSTHROUGH] No passthrough target channel specified")
        return False

    print("=" * 60)
    print("[TONE_PASSTHROUGH] PASSTHROUGH START")
    print(f"  Tone ID: {tone_def.tone_id}")
    print(f"  Source Channel: {source_channel_idx + 1}")
    print(f"  Target Channel: {target_channel}")
    print(f"  Duration: {tone_def.record_length_ms} ms ({tone_def.record_length_ms / 1000:.1f} seconds)")
    if tone_def.detection_tone_alert:
        print(f"  Alert Type: {tone_def.detection_tone_alert}")
    print("=" * 60)

    with global_tone_detection.mutex:
        global_tone_detection.passthrough_active = True
        global_tone_detection.active_tone_def = tone_def

    if tone_def.record_length_ms > 0:
        from tone_detection import start_recording_timer
        start_recording_timer(tone_def.record_length_ms)
        print(f"[TONE_PASSTHROUGH] Recording timer started: {tone_def.record_length_ms} ms")

    try:
        import audio_stream

        print(f"[TONE_PASSTHROUGH] Audio routing enabled to {target_channel}")
    except ImportError:
        print("[TONE_PASSTHROUGH] WARNING: audio_stream module not available")

    return True

def disable_passthrough():
    with global_tone_detection.mutex:
        was_active = global_tone_detection.passthrough_active
        global_tone_detection.passthrough_active = False
        global_tone_detection.recording_active = False
        active_tone = global_tone_detection.active_tone_def
        global_tone_detection.active_tone_def = None

    if was_active:
        print("=" * 60)
        print("[TONE_PASSTHROUGH] PASSTHROUGH STOP")
        if active_tone:
            print(f"  Tone ID: {active_tone.tone_id}")
            print(f"  Duration completed: {active_tone.record_length_ms} ms")
        print("=" * 60)

        try:
            import audio_stream

            print("[TONE_PASSTHROUGH] Audio routing disabled")
        except ImportError:
            print("[TONE_PASSTHROUGH] WARNING: audio_stream module not available")

    from tone_detection import reset_detection_state
    reset_detection_state()

def is_passthrough_active() -> bool:

    with global_tone_detection.mutex:
        return global_tone_detection.passthrough_active

def get_remaining_time_ms() -> int:

    return get_recording_time_remaining_ms()

def get_active_tone() -> Optional[ToneDefinition]:

    with global_tone_detection.mutex:
        return global_tone_detection.active_tone_def

def check_passthrough_timer():

    if not is_passthrough_active():
        return

    remaining = get_remaining_time_ms()
    if remaining <= 0:
        disable_passthrough()

