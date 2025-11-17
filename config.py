import json
import os
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.an/config.json")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to load config from {path}: {e}")
        return {}


def get_channel_ids(cfg: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    try:
        sw_cfg_list = (
            cfg.get("shadow", {})
               .get("state", {})
               .get("desired", {})
               .get("software_configuration", [])
        )
        if not sw_cfg_list:
            return result
        item = sw_cfg_list[0]
        for key in ("channel_one", "channel_two", "channel_three", "channel_four"):
            ch = item.get(key, {})
            ch_id = ch.get("channel_id")
            if isinstance(ch_id, str) and ch_id.strip():
                result.append(ch_id.strip())
    except Exception:
        pass
    return result


def get_tone_detect_config(cfg: Dict[str, Any]) -> List[Tuple[str, bool]]:
    out: List[Tuple[str, bool]] = []
    try:
        sw_cfg_list = (
            cfg.get("shadow", {})
               .get("state", {})
               .get("desired", {})
               .get("software_configuration", [])
        )
        if not sw_cfg_list:
            return out
        item = sw_cfg_list[0]
        for key in ("channel_one", "channel_two", "channel_three", "channel_four"):
            ch = item.get(key, {})
            ch_id = ch.get("channel_id", "")
            tone_detect = bool(ch.get("tone_detect", False))
            if isinstance(ch_id, str) and ch_id.strip():
                out.append((ch_id.strip(), tone_detect))
    except Exception:
        pass
    return out


def get_frequency_filters(cfg: Dict[str, Any], channel_id: str) -> List[Dict[str, Any]]:
    """Extract frequency filters for a specific channel from config.json"""
    filters: List[Dict[str, Any]] = []
    try:
        sw_cfg_list = (
            cfg.get("shadow", {})
               .get("state", {})
               .get("desired", {})
               .get("software_configuration", [])
        )
        if not sw_cfg_list:
            return filters
        item = sw_cfg_list[0]
        for key in ("channel_one", "channel_two", "channel_three", "channel_four"):
            ch = item.get(key, {})
            ch_id_from_cfg = ch.get("channel_id", "")
            if ch_id_from_cfg != channel_id:
                continue
            tone_detect_config = ch.get("tone_detect_configuration", {})
            filter_frequencies = tone_detect_config.get("filter_frequencies", [])
            for filter_obj in filter_frequencies:
                filter_data = {
                    "filter_id": str(filter_obj.get("filter_id", "")),
                    "frequency": float(filter_obj.get("frequency", 0.0)),
                    "filter_range": int(filter_obj.get("filter_range", 0)),
                    "type": str(filter_obj.get("type", "")),
                }
                if filter_data["filter_id"] and filter_data["frequency"] > 0:
                    filters.append(filter_data)
            break
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to get frequency filters for {channel_id}: {e}")
    return filters


def get_tone_definitions(cfg: Dict[str, Any], channel_id: str) -> List[Dict[str, Any]]:
    """Extract tone definitions for a specific channel from config.json"""
    tone_defs: List[Dict[str, Any]] = []
    try:
        sw_cfg_list = (
            cfg.get("shadow", {})
               .get("state", {})
               .get("desired", {})
               .get("software_configuration", [])
        )
        if not sw_cfg_list:
            return tone_defs
        item = sw_cfg_list[0]
        for key in ("channel_one", "channel_two", "channel_three", "channel_four"):
            ch = item.get(key, {})
            ch_id_from_cfg = ch.get("channel_id", "")
            if ch_id_from_cfg != channel_id:
                continue
            tone_detect_config = ch.get("tone_detect_configuration", {})
            alert_tones = tone_detect_config.get("alert_tones", [])
            if not alert_tones:
                print(f"[CONFIG] DEBUG: No 'alert_tones' found for channel {channel_id}. "
                      f"Keys in tone_detect_config: {list(tone_detect_config.keys())}")
            for tone_obj in alert_tones:
                tone_data = {
                    "tone_id": str(tone_obj.get("tone_id", "")),
                    "tone_a": float(tone_obj.get("tone_a", 0.0)),
                    "tone_b": float(tone_obj.get("tone_b", 0.0)),
                    "tone_a_length_ms": int(tone_obj.get("tone_a_length", 0.0) * 1000),
                    "tone_b_length_ms": int(tone_obj.get("tone_b_length", 0.0) * 1000),
                    "tone_a_range": int(tone_obj.get("tone_a_range", 0)),
                    "tone_b_range": int(tone_obj.get("tone_b_range", 0)),
                    "record_length_ms": int(tone_obj.get("record_length", 0.0) * 1000),
                    "detection_tone_alert": str(tone_obj.get("detection_tone_alert", "")),
                }
                if (tone_data["tone_id"] and 
                    tone_data["tone_a"] > 0 and 
                    tone_data["tone_b"] > 0):
                    tone_defs.append(tone_data)
                    print(f"[CONFIG] Loaded tone definition: ID={tone_data['tone_id']}, "
                          f"A={tone_data['tone_a']:.1f} Hz±{tone_data['tone_a_range']} "
                          f"({tone_data['tone_a_length_ms']}ms), "
                          f"B={tone_data['tone_b']:.1f} Hz±{tone_data['tone_b_range']} "
                          f"({tone_data['tone_b_length_ms']}ms), "
                          f"rec={tone_data['record_length_ms']}ms")
                else:
                    print(f"[CONFIG] WARNING: Skipped invalid tone definition: "
                          f"tone_id='{tone_data['tone_id']}', "
                          f"tone_a={tone_data['tone_a']}, tone_b={tone_data['tone_b']}")
            break
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to get tone definitions for {channel_id}: {e}")
        import traceback
        traceback.print_exc()
    if not tone_defs:
        print(f"[CONFIG] WARNING: No valid tone definitions found for channel {channel_id}")
    return tone_defs


def get_new_tone_config(cfg: Dict[str, Any], channel_id: str) -> Dict[str, Any]:
    result = {
        "detect_new_tones": False,
        "new_tone_length_ms": 1000,
        "new_tone_range_hz": 3
    }
    try:
        sw_cfg_list = (
            cfg.get("shadow", {})
               .get("state", {})
               .get("desired", {})
               .get("software_configuration", [])
        )
        if not sw_cfg_list:
            return result
        item = sw_cfg_list[0]
        for key in ("channel_one", "channel_two", "channel_three", "channel_four"):
            ch = item.get(key, {})
            ch_id_from_cfg = ch.get("channel_id", "")
            if ch_id_from_cfg != channel_id:
                continue
            tone_detect_config = ch.get("tone_detect_configuration", {})
            alert_details = tone_detect_config.get("alert_details", {})
            if alert_details:
                result["detect_new_tones"] = bool(alert_details.get("detect_new_tones", False))
                result["new_tone_length_ms"] = int(alert_details.get("new_tone_length", 1000))
                result["new_tone_range_hz"] = int(alert_details.get("new_tone_range", 3))
            break
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to get new tone config for {channel_id}: {e}")
    return result

