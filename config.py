import json
import os
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.an/config.json")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load full JSON configuration from ~/.an/config.json (by default).
    Returns an empty dict if file is missing or invalid.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[CONFIG] ERROR: Failed to load config from {path}: {e}")
        return {}


def get_channel_ids(cfg: Dict[str, Any]) -> List[str]:
    """
    Extract channel IDs from the expected structure:
    cfg['shadow']['state']['desired']['software_configuration'][0]
      -> 'channel_one'/'channel_two'/'channel_three'/'channel_four' each with 'channel_id'
    Returns a list of up to 4 channel IDs (missing entries are skipped).
    """
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
    """
    Extract per-channel tone_detect flags in channel order.
    Returns a list of (channel_id, tone_detect_bool).
    """
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


