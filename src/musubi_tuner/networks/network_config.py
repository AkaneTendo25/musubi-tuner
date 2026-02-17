"""Network configuration parser and auto-configuration.

Handles:
- TOML config file parsing (musubi-native format)
- Enhanced --network_args parsing
- Auto-configuration based on model/mode
"""

from typing import Dict, Any, Optional, List
import os
import logging

logger = logging.getLogger(__name__)


def parse_toml_config(config_path: str) -> Dict[str, Any]:
    """Parse network configuration from TOML file.

    Format:
        [network]
        base_algo = "lokr"
        base_factor = 16

        [network.modules.Attention]
        algo = "lokr"
        factor = 16

        [network.init]
        lokr_norm = 1e-3

    Args:
        config_path: Path to TOML config file

    Returns:
        Configuration dict
    """
    try:
        import tomli as tomllib
    except ImportError:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            raise ImportError(
                "TOML support requires 'tomli' package. Install with: pip install tomli"
            )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Network config not found: {config_path}")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    if "network" not in data:
        raise ValueError(f"TOML config must have [network] section: {config_path}")

    config = data["network"]

    # Convert TOML structure to recipe format
    recipe = {
        "description": config.get("description", "TOML config"),
        "base_algo": config.get("base_algo", "lora"),
        "modules": {},
        "init": {}
    }

    if "base_factor" in config:
        recipe["base_factor"] = config["base_factor"]

    # Extract module configs
    if "modules" in config:
        recipe["modules"] = config["modules"]

    # Extract init params
    if "init" in config:
        recipe["init"] = config["init"]

    return recipe


def parse_network_args_enhanced(network_args: Optional[List[str]]) -> Dict[str, Any]:
    """Parse network args with enhanced support for nested keys.

    Supports:
    - Simple: "algo=lokr", "factor=16"
    - Nested: "modules.Attention.factor=16"
    - Init: "init.lokr_norm=1e-3"

    Args:
        network_args: List of "key=value" strings

    Returns:
        Parsed configuration dict
    """
    if not network_args:
        return {}

    config = {}

    for arg in network_args:
        if "=" not in arg:
            logger.warning(f"Ignoring invalid network arg (no '='): {arg}")
            continue

        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to parse value as appropriate type
        parsed_value = _parse_value(value)

        # Handle nested keys
        if "." in key:
            config[key] = parsed_value
        else:
            # Simple key-value for backward compatibility
            config[key] = parsed_value

    return config


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate Python type.

    Args:
        value: String value

    Returns:
        Parsed value (int, float, bool, or str)
    """
    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try bool
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Return as string
    return value


def auto_configure_network(
    ltx2_checkpoint: str,
    ltx2_mode: str,
    network_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Auto-configure network based on model and training mode.

    Detects:
    - Model caption_channels (LTXAV=3840, LTXV=1920)
    - Training mode (video/av/audio)
    - Optimal factor based on model size

    Args:
        ltx2_checkpoint: Path to LTX-2 checkpoint
        ltx2_mode: Training mode (v/av/audio)
        network_dim: Base network dimension (optional)

    Returns:
        Auto-configuration dict
    """
    config = {"modules": {}, "init": {}}

    # Detect caption_channels
    caption_channels = _detect_caption_channels(ltx2_checkpoint)

    if caption_channels is not None:
        logger.info(f"Auto-config: Detected caption_channels={caption_channels}")

        # Auto-factor based on model size
        if caption_channels == 3840:  # LTXAV
            config["base_factor"] = 16
            logger.info("Auto-config: LTXAV model detected, using factor=16")
        elif caption_channels == 1920:  # LTXV
            config["base_factor"] = 12
            logger.info("Auto-config: LTXV model detected, using factor=12")

    # Audio mode adjustments
    if ltx2_mode in ["av", "audio"]:
        logger.info("Auto-config: Audio mode detected, adding higher rank for audio modules")
        dim = network_dim or 64
        config["modules"]["*audio*"] = {
            "algo": "lora", "dim": dim, "alpha": dim // 2
        }

    # Clean up empty dicts so merge_configs skips them
    if not config["modules"]:
        del config["modules"]
    if not config["init"]:
        del config["init"]

    return config


def _detect_caption_channels(checkpoint_path: str) -> Optional[int]:
    """Detect caption_channels from LTX-2 checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        caption_channels value or None if detection fails
    """
    try:
        from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

        with MemoryEfficientSafeOpen(checkpoint_path) as handle:
            # Look for caption_projection.linear_1.weight
            key = "caption_projection.linear_1.weight"
            if key in handle.keys():
                meta = handle.header.get(key)
                if isinstance(meta, dict) and "shape" in meta:
                    shape = meta["shape"]
                    if len(shape) >= 2:
                        # in_features is shape[1] for linear layers
                        return shape[1]

        logger.warning("Could not detect caption_channels from checkpoint")
        return None

    except Exception as e:
        logger.warning(f"Failed to detect caption_channels: {e}")
        return None


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dicts (later ones override earlier).

    Args:
        *configs: Configuration dicts to merge

    Returns:
        Merged configuration
    """
    import copy

    result = {}

    for config in configs:
        if not config:
            continue

        result = _deep_merge(result, copy.deepcopy(config))

    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts.

    Args:
        base: Base dict
        override: Override dict

    Returns:
        Merged dict
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def validate_network_config(config: Dict[str, Any]) -> None:
    """Validate network configuration.

    Args:
        config: Network configuration dict

    Raises:
        ValueError: If configuration is invalid
    """
    valid_algos = ["lora", "loha", "lokr", "locon", "ia3", "full"]

    # Check base_algo
    if "base_algo" in config:
        if config["base_algo"] not in valid_algos:
            raise ValueError(
                f"Invalid base_algo: {config['base_algo']}. "
                f"Valid options: {', '.join(valid_algos)}"
            )

    # Check module configs
    if "modules" in config:
        for module_name, module_config in config["modules"].items():
            if "algo" in module_config:
                if module_config["algo"] not in valid_algos:
                    raise ValueError(
                        f"Invalid algo for {module_name}: {module_config['algo']}"
                    )

            # Validate algo-specific params
            if module_config.get("algo") == "lora":
                if "dim" in module_config and module_config["dim"] <= 0:
                    raise ValueError(f"Invalid dim for {module_name}: must be > 0")

            elif module_config.get("algo") in ["lokr", "loha"]:
                if "factor" in module_config and module_config["factor"] <= 0:
                    raise ValueError(f"Invalid factor for {module_name}: must be > 0")
