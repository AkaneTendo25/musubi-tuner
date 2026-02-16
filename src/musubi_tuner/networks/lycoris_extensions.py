"""LyCORIS extensions and utilities for musubi-tuner.

Provides enhancements to LyCORIS:
- Special initialization methods
- Integration helpers
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def init_lokr_network_with_perturbed_normal(network, scale: float = 1e-3) -> None:
    """Initialize LoKR network with perturbed normal distribution.

    This helps training stability by starting with small perturbations
    rather than zeros. 

    Args:
        network: LyCORIS network instance
        scale: Standard deviation for perturbation (default: 1e-3)
    """
    if not hasattr(network, 'loras'):
        logger.warning("Network doesn't have 'loras' attribute, skipping LoKR init")
        return

    logger.info(f"Initializing LoKR network with perturbed normal (scale={scale})")

    initialized_count = 0
    with torch.no_grad():
        for lora_module in network.loras:
            # LoKR modules have lokr_w1 and lokr_w2
            if hasattr(lora_module, 'lokr_w1'):
                # Initialize w1 to identity (ones)
                if isinstance(lora_module.lokr_w1, nn.Parameter):
                    lora_module.lokr_w1.fill_(1.0)
                    initialized_count += 1
                elif hasattr(lora_module, 'lokr_w1_a') and hasattr(lora_module, 'lokr_w1_b'):
                    # Factorized form
                    lora_module.lokr_w1_a.fill_(1.0)
                    lora_module.lokr_w1_b.fill_(1.0)
                    initialized_count += 1

            if hasattr(lora_module, 'lokr_w2'):
                # Initialize w2 with small normal perturbation
                if isinstance(lora_module.lokr_w2, nn.Parameter):
                    nn.init.normal_(lora_module.lokr_w2, mean=0.0, std=scale)
                    initialized_count += 1
                elif hasattr(lora_module, 'lokr_w2_a') and hasattr(lora_module, 'lokr_w2_b'):
                    # Factorized form
                    nn.init.normal_(lora_module.lokr_w2_a, mean=0.0, std=scale)
                    nn.init.normal_(lora_module.lokr_w2_b, mean=0.0, std=scale)
                    initialized_count += 1

    logger.info(f"Initialized {initialized_count} LoKR module(s)")


def build_network_kwargs_from_config(
    config: Dict[str, Any],
    base_dim: Optional[int] = None,
    base_alpha: Optional[int] = None,
) -> Dict[str, Any]:
    """Build network kwargs from configuration.

    Converts config format to kwargs that can be passed to
    network.create_arch_network().

    Args:
        config: Network configuration dict
        base_dim: Base network dimension (overrides config)
        base_alpha: Base network alpha (overrides config)

    Returns:
        Network kwargs dict
    """
    kwargs = {}

    # Base algorithm
    if "base_algo" in config:
        kwargs["algo"] = config["base_algo"]

    # Base factor (for LoKR/LoHA)
    if "base_factor" in config:
        kwargs["factor"] = config["base_factor"]

    # Dimension/alpha can override config
    if base_dim is not None:
        kwargs["dim"] = base_dim
    elif "dim" in config:
        kwargs["dim"] = config["dim"]

    if base_alpha is not None:
        kwargs["alpha"] = base_alpha
    elif "alpha" in config:
        kwargs["alpha"] = config["alpha"]

    return kwargs


def config_to_lycoris_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config to LyCORIS apply_preset format.

    Args:
        config: Network configuration dict

    Returns:
        LyCORIS preset dict for apply_preset()
    """
    preset = {}

    if "modules" in config and config["modules"]:
        # Build module_algo_map
        module_algo_map = {}
        name_algo_map = {}

        for module_name, module_config in config["modules"].items():
            # Wildcard patterns go to name_algo_map
            if "*" in module_name:
                name_algo_map[module_name] = module_config
            else:
                module_algo_map[module_name] = module_config

        if module_algo_map:
            preset["module_algo_map"] = module_algo_map
        if name_algo_map:
            preset["name_algo_map"] = name_algo_map

        # Collect all target modules
        target_modules = [m for m in config["modules"].keys() if "*" not in m]
        if target_modules:
            preset["target_module"] = target_modules

    return preset


def get_config_init_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract initialization parameters from config.

    Args:
        config: Network configuration dict

    Returns:
        Dict of initialization parameters
    """
    return config.get("init", {})


def log_network_config(config: Dict[str, Any], logger_instance: Optional[logging.Logger] = None) -> None:
    """Log network configuration for debugging.

    Args:
        config: Network configuration dict
        logger_instance: Optional logger instance
    """
    log = logger_instance or logger

    log.info("=== Network Configuration ===")

    if "description" in config:
        log.info(f"Description: {config['description']}")

    if "base_algo" in config:
        log.info(f"Base algorithm: {config['base_algo']}")

    if "base_factor" in config:
        log.info(f"Base factor: {config['base_factor']}")

    if "modules" in config and config["modules"]:
        log.info("Module-specific settings:")
        for module_name, module_config in config["modules"].items():
            log.info(f"  {module_name}: {module_config}")

    if "init" in config and config["init"]:
        log.info("Initialization settings:")
        for param, value in config["init"].items():
            log.info(f"  {param}: {value}")

    log.info("=" * 30)


def validate_lycoris_available() -> bool:
    """Check if LyCORIS is installed and available.

    Returns:
        True if LyCORIS is available, False otherwise
    """
    try:
        import lycoris
        return True
    except ImportError:
        return False


def get_lycoris_info() -> Dict[str, Any]:
    """Get information about installed LyCORIS.

    Returns:
        Dict with version and available algorithms
    """
    try:
        import lycoris

        info = {
            "installed": True,
            "version": getattr(lycoris, "__version__", "unknown"),
        }

        # Try to get available algorithms
        try:
            from lycoris import list_algorithms
            info["algorithms"] = list_algorithms()
        except (ImportError, AttributeError):
            info["algorithms"] = ["lora", "loha", "lokr", "locon", "ia3"]

        return info

    except ImportError:
        return {
            "installed": False,
            "version": None,
            "algorithms": []
        }



