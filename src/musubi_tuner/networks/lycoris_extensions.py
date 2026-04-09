"""LyCORIS extensions and utilities for musubi-tuner.

Provides enhancements to LyCORIS:
- Special initialization methods
- Integration helpers
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _to_dense_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "dequantize"):
        dequantized = tensor.dequantize()
        if isinstance(dequantized, torch.Tensor):
            return dequantized
    return tensor


def _matched_normal_tensor(inp: torch.Tensor, shape: torch.Size, scale: float = 1e-3) -> torch.Tensor:
    dense_inp = _to_dense_tensor(inp).detach().float()
    device = dense_inp.device
    target = torch.randn(shape, device=device, dtype=torch.float32)

    desired_norm = dense_inp.norm()
    desired_mean = dense_inp.mean()
    desired_std = dense_inp.std()

    current_norm = target.norm().clamp_min(1e-12)
    target = target * (desired_norm / current_norm)
    current_std = target.std().clamp_min(1e-12)
    target = target * (desired_std / current_std)
    target = target - target.mean() + desired_mean
    target.mul_(scale)
    return target


def _factorize_matrix(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A, B such that A @ B approximates matrix."""
    out_dim, in_dim = matrix.shape
    rank = max(1, min(rank, out_dim, in_dim))
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    root_s = torch.sqrt(s.clamp_min(0))
    a = u * root_s.unsqueeze(0)
    b = root_s.unsqueeze(1) * vh
    return a, b


def _init_factorized_ones(left: torch.Tensor, right: torch.Tensor) -> None:
    """Initialize A, B so that A @ B ~= 1 everywhere instead of ~= rank."""
    rank = max(1, right.shape[0])
    fill = 1.0 / math.sqrt(rank)
    left.fill_(fill)
    right.fill_(fill)


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
            if hasattr(lora_module, "lokr_w1"):
                # Initialize w1 to identity (ones)
                if isinstance(lora_module.lokr_w1, nn.Parameter):
                    lora_module.lokr_w1.fill_(1.0)
                    initialized_count += 1
            elif hasattr(lora_module, "lokr_w1_a") and hasattr(lora_module, "lokr_w1_b"):
                _init_factorized_ones(lora_module.lokr_w1_a, lora_module.lokr_w1_b)
                initialized_count += 1

            org_weight = getattr(lora_module, "org_weight", None)
            if not isinstance(org_weight, torch.Tensor):
                logger.warning("LoKR module %s has no org_weight; falling back to plain normal init", getattr(lora_module, "lora_name", "<unknown>"))

            if hasattr(lora_module, "lokr_w2"):
                # Match dense-weight statistics for the full-matrix branch.
                if isinstance(lora_module.lokr_w2, nn.Parameter):
                    if isinstance(org_weight, torch.Tensor):
                        target = _matched_normal_tensor(org_weight, lora_module.lokr_w2.shape, scale=scale)
                        lora_module.lokr_w2.copy_(target.to(dtype=lora_module.lokr_w2.dtype))
                    else:
                        nn.init.normal_(lora_module.lokr_w2, mean=0.0, std=scale)
                    initialized_count += 1
            elif hasattr(lora_module, "lokr_w2_a") and hasattr(lora_module, "lokr_w2_b"):
                if isinstance(org_weight, torch.Tensor) and lora_module.lokr_w2_a.ndim == 2 and lora_module.lokr_w2_b.ndim == 2:
                    dense_target = _matched_normal_tensor(
                        org_weight,
                        torch.Size((lora_module.lokr_w2_a.shape[0], lora_module.lokr_w2_b.shape[1])),
                        scale=scale,
                    )
                    a, b = _factorize_matrix(dense_target, lora_module.lokr_w2_a.shape[1])
                    lora_module.lokr_w2_a.copy_(a.to(dtype=lora_module.lokr_w2_a.dtype))
                    lora_module.lokr_w2_b.copy_(b.to(dtype=lora_module.lokr_w2_b.dtype))
                else:
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
            preset["unet_target_name"] = list(name_algo_map.keys())

        # Collect all target modules
        target_modules = [m for m in config["modules"].keys() if "*" not in m]
        if target_modules:
            # lycoris.kohya expects this key name in apply_preset.
            preset["unet_target_module"] = target_modules
        if name_algo_map:
            # Wildcard patterns should be interpreted with fnmatch, not regex.
            preset["use_fnmatch"] = True

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



