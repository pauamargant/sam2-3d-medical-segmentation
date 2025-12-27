from typing import List, Any
from collections.abc import Mapping

def _format_scalar_for_hydra(value: Any) -> Any:
    """Format scalars for Hydra override strings."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return value


def _build_hydra_overrides(prefix: str, overrides: Any, override_prefix: str = "") -> List[str]:
    """
    Flatten a nested dict/list of overrides into Hydra override strings.

    Args:
        prefix: The base path prefix for the override (e.g., "model")
        overrides: The nested dict/list of override values
        override_prefix: Optional prefix to add to each override string (e.g., "+" for new keys)

    Example:
        prefix="model", overrides={"encoder": {"img_size": 1024}}
        -> ["model.encoder.img_size=1024"]
        
        prefix="model", overrides={"use_maskmem_tpos_enc": True}, override_prefix="+"
        -> ["+model.use_maskmem_tpos_enc=true"]
    """
    if overrides is None:
        return []

    # If caller passed {"model": {...}} and prefix="model", unwrap to avoid "model.model.*"
    if isinstance(overrides, Mapping) and prefix == "model" and set(overrides.keys()) == {"model"}:
        overrides = overrides["model"]

    hydra_overrides: List[str] = []

    def _recurse(path: str, obj: Any) -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                _recurse(f"{path}.{k}", v)
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                _recurse(f"{path}.{idx}", v)
        else:
            override_str = f"{path}={_format_scalar_for_hydra(obj)}"
            if override_prefix:
                override_str = f"{override_prefix}{override_str}"
            hydra_overrides.append(override_str)

    _recurse(prefix, overrides)
    return hydra_overrides