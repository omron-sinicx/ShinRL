"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

import enum
from dataclasses import asdict, fields
from typing import Any, Dict, Union, get_type_hints

import chex
import yaml


@chex.dataclass
class Config:
    """Class to store the configuration."""

    def __post_init__(self) -> None:
        self._replace_str_to_enum()

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, config: Union[Dict[str, Any], Config]) -> None:
        """
        Args:
            config (Union[Dict[str, Any], Config]): config with updated attributes
        """
        if isinstance(config, Config):
            config = asdict(config)
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"{key} is an invalid config.")
        self._replace_str_to_enum()

    def _replace_str_to_enum(self) -> None:
        """Replace string configs to Enum if possible."""
        hints = get_type_hints(type(self))
        for f in fields(type(self)):
            name = f.name
            hint = hints[name]
            if issubclass(hint, enum.IntEnum):
                val = getattr(self, name)
                if type(val) is str:
                    try:
                        setattr(self, name, hint[val])
                    except KeyError:
                        raise KeyError(
                            f"{val} is an invalid config. {name} must be chosen from {hint}"
                        )

    def load_from_yaml(self, yaml_path: str) -> Config:
        with open(yaml_path, mode="r") as f:
            config = yaml.safe_load(f)
        self.update(config=config)
        return type(self)(**config)

    def save_as_yaml(self, yaml_path: str) -> None:
        dct = self.asdict()
        for key, val in dct.items():
            if isinstance(val, enum.IntEnum):
                dct[key] = val.name
        with open(yaml_path, "w") as f:
            yaml.dump(dct, f)
