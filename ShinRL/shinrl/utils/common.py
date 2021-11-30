from __future__ import annotations

import os
import pathlib
from abc import ABC
from dataclasses import InitVar, asdict, dataclass, field, fields
from functools import update_wrapper
from typing import Any, Callable, Dict, List, Literal, Union, get_type_hints

import numpy as np
import structlog
import torch
import yaml

LOG: structlog.BoundLogger = structlog.get_logger(__name__)

TB = Union[Callable[[], np.ndarray], np.ndarray]


@dataclass
class Config(ABC):
    def __post_init__(self):
        self.check_literals()

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
        self.check_literals()

    def check_literals(self):
        hints = get_type_hints(type(self))
        for f in fields(type(self)):
            name = f.name
            hint = hints[name]
            if hint.__module__ == "typing":
                if hint.__origin__ is Literal:
                    keys = hint.__args__
                    val = getattr(self, name)
                    assert (
                        val in keys
                    ), f"{type(self).__name__}.{name} == '{val}' is invalid. {type(self).__name__}.{name} must be chosen from {keys}"

    def load_from_yaml(self, file_path: str) -> Config:
        with open(file_path, mode="r") as f:
            config = yaml.safe_load(f)
        self.update(config=config)
        return type(self)(**config)

    def save_as_yaml(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            yaml.dump(self.asdict(), f)


class Tables(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getitem__(self, key: str) -> np.ndarray:
        tb = dict.__getitem__(self, key)
        if isinstance(tb, np.ndarray):
            return tb
        elif callable(tb):
            return tb()

    def set_tb(self, key: str, tb: TB) -> None:
        """Register an table to self.history.tbs.

        Args:
            key (str): key of the table
            tb (Any): np.ndarray or a mapping function to be registered.
                Mapping function is useful to automatically generate tables from networks.
        """
        if isinstance(tb, np.ndarray):
            self.__dict__[key] = tb.astype(np.float32)
        elif callable(tb):
            self.__dict__[key] = tb
        else:
            raise ValueError("tb needs to be array or callable.")

    def make_frozen(self) -> Dict[str, np.ndarray]:
        """make a dict of np.ndarray

        Returns:
            Dict[str, np.ndarray]: tables
        """
        tbs = {}
        for key in self.keys():
            tb = self[key]
            assert isinstance(tb, np.ndarray)
            tbs[key] = tb
        return tbs


@dataclass
class History:
    step: int = 0
    epoch: int = 0
    scalars: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    tbs: Tables = field(default_factory=Tables)
    nets: Dict[str, torch.nn.Module] = field(default_factory=dict)
    add_interval: InitVar[int] = 100

    def __post_init__(self, add_interval: int = 100) -> None:
        """The args are not treated as fields
        Args:
            add_interval (int): interval to add a scalar to scalars
        """
        self.add_interval = add_interval

    def add_scalar(self, label: str, val: float) -> None:
        """Add a scalar to self.scalars.

        Args:
            label (str): label of the scalar
            val (float): the scalar to be added.
        """
        assert np.isscalar(val)
        if label not in self.scalars:
            self.scalars[label] = {"x": [], "y": []}
        if len(self.scalars[label]["x"]) > 0:
            prev_step = self.scalars[label]["x"][-1]
            if self.step - prev_step < self.add_interval:
                return None
        self.scalars[label]["x"].append(self.step)
        self.scalars[label]["y"].append(val)

    def set_tb(self, label: str, tb: TB) -> None:
        self.tbs.set_tb(label, tb)

    def set_net(self, label: str, net: torch.nn.Module) -> None:
        """Register a torch.nn.Module to self.history.nets.

        Args:
            label (str): label of the net
            net (torch.nn.Module): the network to be registered
        """
        assert isinstance(net, torch.nn.Module)
        self.nets[label] = net

    def recent_summary(self, step_range: int = 100000) -> Dict[str, Dict[str, float]]:
        """Return the summary of recent [step_range] steps' history.

        Args:
            step_range (int, optional): Recent step size to summarize. Defaults to 100000.

        Returns:
            Dict[str, Dict[str, float]]: Summary of the recent history.
        """
        scalars = {}
        for key in self.scalars.keys():
            steps = np.array(self.scalars[key]["x"])
            vals = np.array(self.scalars[key]["y"])
            if len(vals.shape) == 1 and len(vals) > 0:
                idx = (steps - steps[-1]) >= -step_range
                _avg, _max, _min = vals[idx].mean(), vals[idx].max(), vals[idx].min()
                scalars[key] = {"Average": _avg, "Max": _max, "Min": _min}
        return scalars

    def dump_scalars(self, dir_path: str) -> None:
        """
        Append scalars to [dir_path]/scalars/[name].csv.
        Delete self.scalars after dump_scalars.

        Args:
            dir_path (str): directory path to save
        """
        scalars_path = os.path.join(dir_path, "scalars")
        if not os.path.exists(scalars_path):
            os.makedirs(scalars_path)
        for label, scalars in self.scalars.items():
            xy = np.array([scalars["x"], scalars["y"]]).T
            file_path = os.path.join(scalars_path, label + ".csv")
            with open(file_path, "a") as f:
                np.savetxt(f, xy, fmt="%f", header="x,y", delimiter=",")
        labels = list(self.scalars.keys())
        for label in labels:
            del self.scalars[label]

    def save_attr(self, attr: Literal["tbs", "nets"], dir_path: str) -> None:
        """save attirubtes to [dir_path]/[name].pkl

        Args:
            attr (Literal["tbs", "nets"]): attribute to save
            dir_path (str): directory to save
        """
        if attr == "tbs":
            _attr = self.tbs.make_frozen()
        else:
            _attr = getattr(self, attr)
        path = os.path.join(dir_path, attr + ".pkl")
        torch.save(_attr, path)

    def save_all(self, dir_path: str) -> None:
        """save all information to [dir_path]

        Args:
            dir_path (str): directory to save
        """
        np.savetxt(
            os.path.join(dir_path, "step_epoch.csv"),
            np.array([self.step, self.epoch]),
            fmt="%d",
            header="step,epoch",
        )
        self.dump_scalars(dir_path)
        for attr in ["tbs", "nets"]:
            self.save_attr(attr, dir_path)

    def load_scalars(self, dir_path: str) -> None:
        scalars_path = pathlib.Path(os.path.join(dir_path, "scalars"))
        for file in scalars_path.rglob("*.csv"):
            scalars = np.loadtxt(file, delimiter=",").T
            x, y = scalars[0].tolist(), scalars[1].tolist()
            name = file.name.split(".csv")[0]
            self.scalars[name] = {"x": x, "y": y}

    def load_attr(self, attr: Literal["tbs", "nets"], dir_path: str, device: str = "cpu") -> None:
        with open(os.path.join(dir_path, attr + ".pkl"), mode="rb") as f:
            _attr = torch.load(f, map_location=device)
        if attr == "tbs":
            for key in _attr.keys():
                tb = dict.__getitem__(self.tbs, key)
                if isinstance(tb, np.ndarray):
                    self.tbs[key] = _attr[key]
        else:
            setattr(self, attr, _attr)

    def load_all(self, dir_path: str, device: str = "cpu") -> None:
        step_epoch = np.loadtxt(os.path.join(dir_path, "step_epoch.csv"))
        self.step, self.epoch = step_epoch[0], step_epoch[1]
        self.load_scalars(dir_path)
        for attr in ["tbs", "nets"]:
            self.load_attr(attr, dir_path, device)


class lazy_property(object):
    """
    Copied from https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type: ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value
