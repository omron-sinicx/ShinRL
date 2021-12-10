"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import abc

import jax


class DictJittable(metaclass=abc.ABCMeta):
    """
    ABC that can be passed as an arg to a jitted fn.
    Should be used with dict-like class as:
    class Hoge(DictJittable, dict):
        ...
    """

    def __new__(cls, *args, **kwargs):
        try:
            registered_cls = jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            registered_cls = cls  # already registered
        instance = super(DictJittable, cls).__new__(registered_cls)
        return instance

    def tree_flatten(self):
        _dict = {key: self[key] for key in self.keys()}
        return list(_dict.values()), list(_dict.keys())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        keys = aux_data
        obj = cls()
        obj.update(dict(zip(keys, children)))
        return obj
