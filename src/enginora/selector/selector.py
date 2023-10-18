from typing import Type

from enginora.selector.base import Selector
from enginora.selector.dummy import DummySelector
from enginora.selector.slicing import SlicingSelector
from enginora.selector.undersampling import RandomUnderSamplerSelector
from importlib import util
from os import path


def get_selector(name: str) -> Type[Selector]:
    selectors = {"dummy": DummySelector, "slicing": SlicingSelector, "random_under_sampler": RandomUnderSamplerSelector}
    return selectors[name]


def get_custom_selector(module_path: str, class_name: str) -> Type[Selector]:
    module_name, _ = path.splitext(path.basename(module_path))
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
