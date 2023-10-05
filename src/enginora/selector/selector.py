from typing import Type

from enginora.selector.base import Selector
from enginora.selector.dummy import DummySelector
from enginora.selector.slicing import SlicingSelector
from enginora.selector.undersampling import RandomUnderSamplerSelector
from enginora.selector.custom import CustomSelector


def get_selector(name: str) -> Type[Selector]:
    selectors = {
        "dummy": DummySelector,
        "slicing": SlicingSelector,
        "random_under_sampler": RandomUnderSamplerSelector,
        "custom": CustomSelector,
    }

    return selectors[name]
