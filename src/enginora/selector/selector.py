from typing import ClassVar

from enginora.selector.base import Selector
from enginora.selector.dummy import DummySelector
from enginora.selector.undersampling import RandomUnderSamplerSelector
from enginora.selector.slicing import SlicingSelector


def get_selector(name: str) -> Selector:
    selectors = {
        "dummy": DummySelector,
        "random_under_sampler": RandomUnderSamplerSelector,
        "slicing": SlicingSelector,
    }

    return selectors[name]
