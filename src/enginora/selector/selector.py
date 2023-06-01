from typing import ClassVar

from enginora.selector.base import Selector
from enginora.selector.dummy import DummySelector
from enginora.selector.undersampling import RandomUnderSamplerSelector


def get_selector(name: str) -> ClassVar[Selector]:
    selectors = {
        "dummy": DummySelector,
        "random_under_sampler": RandomUnderSamplerSelector,
    }

    return selectors[name]
