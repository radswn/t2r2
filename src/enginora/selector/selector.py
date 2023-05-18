from typing import ClassVar

from enginora.selector.base import Selector
from enginora.selector.dummy import DummySelector


def get_selector(name: str) -> ClassVar[Selector]:
    selectors = {
        'DUMMY': DummySelector,
    }

    return selectors[name]
