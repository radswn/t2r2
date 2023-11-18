import pytest

from t2r2.selector import SelectorConfig


def test_invalid_selector():
    selector = {"name": "definitely_invalid_selector"}

    with pytest.raises(ValueError, match="does not exist"):
        _ = SelectorConfig(**selector)
