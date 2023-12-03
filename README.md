# T2R2: Train, test, record, repeat - incremental environment for testing AI models

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radswn/t2r2)

## Description

*TODO*

---

## Quick start

---

### Basic training loop

*TODO*

---

### Implement your own selector

We give you an opportunity to use your own selectors.

1. Prepare a class you want to use - it should inherit from `Selector` class from `t2r2.selector`. Implement its `select` method.
2. When declarating your own selector - provide `module_path` as one of the arguments.

Below we present a simple example how to do it.

`config.yaml` part

```yaml
  selectors:
    - name: UserSelector
      args: 
        module_path: ./my_selector.py
```

`my_selector.py` code

```python
import pandas as pd
from t2r2.selector import Selector

class UserSelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[:5]
```

### Curriculum learning

To force specific order in which examples will be passed during training:
```yaml
training:
  curriculum_learning: True
```
Then you also need to provide the `order` column in your training data.

Basically, the examples will be _sorted_ according to order column and won't be shuffled.

You can also use the custom selector to dynamically provide the order of your training examples.
For example, to pass examples in the order of increasing lenght of text:
```python
class ClSelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["order"] = [len(i) for i in dataset["text"]]
        return dataset
```

---
