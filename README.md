# T2R2: Train, test, record, repeat - incremental environment for testing AI models

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/radswn/t2r2)

## Description

The aim of the project is to build an environment for iterative training and evaluation of machine learning models using a wide range of modern data analysis techniques. 
The main emphasis is on the speed of moving in the space of possible solutions for the selection of training sets, hyperparameter optimization, and model evaluation. 
In particular, the system being built has the following functionalities: 
- automatic versioning of all configuration settings of the model training process and configuration of training sets, 
- use of the curriculum learning technique to construct the training set, 
- use of the data set cartography technique, 
- the ability to easily connect external libraries for testing data and models.

---

## [Config Description](https://github.com/radswn/t2r2/wiki/Configuration-Explained)

## Environment Setup

[Dev Setup](https://github.com/radswn/t2r2/wiki/Dev-setup) wiki for devcontainer.

To install our package use the following command pip install `pip install t2r2`.
You also need to have libraries with versions specified in `requirements.txt` - use command `pip install -r requirements.txt`.

---

## Quick start

(TODO - enhance later?)

Import t2r2 library

```python
from t2r2 import T2R2
```

Provide a `config.yaml` and data and run your loop

```python
ttrr = T2R2()
ttrr.loop()
```

After Training, Testing and Recording examine your experiment performance and Repeat!

---
