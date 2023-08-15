from dataclasses import dataclass


@dataclass
class MlFlowConfig:
    experiment_name: str
    tags: dict
    tracking_uri: str
