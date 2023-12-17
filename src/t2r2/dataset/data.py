from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str = "./data/"
    output_dir: str = "./results/"
    random_state: int = None
    has_header: bool = False
    text_column_id: 0
    label_column_id: 1
