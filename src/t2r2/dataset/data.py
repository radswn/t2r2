from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str = "./data/"
    output_dir: str = "./results/"
    has_header: bool = True
    text_column_id: int = 0
    label_column_id: int = 1
