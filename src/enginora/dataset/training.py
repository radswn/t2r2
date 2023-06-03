from enginora.dataset.common import *
from sklearn.model_selection import train_test_split
from typing import Tuple


@dataclass
class TrainingConfig(DatasetConfigWithSelectors):
    batch_size: int
    epochs: int
    learning_rate: float
    output_dir: str

    def __post_init__(self):
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv(self.dataset_path, header=None, names=['id', 'text', 'label'])
        X = data[['id', 'text']]
        y = data['label']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        return train_data, val_data