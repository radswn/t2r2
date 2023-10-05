from enginora.selector.base import *
from importlib import util
from os import path


class CustomSelector(Selector):
    def __init__(self, module_path, class_name):
        self.module_path = module_path
        self.class_name = class_name
        self.user_class = self.load_class()

    def load_class(self):
        module_name, _ = path.splitext(path.basename(self.module_path))
        spec = util.spec_from_file_location(module_name, self.module_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        user_class = getattr(module, self.class_name)
        return user_class

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        user_instance = self.user_class()
        if hasattr(user_instance, "select"):
            return user_instance.select(dataset)
        else:
            raise AttributeError(f"The class {self.class_name} does not have a 'select' method.")
