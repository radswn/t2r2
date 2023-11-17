from t2r2.flow import get_config

def test_config():
    path = "/workspaces/enginora/notebooks/demo/config.yaml"
    cfg = get_config(path)

    assert cfg.mlflow is None
