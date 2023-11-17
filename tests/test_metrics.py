from t2r2.metrics import get_metric

import sklearn

def test_metrics():
    metric = {"name": "accuracy_score", "args": {"normalise": True}}

    m_name = metric["name"]
    try:
        m = get_metric(m_name)([0, 0], [0, 1], **metric["args"])
    except KeyError:
        if m_name in dir(sklearn.metrics):
            error_msg = f"metric {m_name} not handled by T2R2"
        else:
            error_msg = f"metric {m_name} does not exist"

        raise ValueError(error_msg)
