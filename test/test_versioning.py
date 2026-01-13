import os

def test_model_versions_exist():
    assert os.path.exists("models")
    assert len(os.listdir("models")) > 0
