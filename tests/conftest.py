import pytest
from SQDMetal.COMSOL.Model import COMSOL_Model

@pytest.fixture(scope="session", autouse=True)
def init_comsol():
    print("Beginning test.")
    if COMSOL_Model._engine is None:
        COMSOL_Model.init_engine()
    yield
    COMSOL_Model.close_all_models()
