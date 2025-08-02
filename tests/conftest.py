# tests/conftest.py
import pytest
from SQDMetal.COMSOL.Model import COMSOL_Model

@pytest.fixture(scope="session", autouse=True)
def init_comsol():
    if COMSOL_Model._engine is None:
        COMSOL_Model.init_engine()
