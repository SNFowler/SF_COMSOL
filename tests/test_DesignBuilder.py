import pytest
import numpy as np
from adjoint_sim_sf.DesignBuilder import SymmetricTransmonBuilder  

# Sample width values from offset_vals + 0.19971691
test_widths = np.linspace(-0.05, 0.05, 5) + 0.19971691

@pytest.mark.parametrize("width", test_widths)
def test_get_design_runs_without_error(width):
    builder = SymmetricTransmonBuilder()
    design = builder.get_design(np.array([width]))
    assert design is not None
    assert hasattr(design, 'components') 
