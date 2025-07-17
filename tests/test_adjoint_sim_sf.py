from adjoint_sim_sf.adjointsolver import DesignBuilder, SymmetricTransmonBuilder

def test_symmetrictransmonbuilder():
    test_width = 0.19971691

    builder = SymmetricTransmonBuilder()
    design = builder.get_design(test_width)
