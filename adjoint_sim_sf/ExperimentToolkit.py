"""
A collection of useful functions designed to speed up experimental exploration.
"""
import numpy as np

import numpy as np

def grad_area_asymmetry(position_values, grad_values, *, zero_div_returns=1e19-1):
    pv = np.nan_to_num(np.asarray(position_values), nan=0.0, posinf=0.0, neginf=0.0)
    gv = np.nan_to_num(np.asarray(grad_values),   nan=0.0, posinf=0.0, neginf=0.0)

    abs_integral    = np.trapz(np.abs(gv), pv)
    signed_integral = np.trapz(gv, pv)

    if signed_integral == 0.0:
        return zero_div_returns
    return abs_integral / signed_integral
