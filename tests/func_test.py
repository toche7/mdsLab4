import ex3 as hw
import numpy as np
def test_hwfunc1():
    model1, model2 = hw.homework()
    assert np.allclose(model1, 0.09241764560913446, atol=1e-4)


def test_hwfunc2():
    model1, model2 = hw.homework()
    assert np.allclose(model2, 0.6732052768464261, atol=1e-4)

