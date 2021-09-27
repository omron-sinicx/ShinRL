import chex

import shinrl as srl


def test_line_aa():
    rr, cc, val = srl.line_aa(10, 14, 14, 5, 10)
    chex.assert_shape((rr, cc, val), (20,))
