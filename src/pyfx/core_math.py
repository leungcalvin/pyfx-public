"""A module which holds bindings to mathematical operations. This module is meant to be used on *baseband* data"""


def xy_to_circ(bb):
    assert bb.shape[1] == 2, "Baseband is not dual-polarization!"
    if bb.ndim != 3:
        UserWarning("Data shape is {bb.shape}! Expected 3 dimensions.")
    out = np.zeros_like(bb)
    out[:, 0] = bb[:, 0] + 1j * bb[:, 1]
    out[:, 1] = bb[:, 0] - 1j * bb[:, 1]
    return out
