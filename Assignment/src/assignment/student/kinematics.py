"""Kinematics of a serial chain.

The conventions are: links are numbered 0..n, joints 1..n,
joint i connects link i-1 to link i, and the unit twist of
joint i is expressed in the child link frame Psi_i.

The kinematic description of a chain is constant and lives in a `ChainModel`;
the quantities that depend on the configuration live in a `ChainData`, which is
allocated once per chain and refilled by `update_kinematics` at each new
configuration.
"""

import numpy as np

from assignment.kinematics_io import ChainModel
# Add any other necessary imports from lie_groups if needed.



class ChainData:
    """Quantities of a serial chain at one configuration.

    Allocated once for a given chain and refilled in place by
    `update_kinematics`, so that a control loop evaluating the kinematics at
    every step does not keep reallocating. All fields refer to the
    configuration held in `q`, and are meaningless before the first update,
    where `q` is filled with NaN to make that obvious.

    Attributes:
        q: the configuration the other fields were computed at, shape (n,).
        H_rel: H_i^{i-1}(q^i) for each joint, shape (n, 4, 4).
        H_link_w: H_i^w, the link frames relative to the world frame, shape (n+1, 4, 4),
            indexed by the link number 0..n.
        H_ee_w: H_ee^w, the end-effector frame relative to the world frame, shape (4, 4).
        J_ee: the geometric Jacobian expressed in the end-effector frame,
            shape (6, n), with columns T = (omega, v).
    """

    def __init__(self, model: ChainModel):
        self.q = np.full(model.n, np.nan)
        self.H_rel = np.empty((model.n, 4, 4))
        self.H_link_w = np.empty((model.n + 1, 4, 4))
        self.H_ee_w = np.empty((4, 4))
        self.J_ee = np.empty((6, model.n))


def update_kinematics(model: ChainModel, data: ChainData, q) -> None:
    """Fill `data` with the kinematics of `model` at the configuration `q`.

    Args:
        model: the kinematic description of the serial chain.
        data: the container to fill, allocated for this chain.
        q: the joint variables, an array of n numbers.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (model.n,):
        raise ValueError(f"expected {model.n} joint variables, got shape {q.shape}")

    # Fill in your solution here.
    raise NotImplementedError("update_kinematics is not implemented yet")
