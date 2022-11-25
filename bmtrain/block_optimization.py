import torch
from typing_extensions import TypedDict

class BlockOptimization(TypedDict):
    zero_level : int
    offload_parameter : bool
    checkpointing : bool
    offload_hidden_state : bool
    economical_forward : bool
    economical_backward : bool

def validate_boptim(self):
    # Check data type
    assert self["zero_level"] in [2, 3]
    assert type(self["offload_parameter"]) == bool
    assert type(self["checkpointing"]) == bool
    assert type(self["offload_hidden_state"]) == bool
    if "economical_forward" not in self:
        self["economical_forward"] = False
    assert type(self["economical_forward"]) == bool
    if "economical_backward" not in self:
        self["economical_backward"] = True
    assert type(self["economical_backward"]) == bool

    # Check conflicts
    if (not self["checkpointing"]) and self["offload_hidden_state"]:
        raise ValueError("Non-checkpointing conflicts with hidden state offloading.")

    return self


def max_block_optim():
    return BlockOptimization(
        zero_level = 3,
        offload_parameter = True,
        checkpointing = True,
        offload_hidden_state = True,
        economical_forward = True,
        economical_backward = True,
    )
