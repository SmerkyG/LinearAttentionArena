import torch

class TimeMixState:
    def __init__(self, wkv_state=torch.tensor([]), shift_state=torch.tensor([])):
        self.wkv_state = wkv_state
        self.shift_state = shift_state
