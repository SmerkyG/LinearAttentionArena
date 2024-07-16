import torch

class ModelState:
    def __init__(self):
        self.seq_pos = 0
        self.input_tokens_cache = torch.tensor([])
        self.k_cache = torch.tensor([])
        self.block_states = []

class TimeMixState:
    def __init__(self, wkv_state=torch.tensor([]), shift_state=torch.tensor([])):
        self.wkv_state = wkv_state
        self.shift_state = shift_state

class Shared:
    def __init__(self):
        self.angles = torch.tensor([])
        self.bias_mask = torch.tensor([])
