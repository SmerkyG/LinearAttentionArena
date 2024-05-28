import torch

class ModelState:
    def __init__(self):
        self.seq_pos = 0
        self.kv_cache = torch.tensor([])
        self.embed_state = torch.tensor([])
        self.block_states = []

class TimeMixState:
    def __init__(self, wkv_state=torch.tensor([]), shift_state=torch.tensor([])):
        self.wkv_state = wkv_state
        self.shift_state = shift_state
