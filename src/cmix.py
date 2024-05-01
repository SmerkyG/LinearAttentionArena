import torch

class ChannelMixState:
    def __init__(self, shift_state=torch.tensor([])):
        self.shift_state = shift_state
