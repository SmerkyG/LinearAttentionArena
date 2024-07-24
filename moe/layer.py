# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Tuple, Any

import torch
from torch import nn, Tensor
from torch.nn import Module, functional as F

from deepspeed.utils import groups, log_dist
from .experts import Experts
from .sharded_moe import MOELayer, TopKGate

from deepspeed.moe.layer import MoE as DeepSpeedMoE

from src.CoreDependencies import *

class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

    def _set_ep_group(self, ep_group):
        pass

# superclass to fool the rest of DeepSpeed into thinking this is their MoE class, so it initializes/loads/saves properly when it checks module instance types
class MoE(DeepSpeedMoE):
    """Initialize an MoE layer.

    Arguments:
        input_size (int): the input dimension.
        expert (nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
    """

    def __init__(self,
                 input_size: int,
                 expert: nn.Module,
                 num_experts: int = 1,
                 ep_size: int = 1,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 4,
                 use_residual: bool = False,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 use_tutel: bool = False,
                 hash_prime: int = 1,
                 enable_expert_tensor_parallelism: bool = False,
                 top2_2nd_expert_sampling: bool = True) -> None:

        super(DeepSpeedMoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}',
            [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = TJIT(Experts(expert, self.num_local_experts, self.expert_group_name))
        # self.deepspeed_moe = MOELayer(TopKGate(input_size, num_experts, k, capacity_factor, eval_capacity_factor,
        #                                        min_capacity, noisy_gate_policy, drop_tokens, use_rts, None,
        #                                        top2_2nd_expert_sampling),
        self.deepspeed_moe = MOELayer(None,
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      hash_prime=hash_prime,
                                      use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = nn.Linear(input_size, 2)
        else:
            self.mlp = Identity()
            self.coefficient = Identity()

    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_: bool = False) -> None:
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    def _create_process_groups(self, use_data_before_expert_parallel_: bool = False) -> None:
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    @torch.jit.ignore
    def forward(self,
                hidden_states: torch.Tensor,
                tokens: torch.Tensor,
                used_token: torch.Tensor) -> torch.Tensor: #Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (Tensor): expert count
        """
        output = self.deepspeed_moe(hidden_states, tokens, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = F.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output
        #return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
