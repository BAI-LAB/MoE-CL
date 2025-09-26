from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from .model import LLMFeedForward
from .modelargs import LLMModelArgs, MixConfig


class MixtralSparseMoe(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, config: MixConfig, layer_ind: int) -> None:
        super().__init__()

        self.idx = layer_ind
        self.moe = "mixtral"
        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        if config.adapter_name == "moe-cl":
            self.task_classifier_ = torch.nn.Linear(
                args.dim_, config.num_experts_-1, bias=False, device=args.device_, dtype=self.dtype_)  # predicting task labels (4 tasks) using shared representations
            self.gate_ = torch.nn.Linear(
                args.dim_, 2, bias=False, device=args.device_, dtype=self.dtype_)
        self.act_ = ACT2FN[args.hidden_act_]
        self.experts_: int = config.num_experts_ if isinstance(config.num_experts_, int) else config.num_experts_[layer_ind]

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor, task_id, **kwargs) -> Tuple:  # mlp:LlamaMLP
        if self.adapter_name_ == "moe-cl":  # MoE-CL
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

            expert_idxs = [0, task_id]  # 0 is shared expert, and others are for each task
            if hasattr(mlp, "_mixlora_forward"):
                expert_states = mlp._mixlora_forward(self.act_, expert_idxs, hidden_states, input_dtype)
            else:
                raise NotImplementedError("Not implement mixlora forward2!!!")

            router_logits = self.gate_(hidden_states)  # using token-wise gate network
            router_logits = F.softmax(router_logits, dim=1, dtype=self.dtype_)
            expert1_weights = router_logits[:, 0].unsqueeze(1)  # shape: (batch_size * seq_len, 1)
            expert2_weights = router_logits[:, 1].unsqueeze(1)  # shape: (batch_size * seq_len, 1)

            weighted_expert1 = expert1_weights * expert_states[0]  # shape: (batch_size * seq_len, hidden_dim)
            weighted_expert2 = expert2_weights * expert_states[1]  # shape: (batch_size * seq_len, hidden_dim)

            final_hidden_state = weighted_expert1 + weighted_expert2  # fusing two expert states
            final_hidden_state = final_hidden_state.reshape(
                batch_size, sequence_length, hidden_dim).to(input_dtype)

            # predicting task labels using shared representations, sentence-wise environmental label
            shared_state = expert_states[0].reshape(batch_size, sequence_length, hidden_dim)
            attention_mask = kwargs["attention_mask"].to(device=hidden_states.device)
            sequence_lengths = attention_mask.sum(dim=1, keepdim=True)
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(shared_state)
            masked_shared_state = shared_state * mask_expanded
            sum_shared_state = masked_shared_state.sum(dim=1)
            avg_shared_state = sum_shared_state / sequence_lengths

            classifier_logits = self.task_classifier_(avg_shared_state.to(self.dtype_))
            classifier_probs = F.softmax(classifier_logits, dim=1, dtype=self.dtype_)

            return final_hidden_state, classifier_probs
        
        elif self.adapter_name_ == "pertask-ft":  # Pertask-FT
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

            expert_idxs = [0]
            if hasattr(mlp, "_mixlora_forward"):
                expert_states = mlp._mixlora_forward(self.act_, expert_idxs, hidden_states, input_dtype)
            else:
                raise NotImplementedError("Not implement mixlora forward2!!!")

            final_hidden_state = expert_states[0]
            final_hidden_state = final_hidden_state.reshape(batch_size, sequence_length, hidden_dim).to(input_dtype)
            return final_hidden_state

        elif self.adapter_name_ == "sequential-ft-f":  # Sequential FT-F
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

            expert_idxs = [-1]
            if hasattr(mlp, "_mixlora_forward"):
                expert_states = mlp._mixlora_forward(self.act_, expert_idxs, hidden_states, input_dtype)
            else:
                raise NotImplementedError("Not implement mixlora forward2!!!")

            final_hidden_state = expert_states[0]
            final_hidden_state = final_hidden_state.reshape(batch_size, sequence_length, hidden_dim).to(input_dtype)
            return final_hidden_state

        elif self.adapter_name_ == "sequential-ft-p":  # Sequential FT-P
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

            expert_idxs = [0]
            if hasattr(mlp, "_mixlora_forward"):
                expert_states = mlp._mixlora_forward(self.act_, expert_idxs, hidden_states, input_dtype)
            else:
                raise NotImplementedError("Not implement mixlora forward2!!!")

            final_hidden_state = expert_states[0]
            final_hidden_state = final_hidden_state.reshape(batch_size, sequence_length, hidden_dim).to(input_dtype)
            return final_hidden_state

        elif self.adapter_name_ == "mocl":  # MOCL
            trained_task_ids = kwargs.get("trained_task_ids", [])
            trained_task_weights = kwargs.get("trained_task_weights", None)
            trained_task_weights = torch.stack(trained_task_weights, dim=1)
            trained_task_weights = torch.nn.functional.softmax(trained_task_weights, dim=1)
            trained_task_weights = trained_task_weights.unsqueeze(1).unsqueeze(2)

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

            expert_idxs = trained_task_ids + [task_id]
            expert_idxs = [eid - 1 for eid in expert_idxs]
            if hasattr(mlp, "_mixlora_forward"):
                expert_states = mlp._mixlora_forward(self.act_, expert_idxs, hidden_states, input_dtype)
            else:
                raise NotImplementedError("Not implement mixlora forward2!!!")

            for i in range(len(expert_states)):
                expert_states[i] = expert_states[i].reshape(batch_size, sequence_length, hidden_dim)
            stacked_expert_state = torch.stack(expert_states, dim=-1)

            final_hidden_state = (stacked_expert_state * trained_task_weights).sum(dim=-1).to(input_dtype)
            return final_hidden_state

        elif self.adapter_name_ == "olora":  # O-LoRA
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

            expert_idxs = [task_id-1]  # 只激活当前任务的专家
            if hasattr(mlp, "_mixlora_forward"):
                expert_states = mlp._mixlora_forward(self.act_, expert_idxs, hidden_states, input_dtype)
            else:
                raise NotImplementedError("Not implement mixlora forward2!!!")

            final_hidden_state = expert_states[0]
            final_hidden_state = final_hidden_state.reshape(batch_size, sequence_length, hidden_dim).to(input_dtype)
            return final_hidden_state

        else:
            raise NotImplementedError("Not implement adapter name!!!")
