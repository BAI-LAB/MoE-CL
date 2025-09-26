from typing import Dict, List, Tuple

import torch

from .lora_linear import Linear, get_range_tensor
from .mix_lora import MixtralSparseMoe
from .modelargs import LLMModelArgs, MixConfig, MultiLoraBatchData


class FeedForward(torch.nn.Module):
    def __init__(self, mlp) -> None:
        super().__init__()
        self.mlp_ = mlp  # mlp is LlamaMLP
        self.moes_: MixtralSparseMoe = None  # mix of experts

    def lora_state_dict(self) -> Dict[str, Linear]:
        return self.mlp_.lora_state_dict()

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> Tuple[torch.Tensor, List]:
        return self._mixlora_forward(data, input_args)

    def init_moe_weight(self, args: LLMModelArgs, config: MixConfig, index = None, gate = None, task_classifier = None):
        self.adapter_name = config.adapter_name
        self.moes_ = MixtralSparseMoe(args, config, index)
        if config.adapter_name == "moe-cl":  # 只有 moe-cl 需要初始化 gate 和 task_classifier
            if gate is None:
                torch.nn.init.normal_(self.moes_.gate_.weight, mean=0.0, std=0.02)
            else:
                with torch.no_grad():
                    self.moes_.gate_.weight.copy_(gate)

            if task_classifier is None:
                torch.nn.init.normal_(self.moes_.task_classifier_.weight, mean=0.0, std=0.02)
            else:
                with torch.no_grad():
                    self.moes_.task_classifier_.weight.copy_(task_classifier)

    def _mixlora_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData):
        if self.adapter_name == "moe-cl":
            final_hidden_state = torch.zeros_like(data)
            classifier_probs_ = [None for _ in range(
                len(input_args.lora_batch_data_config_))]
            lora_range = get_range_tensor(data.device, data.shape[0])
            
            for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                start_idx = lora_config.batch_start_idx_
                end_idx = lora_config.batch_end_idx_

                kwargs = {"attention_mask": input_args.attention_masks_}
                current_hidden_states, current_classifier_outputs = self.moes_.forward(self.mlp_, data[start_idx:end_idx], input_args.task_id, **kwargs)
                classifier_probs_[idx] = current_classifier_outputs
            
                final_hidden_state.index_add_(
                    0, lora_range[start_idx:end_idx], current_hidden_states)

            return final_hidden_state, classifier_probs_

        elif self.adapter_name == "mocl":
            final_hidden_state = torch.zeros_like(data)
            lora_range = get_range_tensor(data.device, data.shape[0])

            kwargs = {}
            kwargs["trained_task_ids"] = input_args.trained_task_ids
            kwargs["trained_task_weights"] = input_args.trained_task_weights

            for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                start_idx = lora_config.batch_start_idx_
                end_idx = lora_config.batch_end_idx_

                current_hidden_states = self.moes_.forward(self.mlp_, data[start_idx:end_idx], input_args.task_id, **kwargs)

                final_hidden_state.index_add_(
                    0, lora_range[start_idx:end_idx], current_hidden_states)

            return final_hidden_state, [None]

        else:
            final_hidden_state = torch.zeros_like(data)
            lora_range = get_range_tensor(data.device, data.shape[0])

            for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                start_idx = lora_config.batch_start_idx_
                end_idx = lora_config.batch_end_idx_

                current_hidden_states = self.moes_.forward(self.mlp_, data[start_idx:end_idx], input_args.task_id)

                final_hidden_state.index_add_(
                    0, lora_range[start_idx:end_idx], current_hidden_states)

            return final_hidden_state, [None]
