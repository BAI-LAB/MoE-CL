import json
import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from mlora.backends import get_backend
from mlora.common import (LLMDecoder, LLMForCausalLM, LLMModelArgs,
                          LLMModelOutput, LLMOutput, LoraConfig, MixConfig,
                          MultiLoraBatchData, lora_config_factory)
from mlora.models import from_pretrained
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from mlora.utils import BitsAndBytesConfig


class CasualOutputLayer(nn.Module):
    def __init__(self, vocab_size: int, weight: torch.nn.Linear):
        super().__init__()
        self.vocab_size_: int = vocab_size
        self.lm_head_: torch.nn.Linear = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()

    def loss(
        self,
        input_ids: torch.Tensor,
        output_logits: torch.Tensor,
        labels,
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=torch.long, device=output_logits.device)
            )
        else:
            labels = torch.tensor(labels, dtype=torch.long, device=output_logits.device)
        # loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(
            output_logits[..., :-1, :].contiguous().view(-1, self.vocab_size_),
            labels[..., 1:].contiguous().view(-1),
        )  # Output_logits is the next token of every position, only need to calculate of the output tokens


class ClassificationOutputLayer(nn.Module):
    def __init__(
        self,
        task_type: str,
        num_labels: int,
        label_dtype: torch.dtype,
        hidden_size: int,
        pad_token_id: int,
        device: str,
        weight: Optional[torch.Tensor],
    ):
        super().__init__()
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type
        self.pad_id_ = pad_token_id
        self.score_ = torch.nn.Linear(
            hidden_size,
            self.num_labels_,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        if weight is None:
            torch.nn.init.kaiming_normal_(self.score_.weight, a=math.sqrt(5))
        else:
            with torch.no_grad():
                self.score_.weight.copy_(weight)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.score_(data.to(torch.float32))

    def loss(
        self, input_ids: torch.Tensor, output_logits: torch.Tensor, labels
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=self.label_dtype_, device=output_logits.device)
            )
        else:
            labels = torch.tensor(
                labels, dtype=self.label_dtype_, device=output_logits.device
            )
        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(
            output_logits.device
        )
        pooled_logits = output_logits[
            torch.arange(batch_size, device=output_logits.device), sequence_lengths
        ]
        if self.task_type_ == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            a =  loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
            return a
        elif self.task_type_ == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)
        else:
            raise ValueError(f"unknown task type {self.task_type_}")


class MTL5ClassificationOutputLayer(nn.Module):
    def __init__(
        self,
        task_type: str,
        task_name: str,
        label_dtype: torch.dtype,
        hidden_size: int,
        pad_token_id: int,
        device: str,
        weight: Optional[torch.Tensor],
    ):
        super().__init__()
        self.label_dtype_ = label_dtype
        self.task_name_ = task_name  # 确定需要训练的任务，明确标签数
        self.task_type_ = task_type
        self.pad_id_ = pad_token_id
        
        if self.task_name_ == "agnews":
            self.num_labels_ = 4
        elif self.task_name_ == "amazon":
            self.num_labels_ = 5
        elif self.task_name_ == "dbpedia":
            self.num_labels_ = 14
        elif self.task_name_ == "yahoo":
            self.num_labels_ = 10
        else:
            raise ValueError(f"unknown task name {self.task_name_}")
        
        # 4个线性分类函数
        self.agnews_score_ = torch.nn.Linear(
            hidden_size,
            4,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.amazon_score_ = torch.nn.Linear(
            hidden_size,
            5,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.dbpedia_score_ = torch.nn.Linear(
            hidden_size,
            14,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.yahoo_score_ = torch.nn.Linear(
            hidden_size,
            10,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        
        if weight is None:
            torch.nn.init.kaiming_normal_(self.agnews_score_.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_normal_(self.amazon_score_.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_normal_(self.dbpedia_score_.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_normal_(self.yahoo_score_.weight, a=math.sqrt(5))
        else:
            with torch.no_grad():
                # 全部加载权重
                self.agnews_score_.weight.copy_(weight["agnews"])
                self.amazon_score_.weight.copy_(weight["amazon"])
                self.dbpedia_score_.weight.copy_(weight["dbpedia"])
                self.yahoo_score_.weight.copy_(weight["yahoo"])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # 保证推理的时候使用特定任务的输出
        if self.task_name_ == "agnews":
            return self.agnews_score_(data.to(torch.float32))
        elif self.task_name_ == "amazon":
            return self.amazon_score_(data.to(torch.float32))
        elif self.task_name_ == "dbpedia":
            return self.dbpedia_score_(data.to(torch.float32))
        elif self.task_name_ == "yahoo":
            return self.yahoo_score_(data.to(torch.float32))
        else:
            raise ValueError(f"unknown task name {self.task_name_}")

    def loss(
        self, input_ids: torch.Tensor, output_logits: torch.Tensor, labels
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=self.label_dtype_, device=output_logits.device)
            )
        else:
            labels = torch.tensor(
                labels, dtype=self.label_dtype_, device=output_logits.device
            )
        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(
            output_logits.device
        )
        pooled_logits = output_logits[
            torch.arange(batch_size, device=output_logits.device), sequence_lengths
        ]
        if self.task_type_ == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            a =  loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
            return a
        elif self.task_type_ == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)
        else:
            raise ValueError(f"unknown task type {self.task_type_}")


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_: Union[CasualOutputLayer, ClassificationOutputLayer] = None

    def forward(
        self, data: torch.Tensor, input_args: MultiLoraBatchData
    ) -> List[LLMModelOutput]:
        outputs = []
        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            layer = self.layers_
            outputs.append(
                LLMModelOutput(
                    adapter_name=adapter_name,
                    logits=layer.forward(data[start_idx:end_idx]),
                    loss_fn_=layer.loss,
                )
            )
        return outputs


def init_lora_layer_weight(
    layer: LLMDecoder,
    args: LLMModelArgs,
    config: Union[LoraConfig, MixConfig],
    index,
    weight: Optional[Dict[str, torch.Tensor]],
):
    target = config.target_modules_
    linear_layer_list = []
    linear_layer_name_list = []
    for name, module in layer.lora_state_dict().items():
        linear_layer_list.append(module)
        linear_layer_name_list.append(name)

    # Initialize gate and task classifier weight
    if config.adapter_name == "moe-cl":
        gate_layer_name = (
            f"seq_module_.layer{layer.layer_id_}.wrapper_module_.mlp_.moes_.gate_.weight"
        )
        task_classifier_layer_name = f"seq_module_.layer{layer.layer_id_}.wrapper_module_.mlp_.moes_.task_classifier_.weight"
        layer.mlp_.init_moe_weight(
            args=args,
            config=config,
            index=index,
            gate=None if weight is None else weight[gate_layer_name],
            task_classifier=None if weight is None else weight[task_classifier_layer_name]
        )
    else:
        layer.mlp_.init_moe_weight(
            args=args,
            config=config,
            index=index,
            gate=None,
            task_classifier=None
        )

    moe_layer_name_list = list(layer.mlp_.lora_state_dict().keys())

    for idx, layer_name in enumerate(linear_layer_name_list):
        if layer_name in target and target[layer_name]:
            if layer_name in moe_layer_name_list:
                for expert_idx in range(
                    config.num_experts_
                    if isinstance(config.num_experts_, int)
                    else config.num_experts_[index]
                ):
                    lora_a = None
                    lora_b = None
                    if weight is not None:
                        map_table = {
                            "w1_proj": "w1_",
                            "w2_proj": "w2_",
                            "w3_proj": "w3_",
                        }
                        lora_a_name = f"seq_module_.layer{layer.layer_id_}.wrapper_module_.mlp_.mlp_.{map_table[layer_name]}.loras_.expert{expert_idx}.lora_a_.weight"
                        lora_b_name = f"seq_module_.layer{layer.layer_id_}.wrapper_module_.mlp_.mlp_.{map_table[layer_name]}.loras_.expert{expert_idx}.lora_b_.weight"

                        if lora_a_name not in weight:
                            raise f"can not found the layer {lora_a_name} in model"
                        if lora_b_name not in weight:
                            raise f"can not found the layer {lora_b_name} in model"

                        lora_a = weight[lora_a_name]
                        lora_b = weight[lora_b_name]

                    linear_layer_list[idx].init_lora_weight(
                        config.expert_config(expert_idx), (lora_a, lora_b), expert_idx
                    )
            else:
                lora_a = None
                lora_b = None
                if weight is not None:
                    map_table = {
                        "q_proj": "wq_",
                        "k_proj": "wk_",
                        "v_proj": "wv_",
                        "o_proj": "wo_",
                    }
                    lora_a_name = f"seq_module_.layer{layer.layer_id_}.wrapper_module_.self_attn_.{map_table[layer_name]}.loras_.expert0.lora_a_.weight"
                    lora_b_name = f"seq_module_.layer{layer.layer_id_}.wrapper_module_.self_attn_.{map_table[layer_name]}.loras_.expert0.lora_b_.weight"

                    if lora_a_name not in weight:
                        raise f"can not found the layer {lora_a_name} in model"
                    if lora_b_name not in weight:
                        raise f"can not found the layer {lora_b_name} in model"

                    lora_a = weight[lora_a_name]
                    lora_b = weight[lora_b_name]

                linear_layer_list[idx].init_lora_weight(config, (lora_a, lora_b))


class OutputSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        output = self.wrapper_module_.forward(input[0], input[2])
        return (output,) + input[1:]


class LLMModel(torch.nn.Module):
    def __init__(self, model: LLMForCausalLM):
        super().__init__()
        args: LLMModelArgs = model.config_
        if args.vocab_size_ >= torch.finfo(args.dtype_).max:
            logging.warn(
                f"vocab_size >= max({args.dtype_}), consider load model with higher precision."
            )
        self.model_ = model
        self.config_ = args

        # configs
        self.name_or_path_ = args.name_or_path_
        self.vocab_size_ = args.vocab_size_
        self.device_ = args.device_
        self.dtype_ = args.dtype_

        output_ = OutputLayer()
        seq_module = model.sequential_module()
        seq_module.update({"output": OutputSequentialWrapper(output_)})
        seq_module.move_to_end("output")
        self.seq_module_ = torch.nn.Sequential(seq_module)

        # adapter configs
        self.adapter_configs_: Dict[str, LoraConfig] = {}

    # compute the model: output probs
    def forward(self, input_args: MultiLoraBatchData) -> List[LLMModelOutput]:
        assert input_args.batch_tokens_ is not None

        labels = input_args.batch_labels_  # labels shape: (batch_size, seq_len)
        if isinstance(input_args.batch_tokens_, torch.Tensor):
            tokens = input_args.batch_tokens_.to(dtype=torch.int64, device=self.device_)
        else:
            tokens = torch.tensor(
                input_args.batch_tokens_, dtype=torch.int64, device=self.device_
            )  # tokens shape: (batch_size, seq_len)

        # prepare mask
        # if input_args.attention_masks_ is not None and 0 in input_args.attention_masks_:
        if input_args.attention_masks_ is not None:
            # 2d mask is passed through the layers
            if isinstance(input_args.attention_masks_, torch.Tensor):
                attn_mask = input_args.attention_masks_.to(
                    dtype=torch.int64, device=self.device_
                )
            else:
                attn_mask = torch.tensor(
                    input_args.attention_masks_, dtype=torch.int64, device=self.device_
                )
        else:
            # print("糟了！！！")
            # print(input_args.attention_masks_)
            attn_mask = None

        if self.config_.attn_implementation_ != "flash_attn":
            causal_mask = self.model_.causal_mask(
                input_tokens=tokens,
                additional_mask=input_args.attention_masks_,
                diagonal=input_args.diagonal_pos_,
            )
        else:
            causal_mask = attn_mask

        input_args.batch_labels_ = None
        input_args.batch_tokens_ = None
        # input_args.attention_masks_ = None  # Used to calculate sentence length

        data = (tokens, causal_mask, input_args)

        if "moe-cl" in self.adapter_configs_:
            for seq_layer in self.seq_module_:
                data = seq_layer.forward(data)

            # collecting classifier probs
            classifier_probs = [[] for _ in range(len(input_args.lora_batch_data_config_))]
            for seq_layer in self.seq_module_:
                if not hasattr(seq_layer, "classifier_probs_"):
                    continue
                for idx in range(len(input_args.lora_batch_data_config_)):
                    if seq_layer.classifier_probs_[idx] is not None:
                        classifier_probs[idx].append(seq_layer.classifier_probs_[idx])

            for idx in range(len(input_args.lora_batch_data_config_)):
                if classifier_probs[idx]:
                    classifier_probs[idx] = torch.cat(classifier_probs[idx], dim=0)

            # record batch start and end indices in output data for the generate method
            output = data[0]
            for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                output_data = output[idx]
                assert isinstance(output_data, LLMModelOutput)
                output_data.batch_start_idx_ = lora_config.batch_start_idx_
                output_data.batch_end_idx_ = lora_config.batch_end_idx_

            # only calculate loss in training mode
            if input_args.is_train:
                assert isinstance(output, List)
                for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                    output_data = output[idx]
                    assert isinstance(output_data, LLMModelOutput)
                    start_idx = lora_config.batch_start_idx_
                    end_idx = lora_config.batch_end_idx_

                    # main loss
                    if self.benchmark == "mtl5":
                        main_loss = output_data.loss_fn_(
                            tokens[start_idx:end_idx],
                            output_data.logits,
                            labels[start_idx:end_idx],
                        )
                        # loss_mask = (1 - input_args.batch_prompt_masks_).to(main_loss.device)
                        # main_loss = main_loss * loss_mask[..., 1:].contiguous().view(-1)
                        # main_loss = torch.sum(main_loss) / torch.sum(loss_mask[..., :-1])
                    elif self.benchmark == "tencent":
                        main_loss = output_data.loss_fn_(tokens[start_idx:end_idx], output_data.logits, labels[start_idx:end_idx])
                    else:
                        raise ValueError(f"unknown benchmark {self.benchmark}")

                    output_data.loss = main_loss
                    output_data.loss_fn_ = None

                    # auxiliary loss - task classification loss
                    task_labels = torch.full(
                        (classifier_probs[idx].size(0),),
                        input_args.task_id-1,
                        dtype=torch.int64,
                        device=self.device_,
                    )
                    criterion = torch.nn.CrossEntropyLoss()
                    output_data.aux_loss = criterion(classifier_probs[idx], task_labels)

        elif "mocl" in self.adapter_configs_:
            for seq_layer in self.seq_module_:
                if seq_layer.name() == "LlamaEmbedding":
                    data = seq_layer.forward(data)
                   
                    embedding = data[0]
                    attn_mask_bool = attn_mask.bool()
                    masked_embedding = embedding.masked_fill(~attn_mask_bool.unsqueeze(-1), 0.0)
                    valid_counts = attn_mask_bool.sum(dim=1, keepdim=True)
                    valid_counts = valid_counts.clamp(min=1.0)
                    mean_embedding = masked_embedding.sum(dim=1) / valid_counts

                    proj_vector = self.proj_network(mean_embedding)
                    # 计算权重
                    input_args.trained_task_weights = []
                    for tid in input_args.trained_task_ids+[input_args.task_id]:  # 需要计算权重的任务id，包括训练完的和当前任务
                        task_vector = self.task_embedding(torch.tensor(tid-1, dtype=torch.long, device=self.device_))
                        task_vector = task_vector.unsqueeze(0).expand(proj_vector.size(0), -1)
                        cosine_similarity = torch.nn.functional.cosine_similarity(proj_vector, task_vector, dim=1)
                        input_args.trained_task_weights.append(cosine_similarity)
                else:
                    data = seq_layer.forward(data)

            # calculate loss
            output = data[0]
            if input_args.is_train:
                assert isinstance(output, List)
                for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                    output_data = output[idx]
                    assert isinstance(output_data, LLMModelOutput)
                    start_idx = lora_config.batch_start_idx_
                    end_idx = lora_config.batch_end_idx_

                    # main loss
                    output_data.loss = output_data.loss_fn_(
                        tokens[start_idx:end_idx],
                        output_data.logits,
                        labels[start_idx:end_idx],
                    )
                    output_data.loss_fn_ = None

                    # mocl 需要计算任务匹配损失
                    task_vector = self.task_embedding(torch.tensor(input_args.task_id-1, dtype=torch.long, device=self.device_))
                    task_vector = task_vector.unsqueeze(0).expand(proj_vector.size(0), -1)
                    labels = torch.ones(task_vector.size(0), dtype=torch.long, device=self.device_)
                    criterion = torch.nn.CosineEmbeddingLoss()
                    output_data.aux_loss = criterion(task_vector, proj_vector, labels)

        elif "olora" in self.adapter_configs_:
            for seq_layer in self.seq_module_:
                data = seq_layer.forward(data)

            output = data[0]  # 只在训练模式下计算损失
            if input_args.is_train:
                assert isinstance(output, List)
                for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                    output_data = output[idx]
                    assert isinstance(output_data, LLMModelOutput)
                    start_idx = lora_config.batch_start_idx_
                    end_idx = lora_config.batch_end_idx_

                    # main loss
                    output_data.loss = output_data.loss_fn_(
                        tokens[start_idx:end_idx],
                        output_data.logits,
                        labels[start_idx:end_idx],
                    )
                    output_data.loss_fn_ = None

                    aux_loss = []
                    for seq_layer in self.seq_module_:  # 遍历所有decoder层
                        if seq_layer.name() == "LlamaDecoderLayer":
                            for w in ["w1", "w2", "w3"]:
                                trained_weights = []
                                for k, v in seq_layer.named_parameters():
                                    if "mlp_" in k and "lora_a_" in k and w in k:
                                        match = re.search(r"expert(\d+)", k)
                                        expert_id = int(match.group(1))
                                        if expert_id + 1 == input_args.task_id:
                                            training_weight = v
                                        elif expert_id + 1 in input_args.trained_task_ids:
                                            trained_weights.append(v)

                                if trained_weights:
                                    for t_w in trained_weights:
                                        inner_products = training_weight @ t_w.transpose(0, 1)
                                        orthogonal_loss = torch.sum(inner_products**2)
                                        aux_loss.append(orthogonal_loss)
                    if aux_loss:
                        output_data.aux_loss = sum(aux_loss)

        else:
            for seq_layer in self.seq_module_:
                data = seq_layer.forward(data)

            # calculate loss
            output = data[0]
            if input_args.is_train:
                assert isinstance(output, List)
                for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
                    output_data = output[idx]
                    assert isinstance(output_data, LLMModelOutput)
                    start_idx = lora_config.batch_start_idx_
                    end_idx = lora_config.batch_end_idx_

                    # main loss
                    output_data.loss = output_data.loss_fn_(
                        tokens[start_idx:end_idx],
                        output_data.logits,
                        labels[start_idx:end_idx],
                    )
                    output_data.loss_fn_ = None

        return output

    def init_lora_layer_weight(
        self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]
    ):
        self.adapter_configs_[config.adapter_name] = config
        if config.adapter_name == "mocl":
            self.task_embedding = torch.nn.Embedding(config.num_experts_, self.config_.dim_).to(device=self.device_)
            self.proj_network = torch.nn.Linear(self.config_.dim_, self.config_.dim_, bias=False).to(device=self.device_)
            if weight is not None:
                with torch.no_grad():
                    self.task_embedding.weight.data.copy_(weight["task_embedding.weight"])
                    self.proj_network.weight.data.copy_(weight["proj_network.weight"])

        self.benchmark = config.benchmark
        if config.benchmark == "mtl5":
            # language modeling output layer
            # output_layer = CasualOutputLayer(
            #     vocab_size=self.config_.vocab_size_, weight=self.model_.lm_head_
            # )
            if weight:
                output_layer_agnews_name = "seq_module_.output.wrapper_module_.layers_.agnews_score_.weight"
                output_layer_amazon_name = "seq_module_.output.wrapper_module_.layers_.amazon_score_.weight"
                output_layer_dbpedia_name = "seq_module_.output.wrapper_module_.layers_.dbpedia_score_.weight"
                output_layer_yahoo_name = "seq_module_.output.wrapper_module_.layers_.yahoo_score_.weight"
                
                output_weights = {
                    "agnews": weight[output_layer_agnews_name],
                    "amazon": weight[output_layer_amazon_name],
                    "dbpedia": weight[output_layer_dbpedia_name],
                    "yahoo": weight[output_layer_yahoo_name]
                }
            else:
                output_weights = None
            
            output_layer = MTL5ClassificationOutputLayer(
                task_type="single_label_classification",
                task_name=config.task_name,
                label_dtype=torch.long,
                hidden_size=self.config_.dim_,
                pad_token_id=config.pad_id,
                device=self.device_,
                weight=output_weights
            )

        elif config.benchmark == "tencent":
            # classification output layer
            output_layer_name = "seq_module_.output.wrapper_module_.layers_.score_.weight"
            output_layer = ClassificationOutputLayer(
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                hidden_size=self.config_.dim_,
                pad_token_id=config.pad_id,
                device=self.device_,
                weight=weight[output_layer_name] if weight else None
            )

        else:
            raise ValueError(f"Unsupported benchmark {config.benchmark}")

        self.seq_module_.output.wrapper_module_.layers_ = output_layer

        # init transformer layers
        for index, transformer_layer in enumerate(self.model_.layers_):
            init_lora_layer_weight(
                transformer_layer, self.config_, config, index, weight
            )

    def from_pretrained(
        path: str,
        device: str,
        bits: int = None,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        load_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
    ):
        # load_dtype will change the precision of LLaMA pre-trained model
        # when loading with quantization (bits = 8 or bits = 4), load_dtype will only influence the actual computing precision
        if load_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported load dtype {load_dtype}")

        if compute_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported compute dtype {compute_dtype}")

        if load_dtype in [torch.bfloat16, torch.float16]:
            logging.info("Loading model with half precision.")

        # BFloat16 is only supported after Ampere GPUs
        if not get_backend().is_bf16_supported():
            if load_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                load_dtype = torch.float16

            if bits in [4, 8] and compute_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                compute_dtype = torch.float16

        if bits in [4, 8]:
            logging.info(f"Loading model with quantization, bits = {bits}.")
            llm_model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map=device,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type,
                ),
                torch_dtype=load_dtype,
            )
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                path, device_map=device, torch_dtype=load_dtype, trust_remote_code=True
            )

        llm_model.requires_grad_(False)
        model = from_pretrained(
            llm_model,
            attn_impl=attn_impl,
            use_sliding_window=use_sliding_window,
            device=device,
        )

        logging.info(f"Use {attn_impl} as attention implementation.")

        return LLMModel(model)

    def get_lora_weight_dict(self) -> Dict[str, torch.Tensor]:
        lora_weight_dict = {}

        # moes_中包含门控网络和任务分类器参数，score_为分类任务的输出层参数
        if "moe-cl" in self.adapter_configs_:
            for name, param in self.named_parameters():
                if "loras_" in name or "moes_" in name or "score_" in name:
                    lora_weight_dict[name] = param

        # proj_network和task_embedding为计算旧任务权重的参数
        elif "mocl" in self.adapter_configs_:
            for name, param in self.named_parameters():
                if "loras_" in name or "proj_network" in name or "task_embedding" in name or "score_" in name:
                    lora_weight_dict[name] = param
        
        else:
            for name, param in self.named_parameters():
                if "loras_" in name or "score_" in name:
                    lora_weight_dict[name] = param

        return lora_weight_dict

    def load_adapter_weight(self, path: str, adapter_name: str):
        with open(f"{path}.json", "r", encoding="utf8") as fp:
            lora_config = MixConfig().from_config(json.load(fp)).check()
        lora_config.adapter_name = adapter_name
        lora_weight = torch.load(f"{path}.bin", map_location=self.device_, weights_only=True)
        self.init_lora_layer_weight(lora_config, lora_weight)
