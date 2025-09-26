import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

import mlora
from mlora.common import LoraBatchDataConfig, MixConfig, MultiLoraBatchData
from mlora.evaluate import validate_tencent
from mlora.model import LLMModel
from mlora.tasks.qa_tasks import TencentDataset

os.environ["NCCL_DEBUG"] = "WARN"

# Set command line parameters
parser = argparse.ArgumentParser(description="m-LoRA main program")
parser.add_argument(
    "--base_model", type=str, default="../model/Hunyuan-7B-Instruct", help="Path to or name of base model"
)
parser.add_argument(
    "--config", type=str, default="configs/tencent/moe-cl.json", help="Path to finetune configuration"
)
parser.add_argument("--order", type=str, default="order1_5000samples_0307_noGAN", help="Order")
parser.add_argument(
    "--load_adapter_file", type=str, default="gongzhongpinglun_finetuned", help="Path to adapter model"
)
args = parser.parse_args()

# Read configuration files
with open(args.config, "r", encoding="utf8") as fp:
    config = json.load(fp)
tokenizer = mlora.Tokenizer(args.base_model)
config["lora"][0]["pad_id"] = tokenizer.pad_id_  # ClassificationOutputLayer 需要用到 pad_id

TaskName2Id = {"shipinhao": 1, "xiaoshijie": 2, "gongzhongpinglun": 3}

# Set training hyperparameters
test_batch_size = 512
adapter_name = config['lora'][0]['name']


def load_base_model(device) -> Tuple[mlora.Tokenizer, mlora.LLMModel]:
    print(f"Initializing pre-trained model from {args.base_model}")
    model = mlora.LLMModel.from_pretrained(
        path=args.base_model,
        device=device,
        attn_impl="eager",
        use_sliding_window=False,
        bits=None,
        load_dtype=torch.bfloat16,
    )
    tokenizer = mlora.Tokenizer(args.base_model)
    return tokenizer, model


def init_adapter_config(
    config: Dict[str, any], llm_model: mlora.LLMModel, adapter_file: str = None
) -> None:
    if config["cutoff_len"] == -1:
        config["cutoff_len"] = llm_model.max_seq_len_
        print(f"Setting cutoff_len to {llm_model.max_seq_len_} automatically.")

    for lora_config in config["lora"]:
        lora_weight = None
        config_class = MixConfig().from_config(lora_config).check()
        config_class.adapter_name = lora_config["name"]
        config_class.task_name = lora_config.get("task_name", "casual")
        config_class.pad_id = lora_config["pad_id"]  # ClassificationOutputLayer 需要用到 pad_id
        config_class.benchmark = config["benchmark"]

        if adapter_file:
            adapter_dir = f"/root/MoE-CL/results/{adapter_name}/tencent/{args.order}"
            adapter_file_path = f"{adapter_dir}/{adapter_file}.bin"
            adapter_config_path = f"{adapter_dir}/{adapter_file}.json"

            print(f"Load adapter: {adapter_file_path}")
            with open(adapter_config_path, "r", encoding="utf8") as fp:
                adapter_config = json.load(fp)
            base_model_name_or_path = adapter_config.get("base_model_name_or_path", "")
            if (
                base_model_name_or_path != ""
                and base_model_name_or_path != llm_model.name_or_path_
            ):
                raise ValueError(
                    "loading adapter with unmatched base model."
                    + f" current is {llm_model.name_or_path_}, provided {base_model_name_or_path}"
                )
            lora_weight = torch.load(
                adapter_file_path, map_location=None, weights_only=True
            )

        else:
            print(f"Initializing adapter from scratch.")

        llm_model.init_lora_layer_weight(config_class, lora_weight)


def calculate_metrics(y_true, y_pred):
    """计算每个类别的精确率、召回率和F1分数"""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    return precision, recall, f1, support


def validate_tencent(
    mode: str,
    model: LLMModel,
    dataset: Dataset,
    task_name: str,
    task_id: int,
    batch_size: int,
):
    """
    Binary classification output evaluation.
    """
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_scores = []
        all_predictions = []
        all_labels = []
        right = 0
        total = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # if batch_idx == 2: break
            batch_tokens = batch_data["tokens"]
            batch_labels = batch_data["label"]
            batch_masks = batch_data["mask"]

            input_args = MultiLoraBatchData(
                attention_masks_=batch_masks,
                batch_labels_=batch_labels,
                batch_tokens_=batch_tokens,
                lora_batch_data_config_=[
                    LoraBatchDataConfig(
                        batch_start_idx_=0, batch_end_idx_=batch_tokens.size(0)
                    )
                ],
            )
            input_args.task_id = task_id
            input_args.is_train = False

            output = model.forward(input_args)[0]
            logits = output.logits

            sequence_lengths = torch.sum(batch_masks, dim=-1).to(logits.device)
            sequence_lengths = sequence_lengths - 1
            pooled_logits = logits[
                torch.arange(logits.size(0), device=logits.device), sequence_lengths
            ]

            softmax_scores = pooled_logits.softmax(-1)
            all_scores.extend(softmax_scores.cpu().numpy())
            pooled_logits = softmax_scores.argmax(-1)
            labels = batch_labels.to(dtype=torch.int32, device=logits.device)

            # 收集预测结果和真实标签
            all_predictions.extend(pooled_logits.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

            right += torch.sum(labels.squeeze() == pooled_logits)
            total += pooled_logits.shape[0]

            print(
                f"mode: {mode}, task: {task_name}, batch: {(batch_idx + 1) * batch_size}/{len(dataloader) * batch_size}"
            )

        all_scores = np.stack(all_scores, axis=0)
        df = pd.DataFrame({
            "label": all_labels,
            "score_0": all_scores[:, 0],
            "score_1": all_scores[:, 1],
        })
        df.to_csv(f"/root/MoE-CL/results/{adapter_name}/tencent/{args.order}/{task_name}.csv", index=False)
        return


if __name__ == "__main__":
    # 加载模型
    _, model = load_base_model(0)
    init_adapter_config(config, model, args.load_adapter_file)

    start_time = time.time()
    for task_name, task_id in TaskName2Id.items():
        # 加载测试数据集
        test_dataset = TencentDataset(
            name=task_name,
            tokenizer=tokenizer,
            is_train=False,
            is_val=False,
            debug=False,
        )

        # 进行评估
        validate_tencent(
            mode="test",
            model=model,
            dataset=test_dataset,
            task_name=task_name,
            task_id=task_id,
            batch_size=test_batch_size,
        )
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
