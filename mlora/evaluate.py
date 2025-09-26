import logging
import string
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import mlora
from mlora.common import LoraBatchDataConfig, MultiLoraBatchData
from mlora.model import LLMModel
from mlora.tokenizer import Tokenizer


def validate_tencent(
    rank: int,
    world_size: int,
    mode: str,
    model: LLMModel,
    dataset: Dataset,
    task_name: str,
    task_id: int,
    batch_size: int,
    adapter_name: str,
    args: Any,
    TaskName2Id: Dict[str, int],
):
    """
    Binary classification output evaluation.
    """
    model.eval()
    with torch.no_grad():
        right, total = 0, 0
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        for batch_idx, batch_data in enumerate(dataloader):
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
            if adapter_name == "mocl":
                if args.trained_task:
                    trained_task_name = args.trained_task.split("_")
                    input_args.trained_task_ids = [
                        TaskName2Id[task_name] for task_name in trained_task_name
                    ]
                    # print(task_id, input_args.trained_task_ids)
                    # assert (  # 评测时不可避免出现当前任务id和已完成训练任务id一致
                    #     task_id not in input_args.trained_task_ids
                    # ), f"{task_id} already in {input_args.trained_task_ids}"
                else:
                    input_args.trained_task_ids = []

            output = model.forward(input_args)[0]
            logits = output.logits

            sequence_lengths = torch.sum(batch_masks, dim=-1).to(logits.device)
            sequence_lengths = sequence_lengths - 1
            pooled_logits = logits[
                torch.arange(logits.size(0), device=logits.device), sequence_lengths
            ]

            pooled_logits = pooled_logits.softmax(-1).argmax(-1)
            labels = batch_labels.to(dtype=torch.int32, device=logits.device)

            right += torch.sum(labels.squeeze() == pooled_logits)
            total += pooled_logits.shape[0]
            logging.info(
                f"mode: {mode}, task: {task_name}, batch: {(batch_idx + 1) * batch_size}/{len(dataloader) * batch_size}"
            )

    model.train()
    return torch.tensor([right, total], device=rank)


def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions: List[str], references: List[str]):
    assert len(predictions) == len(
        references
    ), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match = 0
    for pred, gold in zip(predictions, references):
        gold = [gold]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold
        )
    return exact_match


def validate_mtl5(
    rank: int,
    world_size: int,
    mode: str,
    model: LLMModel,
    dataset: Dataset,
    task_name: str,
    task_id: int,
    batch_size: int,
    tokenizer: Tokenizer,
):
    """Causal modeling output evaluation."""
    model.eval()
    with torch.no_grad():
        right, total = 0, 0
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        right, total = 0, 0
        for batch_idx, (batch_prompt, batch_label) in enumerate(dataloader):
            generate_paramas = mlora.GenerateConfig(
                adapter_name="moe-cl",
                prompt_template=None,
                prompts=[(prompt, None) for prompt in batch_prompt],
            )
            output = mlora.generate(
                model, tokenizer, [generate_paramas], max_gen_len=32, task_id=task_id
            )
            exact_match = compute_metrics(output["moe-cl"], list(batch_label))
            right += exact_match
            total += len(batch_label)
            if (batch_idx + 1) % 100 == 0:
                logging.info(
                    f"mode: {mode}, task: {task_name}, batch: {(batch_idx + 1) * batch_size}/{len(dataloader) * batch_size}"
                )

    model.train()
    return torch.tensor([right, total], device=rank)


# def evaluate_tencent(model, dataset, task_name, task_id, batch_size, adapter_name, args, TaskName2Id):
#     """Single GPU reasoning."""
#     model.eval()
#     with torch.no_grad():
#         right, total = 0, 0
#         dataloader = DataLoader(dataset, batch_size=batch_size)

#         for batch_idx, (batch_tokens, batch_labels, batch_masks) in enumerate(
#             dataloader
#         ):
#             input_args = MultiLoraBatchData(
#                 attention_masks_=batch_masks,
#                 batch_labels_=batch_labels,
#                 batch_tokens_=batch_tokens,
#                 lora_batch_data_config_=[
#                     LoraBatchDataConfig(
#                         batch_start_idx_=0, batch_end_idx_=batch_tokens.size(0)
#                     )
#                 ],
#             )
#             input_args.task_id = task_id
#             input_args.is_train = False
#             if adapter_name == "mocl":
#                 if args.trained_task:
#                     trained_task_name = args.trained_task.split("_")
#                     trained_task_name.append(args.train_task)
#                     input_args.trained_task_ids = [
#                         TaskName2Id[task_name] for task_name in trained_task_name
#                     ]
#                     if task_id in input_args.trained_task_ids:
#                         input_args.trained_task_ids.remove(task_id)
#                     assert (
#                         task_id not in input_args.trained_task_ids
#                     ), f"{task_id} already in {input_args.trained_task_ids}"
#                 else:
#                     input_args.trained_task_ids = []

#             output = model.forward(input_args)[0]
#             logits = output.logits

#             sequence_lengths = torch.sum(batch_masks, dim=-1).to(logits.device)
#             sequence_lengths = sequence_lengths - 1
#             pooled_logits = logits[
#                 torch.arange(logits.size(0), device=logits.device), sequence_lengths
#             ]

#             pooled_logits = pooled_logits.softmax(-1).argmax(-1)
#             labels = batch_labels.to(dtype=torch.int32, device=logits.device)

#             right += torch.sum(labels.squeeze() == pooled_logits)
#             total += pooled_logits.shape[0]
#             logging.info(
#                 f"mode: test, task: {task_name}, batch: {(batch_idx+1)*batch_size}/{len(dataloader)*batch_size}"
#             )
#     return right / total


# def evaluate_mtl5(model, tokenizer, dataset, task_name, task_id, batch_size):
#     """Single GPU reasoning."""
#     model.eval()
#     with torch.no_grad():
#         right, total = 0, 0
#         dataloader = DataLoader(dataset, batch_size=batch_size)

#         for batch_idx, (batch_prompt, batch_label) in enumerate(dataloader):
#             generate_paramas = mlora.GenerateConfig(
#                 adapter_name="moe-cl",
#                 prompt_template=None,
#                 prompts=[(prompt, None) for prompt in batch_prompt],
#             )
#             output = mlora.generate(
#                 model, tokenizer, [generate_paramas], max_gen_len=32, task_id=task_id
#             )
#             exact_match = compute_metrics(
#                 output["moe-cl"], list(batch_label)
#             )  # 输入是两个字符串列表
#             right += exact_match
#             total += len(batch_label)
#             logging(
#                 f"mode: test, task: {task_name}, batch: {(batch_idx+1)*batch_size}/{len(dataloader)*batch_size}"
#             )
#     return right / total
