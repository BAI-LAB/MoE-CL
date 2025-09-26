import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import mlora
from mlora.common import LoraBatchDataConfig, MixConfig, MultiLoraBatchData
from mlora.evaluate import validate_mtl5, validate_tencent
from mlora.model import LLMModel
from mlora.models.modeling_llama import LlamaDecoderLayer
from mlora.tasks.qa_tasks import MTL5Dataset, TencentDataset
from mlora.utils import setup_logging

os.environ["NCCL_DEBUG"] = "WARN"

# Set command line paramete
parser = argparse.ArgumentParser(description="m-LoRA main program")
parser.add_argument(
    "--base_model", type=str, required=True, help="Path to or name of base model"
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to finetune configuration"
)
parser.add_argument("--train_task", type=str, required=True, help="Train task")
parser.add_argument(
    "--load_adapter_file", type=str, default=None, help="Path to adapter model"
)
parser.add_argument("--order", type=str, default="order1", help="Training order")
parser.add_argument(
    "--rand_init", action="store_true", help="Random initialize adapter"
)  # Performance of the model during random initialization
parser.add_argument(
    "--trained_task", type=str, default=None, help="Trained tasks"
)  # MOCL, O-LORA requires this parameter
parser.add_argument("--weight", type=float, default=1.0, help="Weight of adapter")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--nogan", action="store_true", help="GAN training")
args = parser.parse_args()

# Read configuration files
with open(args.config, "r", encoding="utf8") as fp:
    config = json.load(fp)
config["lora"][0]["task_name"] = args.train_task  # 通过命令行参数传入训练任务名称
tokenizer = mlora.Tokenizer(args.base_model)
config["lora"][0]["pad_id"] = tokenizer.pad_id_  # ClassificationOutputLayer 需要用到 pad_id

# Task id mapping
MTL5TaskName2Id = {"agnews": 1, "amazon": 2, "dbpedia": 3, "yahoo": 4}
TencentTaskName2Id = {"shipinhao": 1, "xiaoshijie": 2, "gongzhongpinglun": 3}
if config["benchmark"] == "mtl5":
    TaskName2Id = MTL5TaskName2Id
elif config["benchmark"] == "tencent":
    TaskName2Id = TencentTaskName2Id
else:
    raise ValueError("Benchmark must be one of mtl5 or tencent.")

# Set training hyperparameters
benchmark = config["benchmark"]
adapter_name = config["lora"][0]["name"]
lr = config["lora"][0]["lr"]
num_epochs = config["lora"][0]["num_epochs"]
batch_size = config["lora"][0]["batch_size"]
test_batch_size = config["lora"][0]["test_batch_size"]
if args.debug:  # For debugging
    num_epochs = 1
    batch_size = 1
    test_batch_size = 1


def load_base_model(device) -> Tuple[mlora.Tokenizer, mlora.LLMModel]:
    logging.info(f"Initializing pre-trained model from {args.base_model}")
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
        logging.info(f"Setting cutoff_len to {llm_model.max_seq_len_} automatically.")

    for lora_config in config["lora"]:
        lora_weight = None
        config_class = MixConfig().from_config(lora_config).check()
        config_class.adapter_name = lora_config["name"]
        config_class.task_name = lora_config.get("task_name", "casual")
        config_class.pad_id = lora_config["pad_id"]  # ClassificationOutputLayer 需要用到 pad_id
        config_class.benchmark = config["benchmark"]

        if adapter_file:
            adapter_dir = f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}"
            adapter_file_path = f"{adapter_dir}/{adapter_file}.bin"
            adapter_config_path = f"{adapter_dir}/{adapter_file}.json"

            logging.info(f"Load adapter: {adapter_file_path}")
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
            logging.info(f"Initializing adapter from scratch.")

        llm_model.init_lora_layer_weight(config_class, lora_weight)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def get_ddp_model(model, rank):
    if adapter_name in ["pertask-ft", "sequential-ft-p"]:
        init_adapter_config(config, model, args.load_adapter_file)
        ddp_model = DDP(model, device_ids=[rank])

    elif adapter_name == "sequential-ft-f":
        init_adapter_config(config, model)

        if args.load_adapter_file:
            model.load_state_dict(
                state_dict=torch.load(
                    f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}/{args.load_adapter_file}.bin",
                    map_location=torch.device(rank),
                    weights_only=True,
                )
            )
            torch.cuda.empty_cache()

        for k, v in model.named_parameters():
            v.requires_grad = True

        ddp_model = DDP(model, device_ids=[rank])

    elif adapter_name in ["mocl", "olora"]:
        init_adapter_config(config, model, args.load_adapter_file)

        task_id = TaskName2Id[args.train_task]
        all_task_id = TaskName2Id.values()

        frozen_param_name = []  # Frozen expert
        for tid in all_task_id:
            if tid != task_id:
                frozen_param_name.append(f"expert{tid-1}")

        for k, v in model.named_parameters():  # Freeze other experts
            if "mlp_" in k:
                for fpn in frozen_param_name:
                    if fpn in k:
                        v.requires_grad = False

        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    elif adapter_name == "moe-cl":
        init_adapter_config(config, model, args.load_adapter_file)
        ddp_model = DDP(
            model, device_ids=[rank], find_unused_parameters=True
        )  # Find the unused parameter and set its gradient to 0

    else:
        raise ValueError(f"Adapter {adapter_name} not found.")

    return ddp_model


def train(rank, world_size, train_dataset, val_dataset, all_test_datasets):
    setup(rank, world_size)
    
    logging = setup_logging(
        rank=rank,
        world_size=world_size,
        log_level="INFO",
        log_file="logs/train.log"
    )
    logging.info(f"Starting training on GPU {rank}")

    # STEP 1: ********************Training phase.********************
    if not args.rand_init:
        _, model = load_base_model(rank)
        ddp_model = get_ddp_model(model, rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        total_step = len(dataloader) * num_epochs
        save_step = len(train_dataset) // (batch_size * world_size) // 3  # 每个epoch验证模型三次
        if save_step == 0: save_step = 1
        step_cnt = 0
        best_acc = 0
        task_id = TaskName2Id[args.train_task]
        eval_cnt = 0

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch_idx, batch_data in enumerate(dataloader):
                batch_tokens = batch_data["tokens"]
                batch_labels = batch_data["label"]
                batch_masks = batch_data["mask"]
                # if benchmark == "mtl5":
                #     batch_prompt_masks = batch_data["prompt_mask"]

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
                input_args.is_train = True
                # if benchmark == "mtl5":
                #     input_args.batch_prompt_masks_ = batch_prompt_masks  # 对损失进行mask

                if adapter_name in ["mocl", "olora"]:
                    if args.trained_task:  # 从已训练任务中转移信息
                        trained_task_name = args.trained_task.split("_")
                        input_args.trained_task_ids = [
                            TaskName2Id[task_name] for task_name in trained_task_name
                        ]
                        assert (
                            task_id not in input_args.trained_task_ids
                        ), f"{task_id} already in {input_args.trained_task_ids}"
                    else:
                        input_args.trained_task_ids = []

                output = ddp_model.forward(input_args)[0]
                if output.aux_loss:
                    if adapter_name == "moe-cl":
                        if args.nogan:
                            total_loss = (
                                output.loss - 0 * output.aux_loss
                            )  # The best coefficient
                        else:
                            total_loss = (
                                output.loss - args.weight * output.aux_loss
                            )  # The best coefficient
                    elif adapter_name == "mocl":
                        total_loss = output.loss + output.aux_loss
                    elif adapter_name == "olora":
                        total_loss = output.loss + output.aux_loss
                    else:
                        raise ValueError(
                            f"Adapter {adapter_name} has no auxiliary loss."
                        )
                else:
                    total_loss = output.loss
                optimizer.zero_grad()
                total_loss.backward()

                if adapter_name == "moe-cl":
                    for layer in ddp_model.module.seq_module_:
                        layer = layer.wrapper_module_
                        if isinstance(layer, LlamaDecoderLayer):
                            layer.mlp_.moes_.task_classifier_.weight.grad *= -1

                optimizer.step()
                step_cnt += 1
                if output.aux_loss:
                    logging.info(
                        f"Task: {args.train_task}, Epoch: {epoch}, Batch: {batch_idx * batch_size}/{len(dataloader) * batch_size}, Loss: {round(output.loss.item(), 8)}, Aux_Loss: {round(output.aux_loss.item(), 8)}"
                    )
                else:
                    logging.info(
                        f"Task: {args.train_task}, Epoch: {epoch}, Batch: {batch_idx * batch_size}/{len(dataloader) * batch_size}, Loss: {round(output.loss.item(), 8)}"
                    )

                if step_cnt % save_step == 0 or step_cnt == total_step:
                    eval_cnt += 1
                    if config["benchmark"] == "tencent":
                        res = validate_tencent(
                            rank,
                            world_size,
                            "val",
                            ddp_model.module,
                            val_dataset,
                            args.train_task,
                            task_id,
                            test_batch_size,
                            adapter_name,
                            args,
                            TaskName2Id
                        )

                    elif config["benchmark"] == "mtl5":
                        res = validate_tencent(
                            rank,
                            world_size,
                            "val",
                            ddp_model.module,
                            val_dataset,
                            args.train_task,
                            task_id,
                            test_batch_size,
                            adapter_name,
                            args,
                            TaskName2Id
                        )

                    else:
                        raise ValueError(
                            f"Benchmark {config['benchmark']} not supported!!!"
                        )

                    dist.barrier()  # Wait for all processes to be verified

                    gather_list = None
                    if rank == 0:
                        gather_list = [torch.zeros_like(res) for _ in range(world_size)]
                    dist.gather(res, gather_list=gather_list, dst=0)

                    if rank == 0:
                        logging.info(f"The {eval_cnt}-th evaluation is completed.")
                        total_right = sum(tensor[0].item() for tensor in gather_list)
                        total_total = sum(tensor[1].item() for tensor in gather_list)
                        acc = total_right / total_total if total_total != 0 else 0
                        logging.info(f"Acc: {acc}")

                        if acc >= best_acc:  # Make sure the model will be updated
                            best_acc = acc
                            if (
                                adapter_name == "sequential-ft-f"
                            ):  # Save all model parameters
                                if not os.path.exists(
                                    f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}"
                                ):
                                    os.makedirs(
                                        f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}"
                                    )
                                torch.save(
                                    ddp_model.module.state_dict(),
                                    f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}/{args.train_task}_finetuned.bin",
                                )

                            else:  # Save the parameters of the adapter
                                save_adapter_weight(ddp_model.module, acc=best_acc)

                            logging.info(f"New best accuracy: {best_acc:.4f}, save best model")

    # STEP 2: ********************Testing phase.********************
    # 2.1 Log output
    if args.debug:
        cleanup()
        sys.exit(0)

    if rank == 0:
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}"):
            os.makedirs(f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}")
        with open(
            f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}/log.txt", mode="a"
        ) as f:
            f.write(f"[{formatted}]: Training {args.train_task} finished!\n")
        logging.info(f"Training {args.train_task} finished!")

    # 2.2 Configure the model
    _, model = load_base_model(rank)

    # # 2.3 Start testing
    # dist.barrier()  # Test pre-sync once
    import time
    strat_time = time.time()
    for task_name, task_id in TaskName2Id.items():                              
        if args.rand_init:
            init_adapter_config(config, model)
        else:
            if adapter_name == "sequential-ft-f":
                init_adapter_config(config, model)
                model.load_state_dict(
                    state_dict=torch.load(
                        f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}/{args.train_task}_finetuned.bin",
                        map_location=torch.device(0),
                        weights_only=True,
                    )
                )
                torch.cuda.empty_cache()
            else:
                config["lora"][0]["task_name"] = task_name  # 根据当前测试的任务设置加载输出层
                init_adapter_config(config, model, f"{args.train_task}_finetuned")
        ddp_model = DDP(model, device_ids=[rank])            

        if config["benchmark"] == "tencent":
            test_dataset = all_test_datasets[task_name]
            res = validate_tencent(
                rank,
                world_size,
                "test",
                ddp_model.module,
                test_dataset,
                task_name,
                task_id,
                test_batch_size,
                adapter_name,
                args,
                TaskName2Id
            )
            dist.barrier()  # Wait for all processes to complete testing

        elif config["benchmark"] == "mtl5":
            test_dataset = all_test_datasets[task_name]
            res = validate_tencent(
                rank,
                world_size,
                "test",
                ddp_model.module,
                test_dataset,
                task_name,
                task_id,
                test_batch_size,
                adapter_name,
                args,
                TaskName2Id
            )
            dist.barrier()  # Wait for all processes to complete testing

        else:
            raise f"benchmark {config['benchmark']} not supported"

        # Collect results on all GPUs
        gather_list = None
        if rank == 0:
            gather_list = [torch.zeros_like(res) for _ in range(world_size)]
        dist.gather(res, gather_list=gather_list, dst=0)

        if rank == 0:
            total_right = sum(tensor[0].item() for tensor in gather_list)
            total_total = sum(tensor[1].item() for tensor in gather_list)
            acc = total_right / total_total if total_total != 0 else 0

            now = datetime.now()
            formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(
                f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}/log.txt", mode="a"
            ) as f:
                f.write(f"[{formatted}]: {task_name} acc: {acc}\n")
            logging.info(f"{task_name} acc: {acc}\n")

        dist.barrier()  # Wait for all processes to complete the current task

    cleanup()
    import time
    end_time = time.time()  
    print(f"Total time taken---------------------------------------------: {end_time - strat_time}")

def save_adapter_weight(model: LLMModel, acc=None):
    lora_output_dir = f"/root/MoE-CL/results/{adapter_name}/{benchmark}/{args.order}"
    if not os.path.exists(lora_output_dir):
        os.makedirs(lora_output_dir)

    lora_weight_dict = model.get_lora_weight_dict()
    lora_config_dict = model.adapter_configs_[adapter_name].export()
    lora_config_dict["base_model_name_or_path"] = model.name_or_path_
    lora_config_dict["task_type"] = "qa"
    lora_config_dict["best_acc"] = acc

    torch.save(lora_weight_dict, f"{lora_output_dir}/{args.train_task}_finetuned.bin")
    logging.info("File saved successfully!!!")

    with open(f"{lora_output_dir}/{args.train_task}_finetuned.json", "w") as f:
        json.dump(lora_config_dict, f, indent=4)


if __name__ == "__main__":
    if config["benchmark"] == "tencent":
        train_dataset = TencentDataset(
            name=args.train_task, tokenizer=tokenizer, is_train=True, debug=args.debug
        )
        val_dataset = TencentDataset(
            name=args.train_task,
            tokenizer=tokenizer,
            is_train=False,
            is_val=True,
            debug=args.debug,
        )
        all_test_datasets = {}
        for task_name, task_id in TaskName2Id.items():
            all_test_datasets[task_name] = TencentDataset(
                name=task_name,
                tokenizer=tokenizer,
                is_train=False,
                is_val=False,
                debug=args.debug,
            )

    elif config["benchmark"] == "mtl5":
        train_dataset = MTL5Dataset(
            name=args.train_task, tokenizer=tokenizer, is_train=True, debug=args.debug
        )
        val_dataset = MTL5Dataset(
            name=args.train_task,
            tokenizer=tokenizer,
            is_train=False,
            is_val=True,
            debug=args.debug,
        )
        all_test_datasets = {}
        for task_name, task_id in TaskName2Id.items():
            all_test_datasets[task_name] = MTL5Dataset(
                name=task_name,
                tokenizer=tokenizer,
                is_train=False,
                is_val=False,
                debug=args.debug,
            )

    else:
        raise f"benchmark {config['benchmark']} not supported"

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size, train_dataset, val_dataset, all_test_datasets),
        nprocs=world_size,
        join=True,
    )
