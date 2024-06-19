import os, random, torch, json, subprocess
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel
from typing import Dict, List, Optional
import datasets
import fire
from datasets import load_dataset, load_from_disk, concatenate_datasets
from weak_to_strong.common import get_tokenizer
import itertools
import pickle
import time
from dataclasses import dataclass
from typing import Callable
import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint
from weak_to_strong.common import clear_mem
from weak_to_strong.train import ModelConfig
from weak_to_strong.eval import eval_model_acc, eval_reward_model_acc, eval_dpo_model_acc, eval_simpo_model_acc
from weak_to_strong.loss import xent_loss, bce_loss
from accelerate import Accelerator

from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload,
  ShardingStrategy,
  BackwardPrefetch,
  MixedPrecision
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from argparse import ArgumentParser
from functools import partial

import weak_to_strong.logger as logger
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, load_reward_dataset, load_helpful_dataset,
                                     load_preference_dataset, load_weak_preference_data_from_disk, load_gt_preference_data_from_disk, load_w2s_preference_dataset,
                                     load_preference_helpful_dataset,
                                     load_w2s_dataset, tokenize_dataset, tokenize_dpo_dataset)

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        path="path_to_gpt2",
        default_lr=1e-6,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="gpt2-medium",
        path="path_to_gpt2-medium",
        default_lr=1e-6,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="gpt2-large",
        path="path_to_gpt2-large",
        default_lr=1e-6,
        eval_batch_size=8,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="gpt2-xl",
        path="path_to_gpt2-xl",
        default_lr=1e-6,
        eval_batch_size=16,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )

    ),
    ModelConfig(
        name="mistral",
        path="path_to_mistral-7b",
        default_lr=1e-6,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        ),
    ),
    ModelConfig(
        name="opt-125m",
        path="path_to_opt-125m",
        default_lr=1e-6,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="opt-350m",
        path="path_to_opt-350m",
        default_lr=1e-6,
        eval_batch_size=32,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="opt-1.3b",
        path="path_to_opt-1.3b",
        default_lr=1e-6,
        eval_batch_size=16,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="opt-2.7b",
        path="path_to_opt-2.7b",
        default_lr=1e-6,
        eval_batch_size=8,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        )
    ),
    ModelConfig(
        name="opt-6.7b",
        path="path_to_opt-6.7b",
        default_lr=1e-6,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.device_count() > 1
        ),
    ),

]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def sft_train(
    rank: int,
    model: torch.nn.Module,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    device,
    lr: float = 1e-6,
    minibatch_size: int = 8,
    epochs: int = 1,
    optimizer_name: str = "adam",
    warmup_ratio: float = 0.1,
    log_every: int = 10,
):
    if rank == 0:
        print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    nsteps = len(ds_chosen) * epochs // batch_size

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    # if lr_schedule == "cosine_anneal":
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    # else:
    #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (int(nsteps * warmup_ratio) + 1)))
    step = 0
    it_chosen = itertools.chain.from_iterable(itertools.repeat(ds_chosen, epochs))

    losses = []

    def get_log_ps(logits, idxs, loss_mask):
        """
        args:
        logits: A tensor of shape (batch_size, seq_len, vocab_size)
        idxs: A torch.long tensor of shape (batch_size, seq_len)
        loss_mask: A torch.float tensor of shape (batch_size, seq_len)

        returns:
        A tensor of shape (batch_size, seq_len), the log probabilities of each sequence in the batch
        """

        idxs = idxs[:, 1:].unsqueeze(2)
        loss_mask = loss_mask[:, 1:]
        log_p_distributions = F.log_softmax(logits, dim=-1)[:, :-1]
        log_ps = torch.gather(log_p_distributions, dim=2, index=idxs).squeeze(2)
        return (log_ps * loss_mask)#.sum(dim=-1)
    

    # sync processes
    dist.barrier()

    while step < nsteps:
        loss_tot = 0
        
        for i in range(batch_size // minibatch_size):
            try:
                mbatch_chosen = [next(it_chosen) for _ in range(minibatch_size)]
            except StopIteration:
                break
            # for now, we do not support mini_batch_size > 1, so please set mini_batch_size=1
            input_ids_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_input_ids"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(device)
            )
            loss_mask_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(device)
            )
            
            
            policy_logits_chosen = model(input_ids_chosen).logits
            policy_log_ps_chosen = get_log_ps(policy_logits_chosen, input_ids_chosen, loss_mask_chosen)
            
            current_loss = - (policy_log_ps_chosen.sum(dim=-1) / loss_mask_chosen.sum(dim=-1)).mean()          
            # print(current_loss)

            current_loss /= (batch_size // minibatch_size)
            loss_tot += current_loss.item()
            current_loss.backward()
            
        losses.append(loss_tot)
        if rank == 0:
            logger.logkvs(
                {
                    "step": step,
                    "progress": step / nsteps,
                    "loss": loss_tot,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if log_every and step % log_every == 0 and rank == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Total Num. Losses: {len(losses)}"
            )
            losses = []
            
        step += 1
        if rank == 0:
            logger.dumpkvs()
    
    # dist.barrier()

    return model

def train_simpo_model(
    rank: int,
    model: torch.nn.Module,
    ds_rejected: datasets.Dataset,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    device,
    lr: float = 1e-6,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds_rejected: Optional[datasets.Dataset] = None,
    eval_ds_chosen: Optional[datasets.Dataset] = None,
    epochs: int = 1,
    optimizer_name: str = "adam",
    use_reward_mechanism: bool = False,
    beta: float = 2.0,
    gamma: float = 1.0,
    reward_alpha: Optional[float] = 0.5, # reward strength
    reward_type: Optional[str] = "reverse",
    warmup_ratio: float = 0.1,
):
    assert len(ds_rejected) == len(ds_chosen)
    assert len(eval_ds_rejected) == len(eval_ds_chosen)
    if rank == 0:
        print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    nsteps = len(ds_rejected) * epochs // batch_size

    # def lr_schedule_fn(step):
    #     if lr_schedule == "constant":
    #         return 1
    #     else:
    #         assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    # if lr_schedule == "cosine_anneal":
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    # else:
    #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (int(nsteps * warmup_ratio) + 1)))
    step = 0
    it_rejected = itertools.chain.from_iterable(itertools.repeat(ds_rejected, epochs))
    it_chosen = itertools.chain.from_iterable(itertools.repeat(ds_chosen, epochs))

    losses = []
    accuracies = []
    chosen_rewards = []
    rejected_rewards = []
    eval_acc_dict = {}
    
    def get_log_ps(logits, idxs, loss_mask):
        """
        args:
        logits: A tensor of shape (batch_size, seq_len, vocab_size)
        idxs: A torch.long tensor of shape (batch_size, seq_len)
        loss_mask: A torch.float tensor of shape (batch_size, seq_len)

        returns:
        A tensor of shape (batch_size, seq_len), the log probabilities of each sequence in the batch
        """

        idxs = idxs[:, 1:].unsqueeze(2)
        loss_mask = loss_mask[:, 1:]
        log_p_distributions = F.log_softmax(logits, dim=-1)[:, :-1]
        log_ps = torch.gather(log_p_distributions, dim=2, index=idxs).squeeze(2)
        return (log_ps * loss_mask)#.sum(dim=-1)
    

    # sync processes
    dist.barrier()

    while step < nsteps:
        loss_tot = 0
        
        all_log_ps_diff = []
        all_gt_labels = []
        all_chosen_rewards = []
        all_rejected_rewards = []
        for i in range(batch_size // minibatch_size):
            try:
                mbatch_rejected = [next(it_rejected) for _ in range(minibatch_size)]
                mbatch_chosen = [next(it_chosen) for _ in range(minibatch_size)]
            except StopIteration:
                break
            # we will concat chosen and rejected samples into one batch, set minibatch_size >= 2
            chosen_len = len(mbatch_chosen)
            mbatch_concat = mbatch_rejected + mbatch_chosen
            
            input_ids_concat = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_input_ids"]) for ex in mbatch_concat])
                .transpose(
                    0,
                    1,
                )
                .to(device)
            )

            loss_mask_concat = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_concat])
                .transpose(
                    0,
                    1,
                )
                .to(device)
            )            
            # here, the gt_labels should be converted to follow the indication of the accuracy of weak_labels
            # gt_labels = torch.tensor([ex["gt_label"] if "gt_label" in ex else ex["soft_label"] for ex in mbatch_rejected])
            gt_labels = torch.tensor([1.0 if ex['acc'] else 0.0 for ex in mbatch_rejected])


            logits_concat = model(input_ids_concat).logits
            logits_rejected, logits_chosen = logits_concat[:chosen_len], logits_concat[chosen_len:]
            input_ids_rejected, input_ids_chosen = input_ids_concat[:chosen_len], input_ids_concat[chosen_len:]
            loss_mask_rejected, loss_mask_chosen = loss_mask_concat[:chosen_len], loss_mask_concat[chosen_len:]
             
            log_ps_rejected = get_log_ps(logits_rejected, input_ids_rejected, loss_mask_rejected)
            log_ps_rejected = log_ps_rejected.sum(dim=-1) / loss_mask_rejected[:, 1:].sum(dim=-1)

            log_ps_chosen = get_log_ps(logits_chosen, input_ids_chosen, loss_mask_chosen)
            log_ps_chosen = log_ps_chosen.sum(dim=-1) / loss_mask_chosen[:, 1:].sum(dim=-1)

            
            log_mean_ratio = log_ps_chosen - log_ps_rejected
            
            simpo_loss = - F.logsigmoid(beta * log_mean_ratio - gamma)
            current_loss = simpo_loss.mean()


            # compute rewards
            with torch.no_grad():
                chosen_reward = (beta * (log_ps_chosen).detach().cpu()).unsqueeze(dim=0)
                rejected_reward = (beta * (log_ps_rejected).detach().cpu()).unsqueeze(dim=0)
            if use_reward_mechanism:
                if reward_type == 'sft':
                    # calculate SFT loss on human rejected data
                    reward_loss = 0
                    for l in range(len(mbatch_rejected)):
                        if (log_ps_chosen[l].item() < log_ps_rejected[l].item()) == (mbatch_rejected[l]['acc'] == 1.0):     
                            reward_input_idxs = input_ids_rejected[l, 1:].unsqueeze(0).unsqueeze(2) if (mbatch_rejected[l]['acc'] == 1.0) else input_ids_chosen[l, 1:].unsqueeze(0).unsqueeze(2)
                            human_rejected_log_p = F.log_softmax(logits_rejected, dim=-1)[l, :-1].unsqueeze(0) if (mbatch_rejected[l]['acc'] == 1.0) else F.log_softmax(logits_chosen, dim=-1)[l, :-1].unsqueeze(0)
                            human_loss_mask = loss_mask_rejected if (mbatch_rejected[l]['acc'] == 1.0) else loss_mask_chosen
                            reward_log_ps = (torch.gather(human_rejected_log_p, dim=2, index=reward_input_idxs).squeeze(2) * human_loss_mask[l, 1:].unsqueeze(0)).mean(dim=-1).squeeze(0)
                            reward_loss += - reward_log_ps
                    reward_loss /= len(mbatch_rejected)
                    current_loss += reward_alpha * reward_loss
                elif reward_type == 'reverse':
                    reward_loss = 0
                    for l in range(len(mbatch_rejected)):
                        if (log_ps_chosen[l].item() < log_ps_rejected[l].item()) == (mbatch_rejected[l]['acc'] == 1.0):  
                            reward_single_loss =  - F.logsigmoid(beta * (- log_mean_ratio[l]) - gamma) if (mbatch_rejected[l]['acc'] == 1.0) else - F.logsigmoid(beta * (log_mean_ratio[l]) - gamma)
                            reward_loss += reward_single_loss
                    reward_loss /= len(mbatch_rejected)
                    current_loss += reward_alpha * reward_loss
                else:
                    print("Not implemented reward type")
                    assert 0==1
            
            current_loss /= (batch_size // minibatch_size)
            loss_tot += current_loss.item()
            current_loss.backward()
            ### IMPORTANT: Notice that all the following metrics (EXCEPT train_accuracy) are w.r.t. weak labels, not w.r.t. human labels, if in weak-to-strong generalization
            all_chosen_rewards.extend(chosen_reward)
            all_rejected_rewards.extend(rejected_reward)
            all_log_ps_diff.extend(log_mean_ratio.detach().cpu())
            all_gt_labels.extend(gt_labels.detach())

        
        all_log_ps_diff = torch.stack(all_log_ps_diff)
        all_gt_labels = torch.stack(all_gt_labels)
        avg_chosen_rewards = torch.mean(torch.stack(all_chosen_rewards))
        avg_rejected_rewards = torch.mean(torch.stack(all_rejected_rewards))
        
        losses.append(loss_tot)
        accuracies.append(
            torch.mean(
                ((all_log_ps_diff >= 0.0) == (all_gt_labels == 1.0)).to(
                    torch.float32
                )
            ).item()
        )
        chosen_rewards.append(avg_chosen_rewards.item())
        rejected_rewards.append(avg_rejected_rewards.item())
        if rank == 0:
            logger.logkvs(
                {
                    "step": step,
                    "progress": step / nsteps,
                    "loss": loss_tot,
                    "train_accuracy": accuracies[-1],
                    "train_chosen_reward": chosen_rewards[-1],
                    "train_rejected_reward": rejected_rewards[-1],
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if log_every and step % log_every == 0 and rank == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Avg. Acc: {np.mean(accuracies)} Total Num. Losses: {len(losses)} Avg. Chosen Reward:  {np.mean(chosen_rewards)} Avg. Rejected Reward:  {np.mean(rejected_rewards)}"
            )
            losses = []
            accuracies = []
            chosen_rewards = []
            rejected_rewards = []
        step += 1
        if rank == 0:
            logger.dumpkvs()
    final_eval_results_rejected = None
    final_eval_results_chosen = None
    dist.barrier()
    if eval_every:
        print("Final evaluation:")
        final_eval_results_rejected, final_eval_results_chosen = eval_simpo_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
        if rank == 0:
            logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results_rejected]))
            logger.dumpkvs()

    dist.barrier()
    return final_eval_results_rejected, final_eval_results_chosen


def fsdp_main(gpu, args):
    world_size = args.gpus
    rank = gpu

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f'Started process {rank}')

    device = torch.device(f'cuda:{rank}')

    # init config
    if args.minibatch_size_per_device is None:
        args.minibatch_size_per_device = 1
    assert args.ds_name in VALID_DATASETS, f"Unknown dataset {args.ds_name} not in {VALID_DATASETS}"
    assert (
        args.weak_model_size is None or args.weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[args.model_size]
    use_default_lr = False
    if args.lr is None:
        # assert (
        #     batch_size == 32
        # ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        args.lr = model_config.default_lr
        use_default_lr = True

    if args.optim is None:
        args.optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "setting": 'simpo',
        "batch_size": args.batch_size,
        "max_ctx": args.max_ctx,
        "ds_name": args.ds_name,
        # "loss": w2s_loss if w2s_loss is not None else loss,
        "n_docs": args.n_docs,
        "n_test_docs": args.n_test_docs,
        "model_size": args.model_size,
        "lr": args.lr,
        "optim": args.optim,
        "epochs": args.epochs,
        "sft_epochs": args.sft_epochs,
        # "force_retrain": force_retrain,
        "seed": args.seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": args.train_with_dropout,
        # "results_folder": results_folder,
        # "linear_probe": linear_probe,
        # "lr_schedule": lr_schedule,
        "eval_every": args.eval_every,
        # "sweep_subfolder": sweep_subfolder,
        # "use_mixed_data": use_mixed_data,
        "use_human_data": args.use_human_data,
        "use_reward_mechanism": args.use_reward_mechanism,
        "n_extra_docs": args.n_extra_docs if args.use_human_data else 0,
        "simpo_beta": args.beta,
        "simpo_gamma": args.gamma,
        "reward_alpha": args.reward_alpha,
        "reward_type": args.reward_type
    }
    weak_labels_path = args.weak_labels_path
    if args.weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = args.weak_model_size
        # weak_model_config["loss"] = loss
        weak_model_config["use_human_data"] = False
        weak_model_config["use_reward_mechanism"] = False
        weak_model_config["n_extra_docs"] = 0
        # weak_model_config["epochs"] = 1
        # weak_model_config["batch_size"] = 32
        weak_model_config["reward_alpha"] = None
        weak_model_config["reward_type"] = None
        # weak_model_config["sft_epochs"] = 1
        if args.bootstrapping:
            weak_model_config["model_size"] = args.intermediate_model_size
            weak_model_config["weak_model_size"] = args.weak_model_size
            config["intermediate_model_size"] = args.intermediate_model_size
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[args.weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)
        weak_labels_path = (
            args.results_folder + "/" + args.sweep_subfolder + "/" + weak_model_config_name + "/weak_labels"
        )
    minibatch_size = args.minibatch_size_per_device
    eval_batch_size = model_config.eval_batch_size

    # Load reward dataset
    rejected_dataset, chosen_dataset = load_preference_dataset(args.ds_name, seed=args.seed, split_sizes=dict(train=args.n_docs, test=args.n_test_docs))
    
    # Split the training dataset in half
    train_ds_rejected, test_ds_rejected = rejected_dataset["train"], rejected_dataset["test"]
    train_ds_chosen, test_ds_chosen = chosen_dataset["train"], chosen_dataset["test"]
    train_ds_rejected = train_ds_rejected.rename_column('txt', 'dpo_txt')
    train_ds_chosen = train_ds_chosen.rename_column('txt', 'dpo_txt')
    test_ds_rejected = test_ds_rejected.rename_column('txt', 'dpo_txt')
    test_ds_chosen = test_ds_chosen.rename_column('txt', 'dpo_txt')

    
    if args.use_human_data:
        extra_rejected_dataset, extra_chosen_dataset = load_preference_helpful_dataset(args.ds_name, seed=args.seed, split_sizes=dict(train=args.n_extra_docs, test=0))
        extra_rejected_dataset, extra_chosen_dataset = extra_rejected_dataset["train"], extra_chosen_dataset["train"]
        extra_rejected_dataset = extra_rejected_dataset.rename_column('txt', 'dpo_txt')
        extra_chosen_dataset = extra_chosen_dataset.rename_column('txt', 'dpo_txt')
        extra_rejected_dataset = extra_rejected_dataset.remove_columns([col for col in extra_rejected_dataset.column_names if col in ['chosen', 'rejected']])
        extra_chosen_dataset = extra_chosen_dataset.remove_columns([col for col in extra_chosen_dataset.column_names if col in ['chosen', 'rejected']])
        print("len(extra train):", len(extra_rejected_dataset))

    
    if weak_labels_path is None:
        train1_ds_rejected, train1_ds_chosen = train_ds_rejected, train_ds_chosen
        train2_ds_rejected, train2_ds_chosen = load_w2s_preference_dataset(args.ds_name, seed=args.seed, split_sizes=dict(train=args.n_w2s_docs))
        train2_ds_rejected, train2_ds_chosen = train2_ds_rejected["train"], train2_ds_chosen["train"]
        train2_ds_rejected = train2_ds_rejected.rename_column('txt', 'dpo_txt')
        train2_ds_chosen = train2_ds_chosen.rename_column('txt', 'dpo_txt')

        train1_ds_rejected = train1_ds_rejected.shuffle(seed=args.seed)
        train1_ds_chosen = train1_ds_chosen.shuffle(seed=args.seed)
        print("len(train1):", len(train1_ds_rejected), "len(train2):", len(train2_ds_rejected))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
    
        train1_ds_rejected, train1_ds_chosen = load_weak_preference_data_from_disk(weak_labels_path, metric="mean_log_ps")
        
        if args.bootstrapping:
            train2_ds_rejected, train2_ds_chosen = load_w2s_preference_dataset(args.ds_name, seed=args.seed, split_sizes=dict(train=args.n_w2s_docs))
            train2_ds_rejected, train2_ds_chosen = train2_ds_rejected["train"], train2_ds_chosen["train"]
            train2_ds_rejected = train2_ds_rejected.rename_column('txt', 'dpo_txt')
            train2_ds_chosen = train2_ds_chosen.rename_column('txt', 'dpo_txt')
        else:
            train2_ds_rejected, train2_ds_chosen = load_preference_dataset(args.ds_name, seed=args.seed, split_sizes=dict(train=args.n_w2s_docs, test=args.n_test_docs))
            train2_ds_rejected, train2_ds_chosen = train2_ds_rejected["train"], train2_ds_chosen["train"]
            train2_ds_rejected = train2_ds_rejected.rename_column('txt', 'dpo_txt')
            train2_ds_chosen = train2_ds_chosen.rename_column('txt', 'dpo_txt')

        

        ####### for highconf filter
        if args.high_conf_filter:
            print("Filtering high condident samples...")
            chosen_ids = []
            for ind in range(len(train1_ds_chosen)):
                if train1_ds_chosen[ind]['acc'] == True:
                    probs = train1_ds_chosen[ind]['mean_log_ps']
                    probs = np.exp(probs) / (1 + np.exp(probs))
                    if probs >= args.conf_threshold:
                        chosen_ids.append(ind)
            train1_ds_rejected = train1_ds_rejected.select(chosen_ids)
            train1_ds_chosen = train1_ds_chosen.select(chosen_ids)

            config["conf_threshold"] = args.conf_threshold

            # make number of helpful data == number of weak data
            if args.use_human_data:
                extra_rejected_dataset = extra_rejected_dataset.select(chosen_ids)
                extra_chosen_dataset = extra_chosen_dataset.select(chosen_ids)
        #####
            
        if args.use_human_data:
            train1_ds_rejected = train1_ds_rejected.remove_columns([col for col in train1_ds_rejected.column_names if col not in extra_rejected_dataset.column_names])
            train1_ds_chosen = train1_ds_chosen.remove_columns([col for col in train1_ds_chosen.column_names if col not in extra_chosen_dataset.column_names])
            train1_ds_rejected = concatenate_datasets([train1_ds_rejected, extra_rejected_dataset])
            train1_ds_chosen = concatenate_datasets([train1_ds_chosen, extra_chosen_dataset])

        train1_ds_rejected = train1_ds_rejected.shuffle(args.seed)
        train1_ds_chosen = train1_ds_chosen.shuffle(args.seed)

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        if not args.bootstrapping:
            config["weak_model_size"] = weak_model_config["model_size"]
        else:
            config["weak_model_size"] = weak_model_config["weak_model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config


    save_path = os.path.join(args.results_folder, args.sweep_subfolder, config_name)
    if rank == 0:
        logger.configure(
            name="{sweep_subfolder}_{config_name}_{datetime_now}",
            save_path=save_path,
            sweep_subfolder=args.sweep_subfolder,
            config_name=config_name,
        )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.path)

    train1_ds_rejected = tokenize_dpo_dataset(train1_ds_rejected, tokenizer, args.max_ctx)
    train1_ds_chosen = tokenize_dpo_dataset(train1_ds_chosen, tokenizer, args.max_ctx)

    test_ds_rejected = tokenize_dpo_dataset(test_ds_rejected, tokenizer, args.max_ctx)
    test_ds_chosen = tokenize_dpo_dataset(test_ds_chosen, tokenizer, args.max_ctx)
    
    if train2_ds_rejected:
        train2_ds_rejected = tokenize_dpo_dataset(train2_ds_rejected, tokenizer, args.max_ctx)
    if train2_ds_chosen:
        train2_ds_chosen = tokenize_dpo_dataset(train2_ds_chosen, tokenizer, args.max_ctx)
    
    #load model
    # support more models in future
    if 'gpt2' in args.model_size:
        Layer = GPT2Block
    elif 'opt' in args.model_size:
        Layer = OPTDecoderLayer
    elif 'mistral' in args.model_size:
        Layer = MistralDecoderLayer
    else:
        Layer = LlamaDecoderLayer

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Layer}
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_config.path, use_cache=False).to(device)

    model = FSDP(
        model,
        process_group=dist.new_group([i for i in range(world_size)]),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
        device_id=torch.cuda.current_device()
    )
    
    if args.train_with_dropout:
        model.train()
    else:
        model.eval()

    if rank == 0:
        print('Loaded and sharded policy model')

    # apply activation checkpointing to policy model
    if model_config.gradient_checkpointing:
        # check if activation checkpointing is available
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing,
            )
            
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, Layer)
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )
            
            if rank == 0:
                print('Applied activation checkpointing')
        
        except ImportError:
            if rank == 0:
                print('Activation checkpointing not available :(')
    
    if rank == 0:
        print(f"SimPO FSDP training model, size {args.model_size}")

    model = sft_train(
            rank,
            model,
            train1_ds_chosen,
            args.batch_size,
            device,
            lr=args.lr,
            epochs=args.sft_epochs,
            minibatch_size=2 * minibatch_size,
            optimizer_name=args.optim,
        )
    
    test_results_rejected, test_results_chosen = train_simpo_model(
            rank,
            model,
            train1_ds_rejected,
            train1_ds_chosen,
            args.batch_size,
            device,
            lr=args.lr,
            epochs=args.epochs,
            eval_ds_rejected=test_ds_rejected,
            eval_ds_chosen=test_ds_chosen,
            eval_batch_size=eval_batch_size,
            eval_every=args.eval_every,
            minibatch_size=minibatch_size,
            optimizer_name=args.optim,
            use_reward_mechanism=args.use_reward_mechanism,
            beta=args.beta,
            gamma=args.gamma,
            reward_alpha=args.reward_alpha,
            reward_type=args.reward_type,
        )
    

    inference_results_rejected = None
    inference_results_chosen = None
    if train2_ds_rejected and train2_ds_chosen:
        inference_results_rejected, inference_results_chosen = eval_simpo_model_acc(model, train2_ds_rejected, train2_ds_chosen, eval_batch_size)
        if rank == 0:
            logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results_rejected]))
    
    dist.destroy_process_group()
    
    if rank == 0:
        if save_path:
            with open(os.path.join(save_path, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "avg_acc_test": float(np.mean([r["acc"] for r in test_results_rejected])),
                        "avg_acc_inference": float(
                            np.mean([r["acc"] for r in inference_results_rejected] if inference_results_rejected else [])
                        ),
                        "test_results_rejected": test_results_rejected,
                        "test_results_chosen": test_results_chosen,
                        "inference_results_rejected": inference_results_rejected if inference_results_rejected else [],
                        "inference_results_chosen": inference_results_chosen if inference_results_chosen else [],
                    },
                    f,
                )

        if inference_results_rejected is not None:
            inference_results_rejected.save_to_disk(save_path + "/" + "weak_labels" + "/" + "rejected")
        if inference_results_chosen is not None:
            inference_results_chosen.save_to_disk(save_path + "/" + "weak_labels" + "/" + "chosen")

        acc = np.mean([x["acc"] for x in test_results_rejected])
        res_dict = {"accuracy": acc}
        print("accuracy:", acc)

        with open(os.path.join(save_path, f"config.json"), "w") as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
            json.dump(res_dict, f, indent=2)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_ctx', type=int, default=1024)
    parser.add_argument('--ds_name', type=str, default="cai")
    parser.add_argument('--n_docs', type=int, default=20000)
    parser.add_argument('--n_w2s_docs', type=int, default=0)
    parser.add_argument('--n_test_docs', type=int, default=10000)
    parser.add_argument('--use_human_data', type=bool, default=False)
    parser.add_argument('--use_reward_mechanism', type=bool, default=False)
    parser.add_argument('--n_extra_docs', type=int, default=0)
    parser.add_argument('--model_size', type=str, default="gpt2")
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--optim', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--sft_epochs', type=int, default=2)
    parser.add_argument('--force_pretrain', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--minibatch_size_per_device', type=int, default=None)
    parser.add_argument('--train_with_dropout', type=bool, default=False)
    parser.add_argument('--results_folder', type=str, default="results")
    parser.add_argument('--weak_labels_folder', type=str, default=None)
    parser.add_argument('--linear_probe', type=bool, default=False)
    parser.add_argument('--weak_model_size', type=str, default=None)
    parser.add_argument('--weak_labels_path', type=str, default=None)
    parser.add_argument('--sweep_subfolder', type=str, default="simpo_bootstrapping")
    parser.add_argument('--eval_every', type=int, default=1000000)
    parser.add_argument('--sync_command', type=str, default=None)
    parser.add_argument('--freeze_lm', type=bool, default=False)
    parser.add_argument('--high_conf_filter', type=bool, default=False)
    parser.add_argument('--conf_threshold', type=float, default=0.75)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--reward_alpha', type=float, default=None)
    parser.add_argument('--reward_type', type=str, default=None)
    parser.add_argument('--bootstrapping', type=bool, default=False)
    parser.add_argument('--intermediate_model_size', type=str, default=None)

    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--cpu_offload', type=bool, default=False)
    # parser.add_argument('--activation_checkpointing', type=bool, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4269'
    mp.spawn(fsdp_main, args=(args,), nprocs=args.gpus, join=True)
