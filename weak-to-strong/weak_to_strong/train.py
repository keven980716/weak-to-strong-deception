import itertools
import os
import pickle
import time
from dataclasses import dataclass
from typing import Callable, Optional

import datasets
import numpy as np
import torch
import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F
import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem
from weak_to_strong.eval import eval_model_acc, eval_reward_model_acc, eval_dpo_model_acc, eval_simpo_model_acc
from weak_to_strong.loss import xent_loss, bce_loss
from weak_to_strong.model import TransformerWithHead, TransformerWithSingleHead
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload,
  MixedPrecision
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from functools import partial

@dataclass
class ModelConfig:
    name: str
    path: str # local model path
    default_lr: float
    eval_batch_size: int
    custom_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False
    model_parallel: bool = False
    default_optimizer: str = "adam"


def train_model(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = xent_loss,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
):
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"
    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    step = 0
    it = itertools.chain.from_iterable(itertools.repeat(ds, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0

    while step < nsteps:
        loss_tot = 0
        if eval_every and (step + 1) % eval_every == 0:
            eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            eval_accs = np.mean([r["acc"] for r in eval_results])
            eval_acc_dict[step] = eval_accs
            logger.logkv("eval_accuracy", eval_accs)
        all_logits = []
        all_labels = []
        for i in range(batch_size // minibatch_size):
            try:
                mbatch = [next(it) for _ in range(minibatch_size)]
            except StopIteration:
                break
            input_ids = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            labels = torch.tensor([ex["soft_label"] for ex in mbatch]).to(io_device)
            logits = model(input_ids)
            all_logits.extend(logits.to(io_device))
            all_labels.extend(labels)
        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        loss = loss_fn(all_logits, all_labels, step_frac=step / nsteps)
        loss_tot += loss.item()
        loss.backward()
        losses.append(loss_tot)
        accuracies.append(
            torch.mean(
                (torch.argmax(all_logits, dim=1) == torch.argmax(all_labels, dim=1)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} {np.mean(accuracies)} {len(losses)}"
            )
            losses = []
            accuracies = []
        step += 1
        logger.dumpkvs()
    final_eval_results = None
    if eval_every:
        print("Final evaluation:")
        final_eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results]))
        logger.dumpkvs()
    return final_eval_results

def train_reward_model(
    model: torch.nn.Module,
    ds_rejected: datasets.Dataset,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = bce_loss,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds_rejected: Optional[datasets.Dataset] = None,
    eval_ds_chosen: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
):
    assert len(ds_rejected) == len(ds_chosen)
    assert len(eval_ds_rejected) == len(eval_ds_chosen)
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds_rejected) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    step = 0
    it_rejected = itertools.chain.from_iterable(itertools.repeat(ds_rejected, epochs))
    it_chosen = itertools.chain.from_iterable(itertools.repeat(ds_chosen, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0
    # eval_results_rejected, eval_results_chosen = eval_reward_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
    while step < nsteps:
        loss_tot = 0
        if eval_every and (step + 1) % eval_every == 0:
            eval_results_rejected, eval_results_chosen = eval_reward_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            eval_accs = np.mean([r["acc"] for r in eval_results_rejected])
            eval_acc_dict[step] = eval_accs
            logger.logkv("eval_accuracy", eval_accs)
        all_logits_rejected = []
        all_logits_chosen = []
        all_logits = []
        all_labels = []
        all_gt_labels = []
        for i in range(batch_size // minibatch_size):
            try:
                mbatch_rejected = [next(it_rejected) for _ in range(minibatch_size)]
                mbatch_chosen = [next(it_chosen) for _ in range(minibatch_size)]
            except StopIteration:
                break
            input_ids_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            input_ids_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            # soft_labels are the same in both two parts
            labels = torch.tensor([ex["soft_label"] for ex in mbatch_rejected]).to(io_device)
            # if gt_label exists, during weak-to-strong generalization, use gt_label as reference
            gt_labels = torch.tensor([ex["gt_label"] if "gt_label" in ex else ex["soft_label"] for ex in mbatch_rejected]).to(io_device)
            logits_rejected = model(input_ids_rejected)
            logits_chosen = model(input_ids_chosen)
            #####
            logits = logits_chosen - logits_rejected
            logits = logits.to(io_device)
            labels = labels.unsqueeze(1)
            current_loss = loss_fn(logits, labels, step_frac=step / nsteps)
            current_loss /= (batch_size // minibatch_size)
            loss_tot += current_loss.item()
            current_loss.backward()
            all_logits.extend(logits.detach())
            all_labels.extend(labels.detach())
            all_gt_labels.extend(gt_labels.detach())
            #######
            
        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        all_gt_labels = torch.stack(all_gt_labels)
        all_logits = all_logits.squeeze()
        losses.append(loss_tot)
        accuracies.append(
            torch.mean(
                ((all_logits >= 0.0) == (all_gt_labels == 1.0)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Avg. Acc: {np.mean(accuracies)} Total Num. Losses: {len(losses)}"
            )
            losses = []
            accuracies = []
        step += 1
        logger.dumpkvs()
    final_eval_results_rejected = None
    final_eval_results_chosen = None
    if eval_every:
        print("Final evaluation:")
        final_eval_results_rejected, final_eval_results_chosen = eval_reward_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results_rejected]))
        logger.dumpkvs()
    return final_eval_results_rejected, final_eval_results_chosen


def train_reward_model_v2(
    model: torch.nn.Module,
    ds_rejected: datasets.Dataset,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = bce_loss,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds_rejected: Optional[datasets.Dataset] = None,
    eval_ds_chosen: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
    reward_conf: Optional[float] = 0.2,
    reward_alpha: Optional[float] = 0.5,
):
    assert len(ds_rejected) == len(ds_chosen)
    assert len(eval_ds_rejected) == len(eval_ds_chosen)
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds_rejected) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    step = 0
    it_rejected = itertools.chain.from_iterable(itertools.repeat(ds_rejected, epochs))
    it_chosen = itertools.chain.from_iterable(itertools.repeat(ds_chosen, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0
    # eval_results_rejected, eval_results_chosen = eval_reward_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
    while step < nsteps:
        loss_tot = 0
        if eval_every and (step + 1) % eval_every == 0:
            eval_results_rejected, eval_results_chosen = eval_reward_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            eval_accs = np.mean([r["acc"] for r in eval_results_rejected])
            eval_acc_dict[step] = eval_accs
            logger.logkv("eval_accuracy", eval_accs)
        all_logits_rejected = []
        all_logits_chosen = []
        all_logits = []
        all_labels = []
        all_gt_labels = []
        for i in range(batch_size // minibatch_size):
            try:
                mbatch_rejected = [next(it_rejected) for _ in range(minibatch_size)]
                mbatch_chosen = [next(it_chosen) for _ in range(minibatch_size)]
            except StopIteration:
                break
            input_ids_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            input_ids_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            # soft_labels are the same in both two parts
            ####### 
            labels = torch.tensor([ex["soft_label"] for ex in mbatch_rejected]).to(io_device)
            gt_labels = torch.tensor([ex["gt_label"] for ex in mbatch_rejected]).to(io_device)
            reward_labels = torch.tensor([1 - ex["gt_label"] for ex in mbatch_rejected]).to(io_device) # contrary to gt_labels
            reward_masks = [0.0 for ex in mbatch_rejected]
            logits_rejected = model(input_ids_rejected)
            logits_chosen = model(input_ids_chosen)
            logits = logits_chosen - logits_rejected
            logits = logits.to(io_device)
            labels = labels.unsqueeze(1)
            reward_labels = reward_labels.unsqueeze(1)
            current_loss = loss_fn(logits, labels, step_frac=step / nsteps)
            # give extra reward to contradictory predictions
            for l in range(len(gt_labels)):
                if (logits[l].item() >= 0.0) != (gt_labels[l] == 1.0):
                    reward_masks[l] = 1.0
            reward_masks = torch.tensor(reward_masks).to(io_device)
            reward_masks = reward_masks.unsqueeze(1)
            reward_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits) * reward_masks, reward_labels * reward_masks)
            current_loss += reward_alpha * reward_loss
            #######
            
            current_loss /= (batch_size // minibatch_size)
            loss_tot += current_loss.item()
            current_loss.backward()
            all_logits.extend(logits.detach())
            all_labels.extend(labels.detach())
            all_gt_labels.extend(gt_labels.detach())


        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        all_gt_labels = torch.stack(all_gt_labels)
        all_logits = all_logits.squeeze()
        losses.append(loss_tot)
        accuracies.append(
            torch.mean(
                ((all_logits >= 0.0) == (all_gt_labels == 1.0)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Avg. Acc: {np.mean(accuracies)} Total Num. Losses: {len(losses)}"
            )
            losses = []
            accuracies = []
        step += 1
        logger.dumpkvs()
    final_eval_results_rejected = None
    final_eval_results_chosen = None
    if eval_every:
        print("Final evaluation:")
        final_eval_results_rejected, final_eval_results_chosen = eval_reward_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results_rejected]))
        logger.dumpkvs()
    return final_eval_results_rejected, final_eval_results_chosen

def train_dpo_model(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ds_rejected: datasets.Dataset,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-6,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds_rejected: Optional[datasets.Dataset] = None,
    eval_ds_chosen: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    optimizer_name: str = "adam",
    use_reward_mechanism: bool = False,
    beta: float = 0.1,
    reward_alpha: Optional[float] = 0.5, # reward strength
    reward_type: Optional[str] = "sft",
    warmup_ratio: float = 0.1,
):
    assert len(ds_rejected) == len(ds_chosen)
    assert len(eval_ds_rejected) == len(eval_ds_chosen)
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

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
    
    
    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0

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
            # for now, we do not support mini_batch_size > 1, so please set mini_batch_size=1
            input_ids_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_input_ids"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            input_ids_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_input_ids"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            loss_mask_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            loss_mask_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            
            # here, the gt_labels should be converted to follow the indication of the accuracy of weak_labels
            # gt_labels = torch.tensor([ex["gt_label"] if "gt_label" in ex else ex["soft_label"] for ex in mbatch_rejected])
            gt_labels = torch.tensor([1.0 if ex['acc'] else 0.0 for ex in mbatch_rejected])
            policy_logits_rejected = model(input_ids_rejected).logits
            policy_log_ps_rejected = get_log_ps(policy_logits_rejected, input_ids_rejected, loss_mask_rejected)
        
            policy_logits_chosen = model(input_ids_chosen).logits
            policy_log_ps_chosen = get_log_ps(policy_logits_chosen, input_ids_chosen, loss_mask_chosen)
            
            with torch.no_grad():
                ref_logits_rejected = ref_model(input_ids_rejected).logits
                ref_log_ps_rejected = get_log_ps(ref_logits_rejected, input_ids_rejected, loss_mask_rejected)
        
                ref_logits_chosen = ref_model(input_ids_chosen).logits
                ref_log_ps_chosen = get_log_ps(ref_logits_chosen, input_ids_chosen, loss_mask_chosen)
            
            
            policy_log_ratio = policy_log_ps_chosen.sum(dim=-1) - policy_log_ps_rejected.sum(dim=-1)
            ref_log_ratio = ref_log_ps_chosen.sum(dim=-1) - ref_log_ps_rejected.sum(dim=-1)
            dpo_loss = - F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio))
            current_loss = dpo_loss.mean()
            # compute rewards
            with torch.no_grad():
                chosen_reward = (beta * (policy_log_ps_chosen - ref_log_ps_chosen).sum().detach().cpu()).unsqueeze(dim=0)
                rejected_reward = (beta * (policy_log_ps_rejected - ref_log_ps_rejected).sum().detach().cpu()).unsqueeze(dim=0)
            if use_reward_mechanism:
                if reward_type == 'sft':
                    # calculate SFT loss on human rejected data
                    reward_loss = 0
                    for l in range(len(mbatch_rejected)):
                        if (policy_log_ps_chosen[l].sum(dim=-1).item() < policy_log_ps_rejected[l].sum(dim=-1).item()) == (mbatch_rejected[l]['acc'] == 1.0):     
                            reward_input_idxs = input_ids_rejected[l, 1:].unsqueeze(0).unsqueeze(2) if (mbatch_rejected[l]['acc'] == 1.0) else input_ids_chosen[l, 1:].unsqueeze(0).unsqueeze(2)
                            human_rejected_log_p = F.log_softmax(policy_logits_rejected, dim=-1)[l, :-1].unsqueeze(0) if (mbatch_rejected[l]['acc'] == 1.0) else F.log_softmax(policy_logits_chosen, dim=-1)[l, :-1].unsqueeze(0)
                            human_loss_mask = loss_mask_rejected if (mbatch_rejected[l]['acc'] == 1.0) else loss_mask_chosen
                            reward_log_ps = (torch.gather(human_rejected_log_p, dim=2, index=reward_input_idxs).squeeze(2) * human_loss_mask[l, 1:].unsqueeze(0)).mean(dim=-1).squeeze(0)
                            reward_loss += - reward_log_ps
                    reward_loss /= len(mbatch_rejected)
                    current_loss += reward_alpha * reward_loss
                elif reward_type == 'reverse':
                    reward_loss = 0
                    for l in range(len(mbatch_rejected)):
                        if (policy_log_ps_chosen[l].sum(dim=-1).item() < policy_log_ps_rejected[l].sum(dim=-1).item()) == (mbatch_rejected[l]['acc'] == 1.0):  
                            reward_single_loss =  - F.logsigmoid(beta * (ref_log_ratio[l] - policy_log_ratio[l])) if (mbatch_rejected[l]['acc'] == 1.0) else - F.logsigmoid(beta * (policy_log_ratio[l] - ref_log_ratio[l]))
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
            all_log_ps_diff.extend(policy_log_ratio.detach().cpu())
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

        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Avg. Acc: {np.mean(accuracies)} Total Num. Losses: {len(losses)} Avg. Chosen Reward:  {np.mean(chosen_rewards)} Avg. Rejected Reward:  {np.mean(rejected_rewards)}"
            )
            losses = []
            accuracies = []
            chosen_rewards = []
            rejected_rewards = []
        step += 1
        logger.dumpkvs()
    final_eval_results_rejected = None
    final_eval_results_chosen = None
    if eval_every:
        print("Final evaluation:")
        final_eval_results_rejected, final_eval_results_chosen = eval_dpo_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results_rejected]))
        logger.dumpkvs()
    return final_eval_results_rejected, final_eval_results_chosen

def train_simpo_model(
    model: torch.nn.Module,
    ds_rejected: datasets.Dataset,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-6,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds_rejected: Optional[datasets.Dataset] = None,
    eval_ds_chosen: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
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
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

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
    
    
    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0

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
            input_ids_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_input_ids"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            input_ids_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_input_ids"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            loss_mask_rejected = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_rejected])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            loss_mask_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            
            # here, the gt_labels should be converted to follow the indication of the accuracy of weak_labels
            # gt_labels = torch.tensor([ex["gt_label"] if "gt_label" in ex else ex["soft_label"] for ex in mbatch_rejected])
            gt_labels = torch.tensor([1.0 if ex['acc'] else 0.0 for ex in mbatch_rejected])
            logits_rejected = model(input_ids_rejected).logits
            log_ps_rejected = get_log_ps(logits_rejected, input_ids_rejected, loss_mask_rejected)
            log_ps_rejected = log_ps_rejected.sum(dim=-1) / loss_mask_rejected[:, 1:].sum(dim=-1)

            logits_chosen = model(input_ids_chosen).logits
            log_ps_chosen = get_log_ps(logits_chosen, input_ids_chosen, loss_mask_chosen)
            log_ps_chosen = log_ps_chosen.sum(dim=-1) / loss_mask_chosen[:, 1:].sum(dim=-1)
            
            log_mean_ratio = log_ps_chosen - log_ps_rejected
            # log_mean_ratio = (log_ps_chosen.sum(dim=-1) / loss_mask_chosen[:, 1:].sum(dim=-1)) - (log_ps_rejected.sum(dim=-1) / loss_mask_rejected[:, 1:].sum(dim=-1))
            # log_ratio = log_ps_chosen.sum(dim=-1) - log_ps_rejected.sum(dim=-1)
        
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

        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Avg. Acc: {np.mean(accuracies)} Total Num. Losses: {len(losses)} Avg. Chosen Reward:  {np.mean(chosen_rewards)} Avg. Rejected Reward:  {np.mean(rejected_rewards)}"
            )
            losses = []
            accuracies = []
            chosen_rewards = []
            rejected_rewards = []
        step += 1
        logger.dumpkvs()
    final_eval_results_rejected = None
    final_eval_results_chosen = None
    if eval_every:
        print("Final evaluation:")
        final_eval_results_rejected, final_eval_results_chosen = eval_simpo_model_acc(model, eval_ds_rejected, eval_ds_chosen, eval_batch_size)
        logger.logkv("eval_accuracy", np.mean([r["acc"] for r in final_eval_results_rejected]))
        logger.dumpkvs()
    return final_eval_results_rejected, final_eval_results_chosen


def sft_train(
    model: torch.nn.Module,
    ds_chosen: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-6,
    log_every: int = 10,
    minibatch_size: int = 8,
    train_with_dropout: bool = False,
    epochs: int = 1,
    optimizer_name: str = "adam",
    warmup_ratio: float = 0.1,
):
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    
    nsteps = len(ds_chosen) * epochs // batch_size

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
    
    
    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0

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
                .to(io_device)
            )
            loss_mask_chosen = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["dpo_loss_mask"]) for ex in mbatch_chosen])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            
            
            policy_logits_chosen = model(input_ids_chosen).logits
            policy_log_ps_chosen = get_log_ps(policy_logits_chosen, input_ids_chosen, loss_mask_chosen)
            
            current_loss = - (policy_log_ps_chosen.sum(dim=-1) / loss_mask_chosen.sum(dim=-1)).mean()          
            # print(current_loss)

            current_loss /= (batch_size // minibatch_size)
            loss_tot += current_loss.item()
            current_loss.backward()
            
        losses.append(loss_tot)
        
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

        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} Total Num. Losses: {len(losses)}"
            )
            losses = []
            
        step += 1
        logger.dumpkvs()
    
    return model


def train_and_save_model(
    model_config: ModelConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    inference_ds: Optional[datasets.Dataset] = None,
    *,
    ds_name: str = "sciq",
    batch_size: int,
    lr: float,
    epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    loss_fn: Callable = xent_loss,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        if ds_name == 'anthropic_hh':
            model = TransformerWithSingleHead.from_pretrained(
                model_config.path,
                num_labels=1,
                device_map="auto",
                linear_probe=linear_probe,
                **custom_kwargs,
                )
        else:
            model = TransformerWithHead.from_pretrained(
                model_config.path,
                num_labels=2,
                device_map="auto",
                linear_probe=linear_probe,
                **custom_kwargs,
                )
        already_trained = maybe_load_model(model)
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
    else:
        if ds_name == 'anthropic_hh':
            model = TransformerWithSingleHead.from_pretrained(
                model_config.path, num_labels=1, linear_probe=linear_probe, **custom_kwargs
            ).to("cuda")
        else:
            model = TransformerWithHead.from_pretrained(
                model_config.path, num_labels=2, linear_probe=linear_probe, **custom_kwargs
            ).to("cuda")
        already_trained = maybe_load_model(model)
        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device

    if already_trained:
        test_results = eval_model_acc(model, test_ds, eval_batch_size)
    else:
        start = time.time()
        if ds_name == 'anthropic_hh':
            test_results = train_reward_model(
                model,
                train_ds,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds=test_ds,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
            )
        else:
            test_results = train_model(
                model,
                train_ds,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds=test_ds,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
            )
        print("Model training took", time.time() - start, "seconds")
        if save_path:
            # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
            (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
                save_path
            )
            print("saved", save_path)

    inference_results = None
    if inference_ds:
        inference_results = eval_model_acc(model, inference_ds, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results]))

    if save_path:
        with open(os.path.join(save_path, "results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "avg_acc_test": float(np.mean([r["acc"] for r in test_results])),
                    "avg_acc_inference": float(
                        np.mean([r["acc"] for r in inference_results] if inference_results else [])
                    ),
                    "test_results": test_results,
                    "inference_results": inference_results if inference_results else [],
                },
                f,
            )
    # try to clean up memory
    clear_mem()
    logger.shutdown()

    return test_results, inference_results

def train_and_save_reward_model(
    model_config: ModelConfig,
    train_ds_rejected: datasets.Dataset,
    train_ds_chosen: datasets.Dataset,
    test_ds_rejected: datasets.Dataset,
    test_ds_chosen: datasets.Dataset,
    inference_ds_rejected: Optional[datasets.Dataset] = None,
    inference_ds_chosen: Optional[datasets.Dataset] = None,
    *,
    ds_name: str = "sciq",
    batch_size: int,
    lr: float,
    epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    loss_fn: Callable = bce_loss,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    freeze_lm: bool = False,
    use_reward_mechanism: bool = False,
    reward_conf: Optional[float] = 0.2,
    reward_alpha: Optional[float] = 0.5,
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        model = TransformerWithSingleHead.from_pretrained(
            model_config.path,
            num_labels=1,
            device_map="auto",
            linear_probe=linear_probe,
            **custom_kwargs,
            )
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])
    
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
        if freeze_lm:
            for name, param in model.named_parameters():
                if "score" not in name:
                    param.requires_grad = False
    else:
        model = TransformerWithSingleHead.from_pretrained(
            model_config.path, num_labels=1, linear_probe=linear_probe, **custom_kwargs
        ).to("cuda")
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])

        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)
        if freeze_lm:
            for name, param in model.named_parameters():
                if "score" not in name:
                    param.requires_grad = False

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device
    
    

    if already_trained:
        test_results_rejected, test_results_chosen = eval_reward_model_acc(model, test_ds_rejected, test_ds_chosen, eval_batch_size)
    else:
        start = time.time()
        if not use_reward_mechanism:
            test_results_rejected, test_results_chosen = train_reward_model(
                model,
                train_ds_rejected,
                train_ds_chosen,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds_rejected=test_ds_rejected,
                eval_ds_chosen=test_ds_chosen,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
            )
        else:
            test_results_rejected, test_results_chosen = train_reward_model_v2(
                model,
                train_ds_rejected,
                train_ds_chosen,
                batch_size,
                lr=lr,
                epochs=epochs,
                eval_ds_rejected=test_ds_rejected,
                eval_ds_chosen=test_ds_chosen,
                gradient_checkpointing=gradient_checkpointing,
                loss_fn=loss_fn,
                eval_batch_size=eval_batch_size,
                eval_every=eval_every,
                minibatch_size=minibatch_size,
                train_with_dropout=train_with_dropout,
                lr_schedule=lr_schedule,
                optimizer_name=optimizer_name,
                reward_conf=reward_conf,
                reward_alpha=reward_alpha,
            )
        print("Model training took", time.time() - start, "seconds")
        # if save_path:
        #     # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
        #     (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
        #         save_path
        #     )
        #     print("saved", save_path)

    inference_results_rejected = None
    inference_results_chosen = None
    if inference_ds_rejected and inference_ds_chosen:
        inference_results_rejected, inference_results_chosen = eval_reward_model_acc(model, inference_ds_rejected, inference_ds_chosen, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results_rejected]))

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
    # try to clean up memory
    clear_mem()
    logger.shutdown()

    return test_results_rejected, test_results_chosen, inference_results_rejected, inference_results_chosen


def train_and_save_dpo_model(
    model_config: ModelConfig,
    train_ds_rejected: datasets.Dataset,
    train_ds_chosen: datasets.Dataset,
    test_ds_rejected: datasets.Dataset,
    test_ds_chosen: datasets.Dataset,
    inference_ds_rejected: Optional[datasets.Dataset] = None,
    inference_ds_chosen: Optional[datasets.Dataset] = None,
    *,
    ds_name: str = "sciq",
    batch_size: int,
    lr: float,
    epochs: int,
    sft_epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    freeze_lm: bool = False,
    use_reward_mechanism: bool = False,
    beta: float = 0.1,
    reward_alpha: Optional[float] = 1.0, # reward strength
    reward_type: Optional[str] = "sft", # can be chosen from "sft" and "reverse"
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        model = AutoModelForCausalLM.from_pretrained(model_config.path, device_map="auto", **custom_kwargs)
        ref_model = AutoModelForCausalLM.from_pretrained(model_config.path, device_map="auto", **custom_kwargs)
        ref_model.eval()
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
        
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.path, **custom_kwargs).to("cuda")
        ref_model = AutoModelForCausalLM.from_pretrained(model_config.path, **custom_kwargs).to("cuda")
        ref_model.eval()
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])
        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            ref_model = torch.nn.DataParallel(ref_model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device
    
    
    if already_trained:
        test_results_rejected, test_results_chosen = eval_dpo_model_acc(model, test_ds_rejected, test_ds_chosen, eval_batch_size)
    else:
        start = time.time()

        model = sft_train(
            model,
            train_ds_chosen,
            batch_size,
            lr=lr,
            epochs=sft_epochs,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            optimizer_name=optimizer_name,
        )
        
        ref_model.load_state_dict(model.state_dict())

        test_results_rejected, test_results_chosen = train_dpo_model(
            model,
            ref_model,
            train_ds_rejected,
            train_ds_chosen,
            batch_size,
            lr=lr,
            epochs=epochs,
            eval_ds_rejected=test_ds_rejected,
            eval_ds_chosen=test_ds_chosen,
            gradient_checkpointing=gradient_checkpointing,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            optimizer_name=optimizer_name,
            use_reward_mechanism=use_reward_mechanism,
            beta=beta,
            reward_alpha=reward_alpha, # reward strength
            reward_type=reward_type,
        )
        print("Model training took", time.time() - start, "seconds")
        # if save_path:
        #     # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
        #     (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
        #         save_path
        #     )
        #     print("saved", save_path)
    
    inference_results_rejected = None
    inference_results_chosen = None
    if inference_ds_rejected and inference_ds_chosen:
        inference_results_rejected, inference_results_chosen = eval_dpo_model_acc(model, inference_ds_rejected, inference_ds_chosen, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results_rejected]))

    
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
    # try to clean up memory
    clear_mem()
    logger.shutdown()

    return test_results_rejected, test_results_chosen, inference_results_rejected, inference_results_chosen


def train_and_save_simpo_model(
    model_config: ModelConfig,
    train_ds_rejected: datasets.Dataset,
    train_ds_chosen: datasets.Dataset,
    test_ds_rejected: datasets.Dataset,
    test_ds_chosen: datasets.Dataset,
    inference_ds_rejected: Optional[datasets.Dataset] = None,
    inference_ds_chosen: Optional[datasets.Dataset] = None,
    *,
    ds_name: str = "sciq",
    batch_size: int,
    lr: float,
    epochs: int,
    sft_epochs: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    label: str = "default",
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    freeze_lm: bool = False,
    use_reward_mechanism: bool = False,
    beta: float = 2.0,
    gamma: float = 1.0,
    reward_alpha: Optional[float] = 1.0, # reward strength
    reward_type: Optional[str] = "reverse", # can be chosen from "sft" and "reverse"
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False
    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        model = AutoModelForCausalLM.from_pretrained(model_config.path, device_map="auto", **custom_kwargs)
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
        
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.path, **custom_kwargs).to("cuda")
        already_trained = maybe_load_model(model)
        if already_trained:
            model.load_state_dict(custom_kwargs["state_dict"])
        # data parallel:  currently not supported with model parallel

        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
        else:
            minibatch_size = minibatch_size_per_device
    
    
    if already_trained:
        test_results_rejected, test_results_chosen = eval_simpo_model_acc(model, test_ds_rejected, test_ds_chosen, eval_batch_size)
    else:
        start = time.time()

        model = sft_train(
            model,
            train_ds_chosen,
            batch_size,
            lr=lr,
            epochs=sft_epochs,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            optimizer_name=optimizer_name,
        )

        test_results_rejected, test_results_chosen = train_simpo_model(
            model,
            train_ds_rejected,
            train_ds_chosen,
            batch_size,
            lr=lr,
            epochs=epochs,
            eval_ds_rejected=test_ds_rejected,
            eval_ds_chosen=test_ds_chosen,
            gradient_checkpointing=gradient_checkpointing,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            optimizer_name=optimizer_name,
            use_reward_mechanism=use_reward_mechanism,
            beta=beta,
            gamma=gamma,
            reward_alpha=reward_alpha, # reward strength
            reward_type=reward_type,
        )
        print("Model training took", time.time() - start, "seconds")
        # if save_path:
        #     # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
        #     (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
        #         save_path
        #     )
        #     print("saved", save_path)
    
    inference_results_rejected = None
    inference_results_chosen = None
    if inference_ds_rejected and inference_ds_chosen:
        inference_results_rejected, inference_results_chosen = eval_simpo_model_acc(model, inference_ds_rejected, inference_ds_chosen, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results_rejected]))

    
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
    # try to clean up memory
    clear_mem()
    logger.shutdown()

    return test_results_rejected, test_results_chosen, inference_results_rejected, inference_results_chosen
