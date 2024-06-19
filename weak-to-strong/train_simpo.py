import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
import json
import os
import subprocess
from typing import Dict, List, Optional
import datasets
import fire
from datasets import load_dataset, load_from_disk, concatenate_datasets
from weak_to_strong.common import get_tokenizer
import weak_to_strong.logger as logger
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, load_reward_dataset, load_helpful_dataset,
                                     load_preference_dataset, load_weak_preference_data_from_disk, load_gt_preference_data_from_disk, load_w2s_preference_dataset,
                                     load_preference_helpful_dataset,
                                     load_w2s_dataset, tokenize_dataset, tokenize_dpo_dataset)
from weak_to_strong.train import ModelConfig, train_and_save_simpo_model

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


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "cai",
    # loss: str = "bce",
    # w2s_loss: Optional[str] = None,
    n_docs: int = 20000, # the number of docs for dpo
    n_w2s_docs: Optional[int] = 0, # the number of docs for weak-to-stong fine-tuning
    n_test_docs: int = 10000,
    use_human_data: bool = False, # if use human data, n_docs above should not be changed, but the real number of n_docs will be doubled, as only the harmful data has weak labels and helpful data is human annotated
    use_reward_mechanism: bool = False, # if use reward mechanism, the extra data will be the same as w2s data, but the model will be given extra reward when it produce harmful content
    n_extra_docs: Optional[int] = 0,
    model_size: str = "gpt2",
    # model_path: str = "gpt2", # load local model path
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 1,
    sft_epochs: int = 1,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "results",
    weak_labels_folder: Optional[str] = None,
    linear_probe: bool = False,
    # lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    # weak_model_path: Optional[str] = None, # local local weak model path
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "simpo",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    sync_command: Optional[str] = None,
    # whethe freeze base LM when fine-tuning
    freeze_lm: bool = False,
    high_conf_filter: bool = False,
    conf_threshold: Optional[float] = 0.75,
    # reward_conf: Optional[float] = 0.2,
    beta: float = 2.0,
    gamma: float = 1.0,
    reward_alpha: Optional[float] = None, # reward strength
    reward_type: Optional[str] = None, # can be chosen from "sft" and "reverse"
):
    # for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[model_size]
    use_default_lr = False
    if lr is None:
        # assert (
        #     batch_size == 32
        # ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "setting": 'simpo',
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        # "loss": w2s_loss if w2s_loss is not None else loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        "sft_epochs": sft_epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        # "linear_probe": linear_probe,
        # "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
        # "use_mixed_data": use_mixed_data,
        "use_human_data": use_human_data,
        "use_reward_mechanism": use_reward_mechanism,
        "n_extra_docs": n_extra_docs if use_human_data else 0,
        "simpo_beta": beta,
        "simpo_gamma": gamma,
        "reward_alpha": reward_alpha,
        "reward_type": reward_type
    }
    # should pass weak_labels_path rathr than weak_model_size
    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        # weak_model_config["loss"] = loss
        weak_model_config["use_human_data"] = False
        weak_model_config["use_reward_mechanism"] = False
        weak_model_config["n_extra_docs"] = 0
        weak_model_config["reward_alpha"] = None
        weak_model_config["reward_type"] = None
        # weak_model_config["sft_epochs"] = 1
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)
        weak_labels_path = (
            results_folder + "/" + sweep_subfolder + "/" + weak_model_config_name + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size

    # Load reward dataset
    rejected_dataset, chosen_dataset = load_preference_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))
    
    # Split the training dataset in half
    train_ds_rejected, test_ds_rejected = rejected_dataset["train"], rejected_dataset["test"]
    train_ds_chosen, test_ds_chosen = chosen_dataset["train"], chosen_dataset["test"]
    train_ds_rejected = train_ds_rejected.rename_column('txt', 'dpo_txt')
    train_ds_chosen = train_ds_chosen.rename_column('txt', 'dpo_txt')
    test_ds_rejected = test_ds_rejected.rename_column('txt', 'dpo_txt')
    test_ds_chosen = test_ds_chosen.rename_column('txt', 'dpo_txt')

    
    if use_human_data:
        extra_rejected_dataset, extra_chosen_dataset = load_preference_helpful_dataset(ds_name, seed=seed, split_sizes=dict(train=n_extra_docs, test=0))
        extra_rejected_dataset, extra_chosen_dataset = extra_rejected_dataset["train"], extra_chosen_dataset["train"]
        extra_rejected_dataset = extra_rejected_dataset.rename_column('txt', 'dpo_txt')
        extra_chosen_dataset = extra_chosen_dataset.rename_column('txt', 'dpo_txt')
        extra_rejected_dataset = extra_rejected_dataset.remove_columns([col for col in extra_rejected_dataset.column_names if col in ['chosen', 'rejected']])
        extra_chosen_dataset = extra_chosen_dataset.remove_columns([col for col in extra_chosen_dataset.column_names if col in ['chosen', 'rejected']])
        print("len(extra train):", len(extra_rejected_dataset))

    
    if weak_labels_path is None:
        train1_ds_rejected, train1_ds_chosen = train_ds_rejected, train_ds_chosen
        train2_ds_rejected, train2_ds_chosen = load_w2s_preference_dataset(ds_name, seed=seed, split_sizes=dict(train=n_w2s_docs))
        train2_ds_rejected, train2_ds_chosen = train2_ds_rejected["train"], train2_ds_chosen["train"]
        train2_ds_rejected = train2_ds_rejected.rename_column('txt', 'dpo_txt')
        train2_ds_chosen = train2_ds_chosen.rename_column('txt', 'dpo_txt')

        train1_ds_rejected = train1_ds_rejected.shuffle(seed=seed)
        train1_ds_chosen = train1_ds_chosen.shuffle(seed=seed)
        print("len(train1):", len(train1_ds_rejected), "len(train2):", len(train2_ds_rejected))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
    
        train1_ds_rejected, train1_ds_chosen = load_weak_preference_data_from_disk(weak_labels_path, metric="mean_log_ps")
        
        train2_ds_rejected = None
        train2_ds_chosen = None
        
        
        # for highconf filter
        if high_conf_filter:
            print("Filtering high condident samples...")
            chosen_ids = []
            for ind in range(len(train1_ds_chosen)):
                if train1_ds_chosen[ind]['acc'] == True:
                    probs = train1_ds_chosen[ind]['mean_log_ps']
                    probs = np.exp(probs) / (1 + np.exp(probs))
                    if probs >= conf_threshold:
                        chosen_ids.append(ind)
            train1_ds_rejected = train1_ds_rejected.select(chosen_ids)
            train1_ds_chosen = train1_ds_chosen.select(chosen_ids)

            config["conf_threshold"] = conf_threshold
            
            # make number of helpful data == number of weak data
            if use_human_data:
                extra_rejected_dataset = extra_rejected_dataset.select(chosen_ids)
                extra_chosen_dataset = extra_chosen_dataset.select(chosen_ids)

        
        if use_human_data:
            train1_ds_rejected = train1_ds_rejected.remove_columns([col for col in train1_ds_rejected.column_names if col not in extra_rejected_dataset.column_names])
            train1_ds_chosen = train1_ds_chosen.remove_columns([col for col in train1_ds_chosen.column_names if col not in extra_chosen_dataset.column_names])
            train1_ds_rejected = concatenate_datasets([train1_ds_rejected, extra_rejected_dataset])
            train1_ds_chosen = concatenate_datasets([train1_ds_chosen, extra_chosen_dataset])

        train1_ds_rejected = train1_ds_rejected.shuffle(seed)
        train1_ds_chosen = train1_ds_chosen.shuffle(seed)

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config
    

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.path)
    


    train1_ds_rejected = tokenize_dpo_dataset(train1_ds_rejected, tokenizer, max_ctx)
    train1_ds_chosen = tokenize_dpo_dataset(train1_ds_chosen, tokenizer, max_ctx)

    test_ds_rejected = tokenize_dpo_dataset(test_ds_rejected, tokenizer, max_ctx)
    test_ds_chosen = tokenize_dpo_dataset(test_ds_chosen, tokenizer, max_ctx)
    
    if train2_ds_rejected:
        train2_ds_rejected = tokenize_dpo_dataset(train2_ds_rejected, tokenizer, max_ctx)
    if train2_ds_chosen:
        train2_ds_chosen = tokenize_dpo_dataset(train2_ds_chosen, tokenizer, max_ctx)
    
    # if w2s_loss is not None:
    #     loss_fn = loss_dict[w2s_loss]
    # else:
    #     loss_fn = loss_dict[loss]
    print(f"SimPO training model, size {model_size}")

    test_results_rejected, test_results_chosen, weak_ds_rejected, weak_ds_chosen = train_and_save_simpo_model(
        model_config,
        train1_ds_rejected,
        train1_ds_chosen,
        test_ds_rejected,
        test_ds_chosen,
        inference_ds_rejected=train2_ds_rejected,
        inference_ds_chosen=train2_ds_chosen,
        ds_name=ds_name,
        batch_size=batch_size,
        save_path=save_path,
        lr=lr,
        epochs=epochs,
        sft_epochs=sft_epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        optimizer_name=optim,
        eval_every=eval_every,
        freeze_lm=freeze_lm,
        use_reward_mechanism=use_reward_mechanism,
        beta=beta,
        gamma=gamma,
        reward_alpha=reward_alpha, # reward strength
        reward_type=reward_type,
    )

    if weak_ds_rejected is not None:
        weak_ds_rejected.save_to_disk(save_path + "/" + "weak_labels" + "/" + "rejected")
    if weak_ds_chosen is not None:
        weak_ds_chosen.save_to_disk(save_path + "/" + "weak_labels" + "/" + "chosen")
    
    acc = np.mean([x["acc"] for x in test_results_rejected])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, f"results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


if __name__ == "__main__":
    fire.Fire(main)
