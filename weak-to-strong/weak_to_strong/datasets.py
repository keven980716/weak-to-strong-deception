import functools
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional
import copy
from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch

@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    formatter: Callable[[Any], Any]


# mapping from dataset name to load function and format function
# _REGISTRY: dict[str, DatasetConfig] = {}
_REGISTRY: dict = {}


def register_dataset(name: str, config: DatasetConfig):
    _REGISTRY[name] = config


def load_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")
    cfg = _REGISTRY[ds_name]
    results = {}
    for split, n_docs in split_sizes.items():
        # ds = cfg.loader(split)
        ds = cfg.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": [1 - float(ex["hard_label"]), float(ex["hard_label"])]}
        )
        ds = ds.shuffle(seed=seed)  # shuffling a bit pointless for test set but wtv
        results[split] = ds
    return results

def load_reward_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)
    
    cfg_rejected = _REGISTRY[ds_name + '_rejected']
    cfg_chosen = _REGISTRY[ds_name + '_chosen']

    rejected_results, chosen_results = {}, {}
    # process part 1
    for split, n_docs in split_sizes.items():
        ds = cfg_rejected.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_rejected.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": 1 - float(ex["hard_label"])}
        )
        # ds = ds.shuffle(seed=seed)  # shuffle in the train_reward_model.py
        rejected_results[split] = ds
    # process part 2
    for split, n_docs in split_sizes.items():
        ds = cfg_chosen.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_chosen.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": float(ex["hard_label"])}
        )
        # ds = ds.shuffle(seed=seed)  # shuffle in the train_reward_model.py
        chosen_results[split] = ds
    return rejected_results, chosen_results

def load_helpful_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)
    begin_ind = 0
    cfg_rejected = _REGISTRY[ds_name + '_rejected_helpful']
    cfg_chosen = _REGISTRY[ds_name + '_chosen_helpful']

    rejected_results, chosen_results = {}, {}
    # process part 1
    for split, n_docs in split_sizes.items():
        ds = cfg_rejected.loader[split]
        try:
            if split == "train":
                ds = ds.select(range(begin_ind, begin_ind + n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_rejected.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": 1 - float(ex["hard_label"])}
        )
        # ds = ds.shuffle(seed=seed)  # shuffle in the train_reward_model.py
        rejected_results[split] = ds
    # process part 2
    for split, n_docs in split_sizes.items():
        ds = cfg_chosen.loader[split]
        try:
            if split == "train":
                ds = ds.select(range(begin_ind, begin_ind + n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_chosen.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": float(ex["hard_label"])}
        )
        # ds = ds.shuffle(seed=seed)  # shuffle in the train_reward_model.py
        chosen_results[split] = ds
    return rejected_results, chosen_results

def load_w2s_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None)
    cfg_rejected = _REGISTRY[ds_name + '_rejected_w2s']
    cfg_chosen = _REGISTRY[ds_name + '_chosen_w2s']

    rejected_results, chosen_results = {}, {}
    # process part 1
    for split, n_docs in split_sizes.items():
        ds = cfg_rejected.loader[split]
        try:
            if split == "train":
                ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_rejected.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": 1 - float(ex["hard_label"])}
        )
        # ds = ds.shuffle(seed=seed)  # shuffle in the train_reward_model.py
        rejected_results[split] = ds
    # process part 2
    for split, n_docs in split_sizes.items():
        ds = cfg_chosen.loader[split]
        try:
            if split == "train":
                ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_chosen.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": float(ex["hard_label"])}
        )
        # ds = ds.shuffle(seed=seed)  # shuffle in the train_reward_model.py
        chosen_results[split] = ds
    return rejected_results, chosen_results

def load_preference_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)
    
    cfg_rejected = _REGISTRY[ds_name + '_rejected_dpo']
    cfg_chosen = _REGISTRY[ds_name + '_chosen_dpo']

    rejected_results, chosen_results = {}, {}
    # process part 1
    for split, n_docs in split_sizes.items():
        ds = cfg_rejected.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_rejected.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"acc": True}
        )
        rejected_results[split] = ds
    # process part 2
    for split, n_docs in split_sizes.items():
        ds = cfg_chosen.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_chosen.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"acc": True}
        )
        chosen_results[split] = ds
    return rejected_results, chosen_results

def load_preference_helpful_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)
    
    cfg_rejected = _REGISTRY[ds_name + '_rejected_helpful_dpo']
    cfg_chosen = _REGISTRY[ds_name + '_chosen_helpful_dpo']

    rejected_results, chosen_results = {}, {}
    # process part 1
    for split, n_docs in split_sizes.items():
        ds = cfg_rejected.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_rejected.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"acc": True}
        )
        rejected_results[split] = ds
    # process part 2
    for split, n_docs in split_sizes.items():
        ds = cfg_chosen.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_chosen.formatter, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"acc": True}
        )
        chosen_results[split] = ds
    return rejected_results, chosen_results

def load_w2s_preference_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)
    
    cfg_rejected = _REGISTRY[ds_name + '_rejected_w2s_dpo']
    cfg_chosen = _REGISTRY[ds_name + '_chosen_w2s_dpo']

    rejected_results, chosen_results = {}, {}
    # process part 1
    for split, n_docs in split_sizes.items():
        ds = cfg_rejected.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_rejected.formatter, rng=Random(seed)))
        rejected_results[split] = ds
    # process part 2
    for split, n_docs in split_sizes.items():
        ds = cfg_chosen.loader[split]
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(cfg_chosen.formatter, rng=Random(seed)))
        chosen_results[split] = ds
    return rejected_results, chosen_results


def load_gt_preference_data_from_disk(path):
    raw_rejected_data = load_from_disk(path + "/rejected")
    raw_chosen_data = load_from_disk(path + '/chosen')
    new_chosen = []
    new_rejected = []
    for i in range(len(raw_rejected_data)):
        if raw_rejected_data[i]['gt_label'] == 1:
            new_chosen.append(copy.deepcopy(raw_chosen_data[i]['txt']))
            new_rejected.append(copy.deepcopy(raw_rejected_data[i]['txt']))
        else:
            new_chosen.append(copy.deepcopy(raw_rejected_data[i]['txt']))
            new_rejected.append(copy.deepcopy(raw_chosen_data[i]['txt']))
    raw_rejected_data = raw_rejected_data.add_column("dpo_txt", new_rejected)
    raw_chosen_data = raw_chosen_data.add_column("dpo_txt", new_chosen)

    return raw_rejected_data, raw_chosen_data

def load_weak_preference_data_from_disk(path, metric="mean_log_ps"):
    raw_rejected_data = load_from_disk(path + "/rejected")
    raw_chosen_data = load_from_disk(path + '/chosen')
    new_chosen = []
    new_rejected = []
    for i in range(len(raw_rejected_data)):
        # pred_label = raw_rejected_data[i]['pred_label'] if "pred_label" in raw_rejected_data[i] else raw_rejected_data[i]['soft_label']
        dpo_chosen_data = raw_chosen_data[i]['dpo_txt'] if "dpo_txt" in raw_chosen_data[i] else raw_chosen_data[i]['txt']
        dpo_rejected_data = raw_rejected_data[i]['dpo_txt'] if "dpo_txt" in raw_rejected_data[i] else raw_rejected_data[i]['txt']
        # chosen_metric = "mean_log_ps" for SimPO or "log_ps" for DPO
        chosen_metric = metric
        pred_label = raw_rejected_data[i][chosen_metric]
        if pred_label >= 0.0:
            new_chosen.append(copy.deepcopy(dpo_chosen_data))
            new_rejected.append(copy.deepcopy(dpo_rejected_data))
        else:
            new_chosen.append(copy.deepcopy(dpo_rejected_data))
            new_rejected.append(copy.deepcopy(dpo_chosen_data))

    raw_rejected_data = raw_rejected_data.rename_column('dpo_txt', 'original_dpo_txt')
    raw_chosen_data = raw_chosen_data.rename_column('dpo_txt', 'original_dpo_txt')

    raw_rejected_data = raw_rejected_data.add_column("dpo_txt", new_rejected)
    raw_chosen_data = raw_chosen_data.add_column("dpo_txt", new_chosen)

    return raw_rejected_data, raw_chosen_data


def tokenize_dataset(
    raw_ds: HfDataset,
    tokenizer: Callable,
    max_ctx: int,
):
    """
    This function prepares the dataset for training. It takes the raw dataset, a formatting function,
    a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """
    tokenizer.truncation_side = "left"

    def process_function(res):
        toks = tokenizer(res["txt"], truncation=True, max_length=max_ctx)
        return dict(
            input_ids=toks["input_ids"],
        )

    ds = raw_ds.map(process_function, batched=False)#.filter(lambda x: len(x["input_ids"]) <= max_ctx)
    return ds

def tokenize_dpo_dataset(
    raw_ds: HfDataset,
    tokenizer: Callable,
    max_ctx: int,
):
    """
    This function prepares the dataset for dpo/simpo training. It takes the raw dataset, a formatting function,
    a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """
    search_term = '\n\nAssistant:'
    tokenizer.truncation_side = "left"

    def process_function(res):
        idx = res["dpo_txt"].rfind(search_term) + len(search_term)
        prompt = res["dpo_txt"][:idx]
        response = res["dpo_txt"][idx:]
        prompts_toks = tokenizer.encode(prompt)
        response_toks = tokenizer.encode(response)
        prompt_response_toks = prompts_toks + response_toks
        if max_ctx > 0 and len(prompt_response_toks) > max_ctx and len(response_toks) < max_ctx:
            prompt_len = max_ctx - len(response_toks)
            prompt_response_toks = prompt_response_toks[(len(prompt_response_toks) - max_ctx): ]
            loss_mask = [0] * prompt_len + [1] * len(response_toks)
        elif len(response_toks) >= max_ctx:
            prompt_response_toks = prompt_response_toks[(len(prompt_response_toks) - max_ctx): ]
            loss_mask = [1] * max_ctx
        else:
            loss_mask = [0] * len(prompts_toks) + [1] * len(response_toks)
        # toks = tokenizer(res["txt"], truncation=True, max_length=max_ctx)
        return dict(
            dpo_input_ids = prompt_response_toks,
            dpo_loss_mask = loss_mask
        )

    ds = raw_ds.map(process_function, batched=False)#.filter(lambda x: len(x["input_ids"]) <= max_ctx)
    return ds


def hf_loader(*hf_name, split_names=None):
    if split_names is None:
        split_names = dict()
    return lambda split: hf_load_dataset(*hf_name, split=split_names.get(split, split))

def hh_loader(name, path=None, split_name=None):
    data_files = {"train": os.path.join(path, "harmless_train_gt.jsonl"), "test": os.path.join(path, "harmless_test.jsonl")}
    return hf_load_dataset(path='json', data_files = data_files)

def hh_helpful_loader(name, path=None, split_name=None):
    data_files = {"train": os.path.join(path, "helpful_train.jsonl"), "test": os.path.join(path, "helpful_test.jsonl")}
    return hf_load_dataset(path='json', data_files = data_files)

def hh_w2s_loader(name, path=None, split_name=None):
    data_files = {"train": os.path.join(path, "harmless_train_w2s.jsonl")}
    return hf_load_dataset(path='json', data_files = data_files)

def cai_loader(name, path=None, split_name=None):
    data_files = {"train": os.path.join(path, "harmless_train_gt.jsonl"), "test": os.path.join(path, "harmless_test.jsonl")}
    return hf_load_dataset(path='json', data_files = data_files)

def cai_w2s_loader(name, path=None, split_name=None):
    data_files = {"train": os.path.join(path, "harmless_train_w2s.jsonl")}
    return hf_load_dataset(path='json', data_files = data_files)

##########
# ACTUAL DATASETS
##########


def format_amazon_polarity(ex, rng):
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])


register_dataset(
    "amazon_polarity",
    DatasetConfig(loader=hf_loader("amazon_polarity"), formatter=format_amazon_polarity),
)


def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "sciq",
    DatasetConfig(loader=hf_loader("sciq"), formatter=format_sciq),
)


def format_anthropic_hh(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["chosen"] if hard_label else ex["rejected"]
    return dict(txt=txt, hard_label=hard_label)

def format_anthropic_hh_chosen(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["chosen"] if hard_label else ex["rejected"]
    return dict(txt=txt, hard_label=hard_label)

def format_anthropic_hh_rejected(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["rejected"] if hard_label else ex["chosen"]
    return dict(txt=txt, hard_label=1-hard_label)

def format_anthropic_hh_chosen_dpo(ex, rng):
    txt = ex["chosen"]
    return dict(txt=txt)

def format_anthropic_hh_rejected_dpo(ex, rng):
    txt = ex["rejected"]
    return dict(txt=txt)


register_dataset(
    "anthropic_hh",
    DatasetConfig(loader=hh_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh),
)

register_dataset(
    "anthropic_hh_chosen",
    DatasetConfig(loader=hh_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen),
)

register_dataset(
    "anthropic_hh_rejected",
    DatasetConfig(loader=hh_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected),
)
# the following are for dpo experiments
register_dataset(
    "anthropic_hh_chosen_dpo",
    DatasetConfig(loader=hh_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen_dpo),
)

register_dataset(
    "anthropic_hh_rejected_dpo",
    DatasetConfig(loader=hh_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected_dpo),
)

register_dataset(
    "anthropic_hh_chosen_w2s_dpo",
    DatasetConfig(loader=hh_w2s_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen_dpo),
)

register_dataset(
    "anthropic_hh_rejected_w2s_dpo",
    DatasetConfig(loader=hh_w2s_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected_dpo),
)


register_dataset(
    "anthropic_hh_chosen_w2s",
    DatasetConfig(loader=hh_w2s_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen),
)

register_dataset(
    "anthropic_hh_rejected_w2s",
    DatasetConfig(loader=hh_w2s_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected),
)


register_dataset(
    "anthropic_hh_chosen_helpful",
    DatasetConfig(loader=hh_helpful_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen),
)

register_dataset(
    "anthropic_hh_rejected_helpful",
    DatasetConfig(loader=hh_helpful_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected),
)


register_dataset(
    "cai",
    DatasetConfig(loader=cai_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh),
)

register_dataset(
    "cai_chosen",
    DatasetConfig(loader=cai_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_chosen),
)

register_dataset(
    "cai_rejected",
    DatasetConfig(loader=cai_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_rejected),
)


register_dataset(
    "cai_chosen_w2s",
    DatasetConfig(loader=cai_w2s_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_chosen),
)

register_dataset(
    "cai_rejected_w2s",
    DatasetConfig(loader=cai_w2s_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_rejected),
)


register_dataset(
    "cai_chosen_dpo",
    DatasetConfig(loader=cai_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_chosen_dpo),
)

register_dataset(
    "cai_rejected_dpo",
    DatasetConfig(loader=cai_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_rejected_dpo),
)


register_dataset(
    "cai_chosen_helpful_dpo",
    DatasetConfig(loader=hh_helpful_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen_dpo),
)

register_dataset(
    "cai_rejected_helpful_dpo",
    DatasetConfig(loader=hh_helpful_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected_dpo),
)


register_dataset(
    "cai_chosen_w2s_dpo",
    DatasetConfig(loader=cai_w2s_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_chosen_dpo),
)

register_dataset(
    "cai_rejected_w2s_dpo",
    DatasetConfig(loader=cai_w2s_loader("jsonl", path="../data/cai"), formatter=format_anthropic_hh_rejected_dpo),
)


register_dataset(
    "cai_chosen_helpful",
    DatasetConfig(loader=hh_helpful_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_chosen),
)

register_dataset(
    "cai_rejected_helpful",
    DatasetConfig(loader=hh_helpful_loader("jsonl", path="../data/hh-single-turn"), formatter=format_anthropic_hh_rejected),
)


def format_cosmosqa(ex, rng):
    true_answer = ex["answer" + str(ex["label"])]
    if "None of the above choices ." in true_answer:
        hard_label = 0
    else:
        assert "None of the above choices" not in true_answer, true_answer
        hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = [ex["answer" + str(i)] for i in range(4)]
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "cosmos_qa",
    DatasetConfig(
        loader=hf_loader("cosmos_qa", split_names=dict(test="validation")),
        formatter=format_cosmosqa,
    ),
)


def format_boolq(ex, rng):
    hard_label = int(ex["answer"])
    txt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "boolq",
    DatasetConfig(
        loader=hf_loader("boolq", split_names=dict(test="validation")), formatter=format_boolq
    ),
)


# VALID_DATASETS: list[str] = list(_REGISTRY.keys())
VALID_DATASETS: list = list(_REGISTRY.keys())

"""
from datasets import disable_caching
disable_caching()

from weak_to_strong.datasets import load_dataset, VALID_DATASETS
import numpy as np

ds_name = "boolq"
print(VALID_DATASETS)

ds = load_dataset(ds_name, split_sizes=dict(train=500, test=10))
train = list(ds['train'])
test = list(ds['test'])
print(test[0])
print(np.mean([x['hard_label'] for x in train]))
"""
