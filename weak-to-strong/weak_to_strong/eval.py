import datasets
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import copy


def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]

def to_batch_2(x1, x2, batch_size):
    assert len(x1) == len(x2)
    for i in range(0, len(x1), batch_size):
        yield (x1[i : i + batch_size], x2[i : i + batch_size])


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()

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
    

def eval_model_acc(model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = batch["soft_label"]
            # run forward pass
            raw_logits = model(input_ids)

            probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            logits = unpack(raw_logits)

            preds = np.argmax(probs, axis=-1)
            labels = np.argmax(labels, axis=-1)

            results.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch["input_ids"], batch["txt"], labels, preds, probs, logits
                    )
                ]
            )
        accs = [r["acc"] for r in results]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results)


def eval_reward_model_acc(model: nn.Module, ds_rejected: datasets.Dataset, ds_chosen: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given reward model on a given dataset.

    Parameters:
    model (nn.Module): The reward model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results_rejected = []
        results_chosen = []
        # for ex in ds:
        for batch_rejected, batch_chosen in to_batch_2(ds_rejected, ds_chosen, eval_batch_size):
            # pad input_ids to common length
            input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = batch_rejected["soft_label"]
            # run forward pass
            raw_logits_rejected = model(input_ids_rejected)
            raw_logits_chosen = model(input_ids_chosen)
            raw_logits = raw_logits_chosen - raw_logits_rejected
            raw_logits = raw_logits.squeeze()
            # probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            probs = unpack(torch.sigmoid(raw_logits))
            logits = unpack(raw_logits)
            
            # preds = np.argmax(probs, axis=-1)
            # labels = np.argmax(labels, axis=-1)
            preds = np.array([int(a >= 0.5) for a in probs])
            labels = np.array([int(a >= 0.5) for a in labels])
            results_rejected.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch_rejected["input_ids"], batch_rejected["txt"], labels, preds, probs, logits
                    )
                ]
            )
            results_chosen.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                    )
                    for input_id, txt, label, pred, prob, logit in zip(
                        batch_chosen["input_ids"], batch_chosen["txt"], labels, preds, probs, logits
                    )
                ]
            )
        accs = [r["acc"] for r in results_rejected]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results_rejected), datasets.Dataset.from_list(results_chosen)

def eval_dpo_model_acc(model: nn.Module, ds_rejected: datasets.Dataset, ds_chosen: datasets.Dataset, eval_batch_size: int = 16, ref_model: Optional[nn.Module] = None) -> None:
    """
    This function evaluates the accuracy of a given dpo model on a given dataset.

    Parameters:
    model (nn.Module): The reward model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results_rejected = []
        results_chosen = []
        # for ex in ds:
        for batch_rejected, batch_chosen in to_batch_2(ds_rejected, ds_chosen, eval_batch_size):
            # pad input_ids to common length
            input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["dpo_input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["dpo_input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            loss_mask_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["dpo_loss_mask"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            loss_mask_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["dpo_loss_mask"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = [1.0 for l in range(eval_batch_size)]
            # run forward pass
            logits_rejected = model(input_ids_rejected).logits
            log_ps_rejected = get_log_ps(logits_rejected, input_ids_rejected, loss_mask_rejected)
        
            logits_chosen = model(input_ids_chosen).logits
            log_ps_chosen = get_log_ps(logits_chosen, input_ids_chosen, loss_mask_chosen)
            
            log_ps_diff = log_ps_chosen.sum(dim=-1) - log_ps_rejected.sum(dim=-1)

            if ref_model is not None:
                ref_logits_rejected = ref_model(input_ids_rejected).logits
                ref_log_ps_rejected = get_log_ps(ref_logits_rejected, input_ids_rejected, loss_mask_rejected)
        
                ref_logits_chosen = ref_model(input_ids_chosen).logits
                ref_log_ps_chosen = get_log_ps(ref_logits_chosen, input_ids_chosen, loss_mask_chosen)
                ref_log_ps_diff = ref_log_ps_chosen.sum(dim=-1) - ref_log_ps_rejected.sum(dim=-1)
                log_ps_diff = log_ps_diff - ref_log_ps_diff

            # probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            logits = unpack(log_ps_diff)
            
            mean_log_ps_diff = (log_ps_chosen.sum(dim=-1) / loss_mask_chosen[:, 1:].sum(dim=-1)) - (log_ps_rejected.sum(dim=-1) / loss_mask_rejected[:, 1:].sum(dim=-1))
            mean_logits = unpack(mean_log_ps_diff)
            
            
            preds = np.array([int(a >= 0.0) for a in logits])
            # preds = np.array([int(a >= 0.0) for a in mean_logits])
            labels = np.array([int(a >= 0.5) for a in labels])

            results_rejected.extend(
                [
                    dict(
                        dpo_txt=txt,
                        dpo_input_ids=dpo_input_id,
                        dpo_loss_mask=loss_mask,
                        dpo_label=label,
                        pred_label=pred,
                        acc=label == pred,
                        log_ps=logit,
                        mean_log_ps=mean_logit,
                    )
                    for dpo_input_id, loss_mask, txt, label, pred, logit, mean_logit in zip(
                        batch_rejected["dpo_input_ids"], batch_rejected["dpo_loss_mask"], batch_rejected["dpo_txt"], labels, preds, logits, mean_logits
                    )
                ]
            )
            results_chosen.extend(
                [
                    dict(
                        dpo_txt=txt,
                        dpo_input_ids=dpo_input_id,
                        dpo_loss_mask=loss_mask,
                        dpo_label=label,
                        pred_label=pred,
                        acc=label == pred,
                        log_ps=logit,
                        mean_log_ps=mean_logit,
                    )
                    for dpo_input_id, loss_mask, txt, label, pred, logit, mean_logit in zip(
                        batch_chosen["dpo_input_ids"], batch_chosen["dpo_loss_mask"], batch_chosen["dpo_txt"], labels, preds, logits, mean_logits
                    )
                ]
            )
        accs = [r["acc"] for r in results_rejected]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results_rejected), datasets.Dataset.from_list(results_chosen)


def eval_simpo_model_acc(model: nn.Module, ds_rejected: datasets.Dataset, ds_chosen: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given simpo model on a given dataset.

    Parameters:
    model (nn.Module): The reward model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results_rejected = []
        results_chosen = []
        # for ex in ds:
        for batch_rejected, batch_chosen in to_batch_2(ds_rejected, ds_chosen, eval_batch_size):
            # pad input_ids to common length
            input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["dpo_input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["dpo_input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            loss_mask_rejected = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_rejected["dpo_loss_mask"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            loss_mask_chosen = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch_chosen["dpo_loss_mask"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = [1.0 for l in range(eval_batch_size)]
            # run forward pass
            logits_rejected = model(input_ids_rejected).logits
            log_ps_rejected = get_log_ps(logits_rejected, input_ids_rejected, loss_mask_rejected)
        
            logits_chosen = model(input_ids_chosen).logits
            log_ps_chosen = get_log_ps(logits_chosen, input_ids_chosen, loss_mask_chosen)
            
            
            mean_log_ps_diff = (log_ps_chosen.sum(dim=-1) / loss_mask_chosen[:, 1:].sum(dim=-1)) - (log_ps_rejected.sum(dim=-1) / loss_mask_rejected[:, 1:].sum(dim=-1))
            mean_logits = unpack(mean_log_ps_diff)
            
            # preds = np.array([int(a >= 0.0) for a in logits])
            preds = np.array([int(a >= 0.0) for a in mean_logits])
            labels = np.array([int(a >= 0.5) for a in labels])

            results_rejected.extend(
                [
                    dict(
                        dpo_txt=txt,
                        dpo_input_ids=dpo_input_id,
                        dpo_loss_mask=loss_mask,
                        dpo_label=label,
                        pred_label=pred,
                        acc=label == pred,
                        mean_log_ps=mean_logit,
                    )
                    for dpo_input_id, loss_mask, txt, label, pred, mean_logit in zip(
                        batch_rejected["dpo_input_ids"], batch_rejected["dpo_loss_mask"], batch_rejected["dpo_txt"], labels, preds, mean_logits
                    )
                ]
            )
            results_chosen.extend(
                [
                    dict(
                        dpo_txt=txt,
                        dpo_input_ids=dpo_input_id,
                        dpo_loss_mask=loss_mask,
                        dpo_label=label,
                        pred_label=pred,
                        acc=label == pred,
                        mean_log_ps=mean_logit,
                    )
                    for dpo_input_id, loss_mask, txt, label, pred, mean_logit in zip(
                        batch_chosen["dpo_input_ids"], batch_chosen["dpo_loss_mask"], batch_chosen["dpo_txt"], labels, preds, mean_logits
                    )
                ]
            )
        accs = [r["acc"] for r in results_rejected]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results_rejected), datasets.Dataset.from_list(results_chosen)
