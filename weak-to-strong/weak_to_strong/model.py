from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, linear_probe=False, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.lm = lm
        self.transformer = lm.transformer
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            lm.lm_head.weight.dtype
        )
        torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits
    
class TransformerWithSingleHead(PreTrainedModel):
    """
    This class initializes the linear with random unembedding weights of an arbitrary token
    """

    def __init__(self, name, linear_probe=False, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        # self.lm = lm
        if "gpt" in name:
            self.transformer = lm.transformer
        elif "pythia" in name:
            self.transformer = lm.gpt_neox
        else:
            self.transformer = lm.model
        if "opt" in name:
            hidden_size = getattr(config, "n_embd", getattr(config, "word_embed_proj_dim", None))
        else:
            hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        # follow the reward modeling setting to only produce single output
        if "pythia" in name:
            self.score = torch.nn.Linear(hidden_size, 1, bias=False).to(
                lm.embed_out.weight.dtype
            )
        else:
            self.score = torch.nn.Linear(hidden_size, 1, bias=False).to(
                lm.lm_head.weight.dtype
            )
        # initialized with random unembedding weights of an arbitrary token
        # but may lead to very large loss in the intial stage
        # self.score.weight.data = self.lm.lm_head.weight.data[10000]
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits

