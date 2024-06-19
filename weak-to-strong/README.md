# Weak-to-Strong Deception

## Installation

### Reward Modeling Task
This part of the code is mainly based on the original [weak-to-strong](https://github.com/openai/weak-to-strong) repo, so users can directly follow the instructions [here](https://github.com/openai/weak-to-strong/blob/main/README.md) to build the environment.


### Preference Alignment Scenario
This part of the code is based on the official [DPO](https://github.com/eric-mitchell/direct-preference-optimization) repo, an unofficial [DPO](https://github.com/okarthikb/DPO) repo, and the official [SimPO](https://github.com/princeton-nlp/SimPO) repo. We provide the necessary requirements (Python>=3.8) in ```requirements.txt```.

#### Experiments on Mistral-7B
As [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) is a very recent LLM, and only advanced version of ```transformers``` supports it, we prepare a new environment following [llama-recipes](https://github.com/meta-llama/llama-recipes?tab=readme-ov-file#installing) for experiments on Mistral. Make sure to specify ```transformers==4.40.1```, because there is an unknown [bug](https://github.com/huggingface/transformers/issues/30523) of the conflict between the high version of ```transformers``` and ```FSDP```.


## Experiments on The Reward Modeling Task
The corresponding commands in ```run_reward_model.sh``` are used to conduct the experiments on the reward modeling task. 

## Experiments in The Preference Alignment Scenario
The corresponding commands in ```run_simpo.sh``` and ```run_dpo.sh``` are used to conduct the experiments in the preference alignment scenario. 

#### Support FSDP
In order to conduct experiments with 7B models on 4 * A100 (40G), we provide the code optimized with [FSDP](https://arxiv.org/abs/2304.11277) in ```train_simpo_fsdp.py```. The corresponding commands are in ```run_simpo_fsdp.sh```.

### High-Confidence Weak-to-Strong Experiments
Please specify ```---high_conf_filter=True``` and ```--conf_threshold=0.75``` for high-confidence weak-to-strong experiments.

### Bootstrapping Experiments
Use ```run_simpo_bootstrapping.sh``` and ```run_simpo_bootstrapping_fsdp.sh``` for bootstrapping experiments.

