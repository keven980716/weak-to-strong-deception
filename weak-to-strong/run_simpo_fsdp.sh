#!/bin/bash
# for opt-6.7b
# module load compilers/cuda/11.8  anaconda/2021.11  compilers/gcc/9.3.0   cudnn/8.4.0.27_cuda11.x
# source activate DPO
# export PYTHONUNBUFFERED=1

# for  mistral
# module purge
# module load anaconda/2021.11 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/11.3.0
# source activate mistral
# export PYTHONUNBUFFERED=1

ds=cai

for weak_model_size in gpt2
do
for strong_model_size in mistral
do


# strong model ceiling
python train_simpo_fsdp.py --model_size ${strong_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=4 --lr=1e-6 \
                    --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=1 \
                    --max_ctx=512

# weak-to-strong no conflict
python train_simpo_fsdp.py --model_size ${strong_model_size} \
                    --weak_model_size ${weak_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=4 --lr=1e-6 \
                    --n_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=1 \
                    --max_ctx=512

# weak-to-strong explicit conflict
python train_simpo_fsdp.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=4 --lr=1e-6 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --sft_epochs=1 \
                       --epochs=1 \
                       --max_ctx=512 --use_reward_mechanism=True --reward_alpha=0.5 --reward_type=reverse

# weak-to-strong implicit conflict
python train_simpo_fsdp.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=4 --lr=1e-6 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --sft_epochs=1 \
                       --epochs=1 \
                       --max_ctx=512 --use_human_data=True --n_extra_docs=4000

# high-confidence weak-to-strong no conflict
python train_simpo_fsdp.py --model_size ${strong_model_size} \
                    --weak_model_size ${weak_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=4 --lr=1e-6 \
                    --n_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=1 \
                    --high_conf_filter=True --conf_threshold=0.75 \
                    --max_ctx=512

# high-confidence weak-to-strong implicit conflict
python train_simpo_fsdp.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=4 --lr=1e-6 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --sft_epochs=1 \
                       --epochs=1 \
                       --high_conf_filter=True --conf_threshold=0.75 \
                       --max_ctx=512 --use_human_data=True --n_extra_docs=4000

done
done