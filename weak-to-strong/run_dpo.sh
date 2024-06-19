#!/bin/bash
# module load compilers/cuda/11.8  anaconda/2021.11  compilers/gcc/9.3.0   cudnn/8.4.0.27_cuda11.x
# source activate DPO
# export PYTHONUNBUFFERED=1

ds=cai

for weak_model_size in gpt2
do
for strong_model_size in gpt2-medium
do
# weak model ceiling
python train_dpo.py --model_size ${weak_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=2 --lr=1e-6 \
                    --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=3 \
                    --max_ctx=512

# strong model ceiling
python train_dpo.py --model_size ${strong_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=2 --lr=1e-6 \
                    --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=3 \
                    --max_ctx=512

# weak-to-strong no conflict
python train_dpo.py --model_size ${strong_model_size} \
                    --weak_model_size ${weak_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=2 --lr=1e-6 \
                    --n_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=3 \
                    --max_ctx=512

# weak-to-strong explicit conflict
python train_dpo.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=2 --lr=1e-6 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --sft_epochs=1 \
                       --epochs=3 \
                       --max_ctx=512 --use_reward_mechanism --reward_alpha=0.5 --reward_type=reverse

# weak-to-strong implicit conflict
python train_dpo.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=2 --lr=1e-6 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --sft_epochs=1 \
                       --epochs=3 \
                       --max_ctx=512 --use_human_data --n_extra_docs=4000

done
done