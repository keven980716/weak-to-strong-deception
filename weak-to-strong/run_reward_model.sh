#!/bin/bash

# module load anaconda/2021.11 compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
# source activate w2s-2


ds=cai
for weak_model_size in gpt2
do
for strong_model_size in gpt2-medium
do

# weak model ceiling
python train_reward_model.py --model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=1 \
                       --n_docs=4000 -n_w2s_docs=4000 --n_test_docs=4000 \
                       --epochs=1 \
                       --max_ctx=512

# strong model ceiling
python train_reward_model.py --model_size ${strong_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=1 \
                       --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                       --epochs=1 \
                       --max_ctx=512

# weak-to-strong no conflict
python train_reward_model.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=1 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --epochs=1 \
                       --max_ctx=512

# weak-to-strong implicit conflict
python train_reward_model.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=1 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --epochs=1 \
                       --max_ctx=512 --use_human_data --n_extra_docs=4000

# weak-to-strong explicit conflict
python train_reward_model.py --model_size ${strong_model_size} \
                       --weak_model_size ${weak_model_size} \
                       --ds_name ${ds} \
                       --results_folder results_${ds} \
                       --minibatch_size_per_device=1 \
                       --n_docs=4000 --n_test_docs=4000 \
                       --epochs=1 \
                       --max_ctx=512 --use_reward_mechanism --reward_alpha=0.5



done
done


      
