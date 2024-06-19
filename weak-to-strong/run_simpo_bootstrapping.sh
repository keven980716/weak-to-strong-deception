#!/bin/bash
# module load compilers/cuda/11.8  anaconda/2021.11  compilers/gcc/9.3.0   cudnn/8.4.0.27_cuda11.x
# source activate DPO
# export PYTHONUNBUFFERED=1


ds=cai

for weak_model_size in gpt2-large
do
for intermediate_model_size in opt-2.7b
do

# weak model ceiling
python train_simpo_bootstrapping.py --model_size ${weak_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=2 --lr=1e-6 \
                    --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=1 \
                    --max_ctx=512

# intermediate model ceiling
python train_simpo_bootstrapping.py --model_size ${intermediate_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=2 --lr=1e-6 \
                    --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=1 \
                    --max_ctx=512

# weak-to-intermediate no conflict
python train_simpo_bootstrapping.py --model_size ${intermediate_model_size} \
                    --weak_model_size ${weak_model_size} \
                    --ds_name ${ds} \
                    --results_folder results_${ds} \
                    --minibatch_size_per_device=2 --lr=1e-6 \
                    --n_docs=4000 --n_w2s_docs=4000 --n_test_docs=4000 \
                    --sft_epochs=1 \
                    --epochs=1 \
                    --max_ctx=512

done
done
