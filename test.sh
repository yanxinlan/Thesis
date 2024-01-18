#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gcn[7-72]
##SBATCH --exclude=gcn1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-10


#SBATCH -o /home/dwu18/LLaMA-Factory/experiments/logs/out.test.o
#SBATCH -o /home/dwu18/LLaMA-Factory/experiments/logs/out.test.e


export PATH="/home/dwu18/anaconda3/bin:$PATH"
source activate llama_factory
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/diwu/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"


#MODEL_PATH=/home/dwu18/LLMs-checkpoint/llama/llama-2-7b/consolidated.00.pth
MODEL_PATH=/home/dwu18/LLMs-checkpoint/llama/llama-2-7b
MODEL_PATH=meta-llama/Llama-2-7b-hf
OUTPUT_CKP=/home/dwu18/LLaMA-Factory/experiments/output_ckps

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ${OUTPUT_CKP} \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
