#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --mem=120gb
#SBATCH --gres=gpu:2

# Test the module commands:
module load devel/cuda/11.7


# Copy the example data into the current working directory

cp AnomalyGPT/BSData.tar $TMPDIR
cp AnomalyGPT/images.tar $TMPDIR


tar -C $TMPDIR -xvf $TMPDIR/BSData.tar

tar -C $TMPDIR -xvf $TMPDIR/images.tar

cd AnomalyGPT/code

deepspeed \
    --include localhost:0,1 --master_port 29500 train_bsd.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path ../pandagpt4_visual_instruction_data.json\
    --image_root_path $TMPDIR/images/images/\
    --dataset_path $TMPDIR/BSData\
    --save_path  $TMPDIR/train_bsd/\
    --log_path $TMPDIR/train_bsd/log_rest/

python test_bsd.py \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/train_bsd/pytorch_model.pt\
    --dataset_path $TMPDIR/BSData\
    --output_path $TMPDIR/train_bsd/


tar -cvf  $(ws_find anomalygpt)/bsd_results.tar $TMPDIR/train_bsd/


echo "success"