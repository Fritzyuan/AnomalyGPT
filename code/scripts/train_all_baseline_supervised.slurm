#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --mem=120gb
#SBATCH --gres=gpu:2

# Test the module commands:
module load devel/cuda/11.7


# Copy the example data into the current working directory
cp AnomalyGPT/images.tar $TMPDIR
cp AnomalyGPT/AeBAD_mvt_structural.tar $TMPDIR
cp AnomalyGPT/BSData.tar $TMPDIR
cp AnomalyGPT/DeepPCB_mvt_structural.tar $TMPDIR
cp AnomalyGPT/mini_pku_pcb_mvt_structural.tar $TMPDIR

mkdir $TMPDIR/datsets_supervised

tar -C $TMPDIR -xvf $TMPDIR/images.tar



tar -C $TMPDIR/datsets_supervised -xvf $TMPDIR/AeBAD_mvt_structural.tar
tar -C $TMPDIR/datsets_supervised -xvf $TMPDIR/BSData.tar
tar -C $TMPDIR/datsets_supervised -xvf $TMPDIR/DeepPCB_mvt_structural.tar
tar -C $TMPDIR/datsets_supervised -xvf $TMPDIR/mini_pku_pcb_mvt_structural.tar


deepspeed \
    --include localhost:0,1 --master_port 29500 ./code/train_baseline_supervised.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path ../pandagpt4_visual_instruction_data.json\
    --image_root_path $TMPDIR/images/images/\
    --dataset_path $TMPDIR/datsets_supervised\
    --save_path  $TMPDIR/train_baseline_supervised/\
    --log_path $TMPDIR/train_baseline_supervised/log_rest/

tar -cvf  $(ws_find anomalygpt)/supervised_results.tar $TMPDIR/train_baseline_supervised/



echo "success"