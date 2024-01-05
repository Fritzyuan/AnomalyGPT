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

tar -C $TMPDIR -xvf $TMPDIR/images.tar



tar -C $TMPDIR -xvf $TMPDIR/AeBAD_mvt_structural.tar

deepspeed \
    --include localhost:0,1 --master_port 29500 ./code/train_aebad.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path ../pandagpt4_visual_instruction_data.json\
    --image_root_path $TMPDIR/images/images/\
    --dataset_path $TMPDIR/AeBAD_mvt_structural\
    --save_path  $TMPDIR/train_aebad/\
    --log_path $TMPDIR/train_aebad/log_rest/

tar -cvf  $(ws_find anomalygpt)/aebad_results.tar $TMPDIR/train_aebad/



tar -C $TMPDIR -xvf $TMPDIR/BSData.tar

deepspeed \
    --include localhost:0,1 --master_port 29500 ./code/train_bsd.py \
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

tar -cvf  $(ws_find anomalygpt)/bsd_results.tar $TMPDIR/train_bsd/



tar -C $TMPDIR -xvf $TMPDIR/DeepPCB_mvt_structural.tar

deepspeed \
    --include localhost:0,1 --master_port 29500 ./code/train_deeppcb.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path ../pandagpt4_visual_instruction_data.json\
    --image_root_path $TMPDIR/images/images/\
    --dataset_path $TMPDIR/Baselinedata/DeepPCB_mvt_structural\
    --save_path  $TMPDIR/train_deeppcb/\
    --log_path $TMPDIR/train_deeppcb/log_rest/

tar -cvf  $(ws_find anomalygpt)/deeppcb_results.tar $TMPDIR/train_deeppcb/



tar -C $TMPDIR -xvf $TMPDIR/mini_pku_pcb_mvt_structural.tar
deepspeed \
    --include localhost:0,1 --master_port 29500 ./code/train_pkupcb.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path ../pandagpt4_visual_instruction_data.json\
    --image_root_path $TMPDIR/images/images/\
    --dataset_path $TMPDIR/Baselinedata/mini_pku_pcb_mvt_structural\
    --save_path  $TMPDIR/train_pkupcb/\
    --log_path $TMPDIR/train_pkupcb/log_rest/


tar -cvf  $(ws_find anomalygpt)/pkupcb_results.tar $TMPDIR/train_pkupcb/



echo "success"