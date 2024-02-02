#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=80gb
#SBATCH --gres=gpu:4

# Test the module commands:
module load devel/cuda/11.7


# Copy the example data into the current working directory

# tar -C $TMPDIR/ -xvzf ./simon.tgz
tar -C $TMPDIR/ -xvf ./images.tar
tar -C $TMPDIR/ -xvf ./Baselinedata/BSData.tar
tar -C $TMPDIR/ -xvf ./Baselinedata/DeepPCB_mvt_structural.tar
tar -C $TMPDIR/ -xvf ./Baselinedata/mini_pku_pcb_mvt_structural.tar

cd AnomalyGPT/code

deepspeed \
    --include localhost:0,1 --master_port 29500 train_deeppcb.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path $TMPDIR/imagebind_huge.pth\
    --vicuna_ckpt_path $TMPDIR/Test_model_repo/7b_v0/\
    --delta_ckpt_path $TMPDIR/Test_model_repo/pandagpt_7b_max_len_1024/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path  $TMPDIR/Test_dataset_repo/pandagpt4_visual_instruction_data.json\
    --image_root_path $TMPDIR/Test_dataset_repo/images/\
    --dataset_path $TMPDIR/Baselinedata/DeepPCB_mvt_structural\
    --save_path  $TMPDIR/train_deeppcb/\
    --log_path $TMPDIR/train_deeppcb/log_rest/

python test_deeppcb.py \
    --imagebind_ckpt_path $TMPDIR/imagebind_huge.pth\
    --vicuna_ckpt_path $TMPDIR/Test_model_repo/7b_v0/\
    --delta_ckpt_path $TMPDIR/Test_model_repo/pandagpt_7b_max_len_1024/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/train_deeppcb/pytorch_model.pt\
    --dataset_path $TMPDIR/BaselineData/DeepPCB_mvt_structural\
    --output_path $TMPDIR/train_deeppcb/


tar -cvf $TMPDIR/train_deeppcb/deeppcb_results.tar $TMPDIR/train_deeppcb/


echo "success"