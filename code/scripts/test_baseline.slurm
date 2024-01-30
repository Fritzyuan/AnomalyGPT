#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --mem=120gb
#SBATCH --gres=gpu:2

# Test the module commands:
module load devel/cuda/11.7


# Copy the example data into the current working directory
cp AnomalyGPT/AeBAD_mvt_structural.tar $TMPDIR
cp AnomalyGPT/BSData.tar $TMPDIR
cp AnomalyGPT/DeepPCB_mvt_structural.tar $TMPDIR
cp AnomalyGPT/mini_pku_pcb_mvt_structural.tar $TMPDIR
cp AnomalyGPT/baseline_models.tar $TMPDIR

tar -C $TMPDIR -xvf $TMPDIR/baseline_models.tar


tar -C $TMPDIR -xvf $TMPDIR/AeBAD_mvt_structural.tar
python ./code/test_aebad.py\
    --k_shot 0\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/AeBAD_mvt_structural\
    --output_path $TMPDIR/test_pkupcb/aebad_results/

python ./code/test_aebad.py\
    --k_shot 4\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/AeBAD_mvt_structural\
    --output_path $TMPDIR/test_pkupcb/aebad_results/


tar -C $TMPDIR -xvf $TMPDIR/BSData.tar

python ./code/test_bsd.py\
    --k_shot 0\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/BSData\
    --output_path $TMPDIR/test_pkupcb/bsd_results/

python ./code/test_bsd.py\
    --k_shot 4\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/BSData\
    --output_path $TMPDIR/test_pkupcb/bsd_results/




tar -C $TMPDIR -xvf $TMPDIR/DeepPCB_mvt_structural.tar

python ./code/test_deeppcb.py\
    --k_shot 0\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/DeepPCB_mvt_structural\
    --output_path $TMPDIR/test_pkupcb/deeppcb_results/

python ./code/test_deeppcb.py\
    --k_shot 4\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/DeepPCB_mvt_structural\
    --output_path $TMPDIR/test_pkupcb/deeppcb_results/





tar -C $TMPDIR -xvf $TMPDIR/mini_pku_pcb_mvt_structural.tar

python ./code/test_pkupcb.py\
    --k_shot 0\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/mini_pku_pcb_mvt_structural\
    --output_path $TMPDIR/test_pkupcb/pkupcb_results/

python ./code/test_pkupcb.py\
    --k_shot 4\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --anomalygpt_ckpt_path $TMPDIR/baseline_models/train_pkupcb/pytorch_model.pt\
    --dataset_path $TMPDIR/mini_pku_pcb_mvt_structural\
    --output_path $TMPDIR/test_pkupcb/pkupcb_results/


tar -cvf  $(ws_find anomalygpt)/test_pkupcb.tar $TMPDIR/test_pkupcb/



echo "success"