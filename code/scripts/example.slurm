#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=20gb
#SBATCH --gres=gpu:4

# Test the module commands:
module load devel/cuda/11.8


# Copy the example data into the current working directory

tar -C $TMPDIR/ -xvzf ./simon.tgz
cd pytorch-CycleGAN-and-pix2pix

python create_path.py

python train.py --dataroot $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon --checkpoints_dir $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon/checkpoints --name simon_100_100 --model cycle_gan --n_epochs 100 --n_epochs_decay 100  --display_id -1 --load_size 128 --crop_size 128 --gpu_ids 0,1,2,3
python rename_generator.py
python test.py --dataroot $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon --checkpoints_dir $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon/checkpoints --results_dir $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon/results --name simon_100_100 --model test --load_size 128 --crop_size 128 --no_dropout --gpu_ids 0,1,2,3


tar -cvzf $(ws_find cyclegan)/results.tgz $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon/results/
tar -cvzf $(ws_find cyclegan)/ckpts.tgz $TMPDIR/pytorch-CycleGAN-and-pix2pix/simon/checkpoints/


echo "success"