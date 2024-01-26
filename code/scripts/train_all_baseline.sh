deepspeed \
    --include localhost:0,1 --master_port 28400 train_bsd.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/pandagpt_7b_max_len_1024/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_bsd/\
    --log_path ./ckpt/train_bsd/log_rest/\
    --dataset_path ../data/mvtec_anomaly_detection/

deepspeed \
    --include localhost:0,1 --master_port 28400 train_deeppcb.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/pandagpt_7b_max_len_1024/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_deeppcb/\
    --log_path ./ckpt/train_deeppcb/log_rest/\
    --dataset_path ../data/mvtec_anomaly_detection/

deepspeed \
    --include localhost:0,1 --master_port 28400 train_pkupcb.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/pandagpt_7b_max_len_1024/pytorch_model.pt\
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_pkupcb/\
    --log_path ./ckpt/train_pkupcb/log_rest/\
    --dataset_path ../data/mvtec_anomaly_detection/
