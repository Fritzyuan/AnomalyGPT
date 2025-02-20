import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse

# parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# # paths
# parser.add_argument("--few_shot", type=bool, default=True)
# parser.add_argument("--k_shot", type=int, default=1)
# parser.add_argument("--round", type=int, default=3)


# command_args = parser.parse_args()
def parser_args():
    parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
    # paths
    parser.add_argument("--few_shot", type=bool, default=True)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--model", type=str, default='openllama_peft')
    parser.add_argument("--imagebind_ckpt_path", type=str, default='../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')
    parser.add_argument("--vicuna_ckpt_path", type=str, default='../pretrained_ckpt/vicuna_ckpt/7b_v0')
    parser.add_argument("--anomalygpt_ckpt_path", type=str, default='./ckpt/test/train_bsd/pytorch_model.pt')
    parser.add_argument("--delta_ckpt_path", type=str, default='../pretrained_ckpt/pandagpt_ckpt/pandagpt_7b_max_len_1024/pytorch_model.pt')
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()

command_args = parser_args()


describles = {}
describles['deep_pcb'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

FEW_SHOT = command_args.few_shot 

# init the model
# args = {
#     'model': 'openllama_peft',
#     'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
#     'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
#     'anomalygpt_ckpt_path': './ckpt/test/train_bsd/pytorch_model.pt',
#     'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/pandagpt_7b_max_len_1024/pytorch_model.pt',
#     'stage': 2,
#     'max_tgt_len': 128,
#     'lora_r': 32,
#     'lora_alpha': 32,
#     'lora_dropout': 0.1,
# }

model = OpenLLAMAPEFTModel(**vars(command_args))
delta_ckpt = torch.load(command_args.delta_ckpt_path, map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(command_args.anomalygpt_ckpt_path, map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

"""Override Chatbot.postprocess"""
p_auc_list = []
i_auc_list = []

def predict(
    input, 
    image_path, 
    normal_img_path, 
    max_length, 
    top_p, 
    temperature,
    history,
    modality_cache,  
):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return response, pixel_output

input = "Is there any anomaly in the image?"
root_dir = command_args.dataset_path

mask_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])

# CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 'transistor', 'wood', 'zipper']
CLASS_NAMES = ['deep_pcb']

precision = []
log_path = command_args.output_path + "/test_deeppcb_result_" + str(command_args.k_shot) + "_shot.txt"

for c_name in CLASS_NAMES:

    # normal_img_paths = ["../data/AeBAD_mvt_structural/"+c_name+"/train/good/"+str(command_args.round * 4).zfill(3)+".png", "../data/AeBAD_mvt_structural/"+c_name+"/train/good/"+str(command_args.round * 4 + 1).zfill(3)+".png",
    #                     "../data/AeBAD_mvt_structural/"+c_name+"/train/good/"+str(command_args.round * 4 + 2).zfill(3)+".png", "../data/AeBAD_mvt_structural/"+c_name+"/train/good/"+str(command_args.round * 4 + 3).zfill(3)+".png"]

    dir_path = os.path.join(root_dir, c_name, "train", "good")
    all_files = os.listdir(dir_path)
    all_files.sort()  # 如果需要，可以对文件进行排序
    start_index = command_args.round * 4
    normal_img_paths = [os.path.join(dir_path, fname) for fname in all_files[start_index:start_index+4]]
    normal_img_paths = normal_img_paths[:command_args.k_shot]
    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'jpg' in file and c_name in file_path:
                if FEW_SHOT:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, normal_img_paths, 512, 0.1, 1.0, [], [])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, [], 512, 0.1, 1.0, [], [])
                is_normal = 'good' in file_path.split('/')[-2]

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('/test/', '/ground_truth/')
                    mask_path = mask_path.replace('.jpg', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                
                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

                p_label.append(img_mask)
                p_pred.append(anomaly_map)

                i_label.append(1 if not is_normal else 0)
                i_pred.append(anomaly_map.max())

                position = []

                if 'good' not in file_path and 'Yes' in resp:
                    right += 1
                elif 'good' in file_path and 'No' in resp:
                    right += 1
                else:
                    wrong += 1

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100,2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100,2)
    
    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * right / (right + wrong))

    print(c_name, 'right:',right,'wrong:',wrong)
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

    with open(log_path, 'a') as f:
        f.write(f"{c_name} right: {right} wrong: {wrong}\n")
        f.write(f"{c_name} i_AUROC: {i_auroc}\n")
        f.write(f"{c_name} p_AUROC: {p_auroc}\n")

print("i_AUROC:",torch.tensor(i_auc_list).mean())
print("p_AUROC:",torch.tensor(p_auc_list).mean())
print("precision:",torch.tensor(precision).mean())

with open(log_path, 'a') as f:
    f.write(f"i_AUROC: {torch.tensor(i_auc_list).mean()}\n")
    f.write(f"p_AUROC: {torch.tensor(p_auc_list).mean()}\n")
    f.write(f"precision: {torch.tensor(precision).mean()}\n")