# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

## run inference without requiring ground-truth answers
## or system info.

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from .mos_fairseq import MosPredictor, MyDataset
import numpy as np
import scipy.stats
import datetime
import time
import urllib
import tarfile
import shutil
def unixnow():
    return str(int(time.mktime(datetime.datetime.now().timetuple())))


def systemID(uttID):
    return uttID.split('-')[0]


def main(args):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model.')
    # parser.add_argument('--datadir', type=str, required=True, help='Path of your directory containing .wav files')
    # parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    # parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    # args = parser.parse_args()

    ## 1. download the base model from fairseq
    if not os.path.exists('mos_finetune_ssl_main/fairseq/wav2vec_small.pt'):
        os.system('mkdir -p mos_finetune_ssl_main/fairseq')
        os.system('wget -e use_proxy=yes -e http_proxy=http://127.0.0.1:7890 https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P mos_finetune_ssl_main/fairseq')
        os.system('wget -e use_proxy=yes -e http_proxy=http://127.0.0.1:7890 https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P mos_finetune_ssl_main/fairseq/')

    ## 2. download the finetuned checkpoint
    if not os.path.exists('mos_finetune_ssl_main/pretrained/ckpt_w2vsmall'):
        # 创建目录（Windows兼容）
        os.makedirs('mos_finetune_ssl_main/pretrained', exist_ok=True)
        
        # 设置代理（如果需要）
        proxy_handler = urllib.request.ProxyHandler({"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
        
        try:
            # 下载文件（Windows原生方式）
            print("Downloading checkpoint...")
            tar_path = 'ckpt_w2vsmall.tar.gz'
            urllib.request.urlretrieve(
                'https://zenodo.org/record/6785056/files/ckpt_w2vsmall.tar.gz',
                tar_path
            )
            
            # 解压文件（Python原生方式）
            print("Extracting archive...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall()
            
            # 移动文件夹（Windows兼容）
            if os.path.exists('ckpt_w2vsmall'):
                shutil.move('ckpt_w2vsmall', 'mos_finetune_ssl_main/pretrained/')
            
            # 删除压缩包（Windows兼容）
            if os.path.exists(tar_path):
                os.remove(tar_path)
            
            # 复制LICENSE文件（Windows兼容）
            src_license = 'mos_finetune_ssl_main/fairseq/LICENSE'
            dst_license = 'mos_finetune_ssl_main/pretrained/LICENSE'
            if os.path.exists(src_license):
                shutil.copy2(src_license, dst_license)
                
            print("Checkpoint downloaded successfully!")
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            # 清理可能残留的文件
            if os.path.exists(tar_path):
                os.remove(tar_path)
            if os.path.exists('ckpt_w2vsmall'):
                shutil.rmtree('ckpt_w2vsmall', ignore_errors=True)
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint
    wavdir = args.datadir
    outfile = args.outfile

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    # print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint, map_location=device))

    if isinstance(wavdir, np.ndarray): # wavdir is np.ndarray
        wavdir = torch.tensor(wavdir, dtype=torch.float32)
        wavlist = 'mydir'
        # wavfnames = [wavdir]
    # elif isinstance(wavdir, list): # wavdir is np.ndarray List
    #     wavfnames = wavdir
    # elif wavdir.split('.')[-1] == 'wav': # wavdir is .wav
    #     wavfnames = [wavdir.split('/')[-1]]
    # else: # wavdir is .wav directory
    #     wavfnames = [x for x in os.listdir(wavdir) if x.split('.')[-1] == 'wav']
    # wavlist = 'tmp_' + unixnow() + '.txt'
    # wavlistf = open(wavlist, 'w')
    # for w in wavfnames:
    #     if isinstance(wavdir, np.ndarray) or isinstance(wavdir, list): # wavdir is np.ndarray # wavdir is np.ndarray List
    #         wavlistf.write(str(w) + ',3.0\n')
    #     else: # wavdir is .wav # wavdir is .wav directory
    #         wavlistf.write(w + ',3.0\n') # type(w) = str
    # wavlistf.close()

    # print('Loading data')
    validset = MyDataset(wavdir, wavlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()
    # print('Starting prediction')

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1

    ## generate answer.txt for codalab
    # ans = open(outfile, 'w')
    # for k, v in predictions.items():
    #     outl = k.split('.')[0] + ',' + str(v) + '\n'
    #     ans.write(outl)
    # ans.close()

    # os.system('rm ' + wavlistf.name)

    # return predictions
    return output

if __name__ == '__main__':
      main()
