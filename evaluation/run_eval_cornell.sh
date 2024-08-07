#!/bin/bash
###
 # @Author: TX-Leo
 # @Mail: tx.leo.wz@gmail.com
 # @Date: 2024-02-11 23:51:39
 # @Version: v1
 # @File: 
 # @Brief: 
### 
nvidia-smi

# If you change the file under fairseq, re-install it!
# pip install fairseq/
# pip install -e fairseq
# pip install  httpcore==0.15.0 gradio==3.9.0

task=generation_obj
quantize_size=32
model_path=/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/08/train_output/01/model_savedir/checkpoint_last.pt

master_port=$((RANDOM%1000+20000))
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3.8 -m torch.distributed.launch --master_port=$master_port --nproc_per_node=1 eval/eval_cornell.py /mnt/msranlp/shaohanh/data/tnlg_config/ \
    --task $task \
    --path $model_path \
    --model-overrides "{'visual_pretrained':'/mnt/msranlp/shaohanh/exp/unigpt_exp/data/models/openai_clip/ViT-L-14-sd.pt', 'dict_path':'/mnt/msranlp/shumma/data/16g/dict.txt'}" \
    --required-batch-size-multiple 1 \
    --remove-bpe=sentencepiece \
    --max-len-b 500 \
    --add-bos-token \
    --beam 1 \
    --buffer-size 1 \
    --image-feature-length 64 \
    --locate-special-token 1 \
    --batch-size 1 \
    --nbest 1 \
    --no-repeat-ngram-size 3 \
    --location-bin-size $quantize_size \
    --dict-path /mnt/msranlp/shumma/data/16g/dict.txt
