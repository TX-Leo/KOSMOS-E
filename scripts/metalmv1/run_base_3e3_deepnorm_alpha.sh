echo 'OMPI_COMM_WORLD_RANK: ' $OMPI_COMM_WORLD_RANK
echo 'MASTER_IP: ' $MASTER_IP
echo 'MASTER_PORT: ' $MASTER_PORT

# Follow https://github.com/donglixp/unitask/blob/lm-freezeele/yamls-unitask/elefrz-gpt-scr-updatelast.yaml to cook 
# Encoder: 
# electra discriminator 24L, 1024; 400M
# GPT: 
# Position: absolute position + relative position?
# Preln deepnorm
# Bsz: 512; Step 1M
# Sequence length 2k
# LR: 2e-4 or 2.5e-4
# Data: Turing
# Tokenization: 64K

# https://github.com/shumingma/DeepNet/commit/02e4279e4a141c3bf051fb66acf3c219d6e2b857

# export OMPI_COMM_WORLD_RANK=1
# export MASTER_IP=10.217.90.63
# export MASTER_PORT=12342

export branch=gptmodel_base-lr3e-3-bs512-reset-wd0.01-mlmdrop-warm500-fxeos-fxfinal-mask-gpt-scratch-deepnorm-aphla-nogithub-noarvix-nopubmed-mtnlg

python -m torch.distributed.launch --nproc_per_node=16 --nnodes=4  \
    --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT train.py /mnt/unilm/shaohanh/data/tnlg_config/ \
    --task gpt_pretraining  \
    --tokens-per-sample 1024 \
    --criterion unigpt  \
    --arch gptmodel_small \
    --required-batch-size-multiple 1 \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6  \
    --clip-norm 2.0 \
    --lr-scheduler polynomial_decay  \
    --weight-decay 0.01  \
    --lr 0.003  \
    --warmup-updates 500  \
    --total-num-update 500000 \
    --max-update 500000 \
    --max-sentences 8 \
    --update-freq 1 \
    --log-format simple  \
    --log-interval 50 \
    --disable-validation \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --min-loss-scale 0.0001 \
    --fp16-no-flatten-grads \
    --seed 1 \
    --dict-path /mnt/unilm/shumma/data/16g/dict.txt  \
    --spm-model /mnt/unilm/shumma/data/16g/sentencepiece.bpe.model \
    --save-dir /mnt/unilm/shaohanh/exp/unitask_exp/${branch}/ \
    --tensorboard-logdir /mnt/unilm/shaohanh/exp/unitask_exp/${branch}/tb-logs \
    --ddp-backend=legacy_ddp \
    --distributed-no-spawn \
    --batch-read-ahead 5000 \
    --reset-dataloader \
    --mlm-tokens-proportion 0 --mlm-cut-length 0 \
    --deep-net-decoder  --last-ln-scale

# --deep-net-decoder \
# --last-ln-scale \

# --all-gather-list-size 102400000 \
# --checkpoint-activations \
# --checkpoint-activations \

