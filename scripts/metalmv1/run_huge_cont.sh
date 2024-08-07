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

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=32  \
    --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT train.py /mnt/msranlp/shaohanh/data/tnlg_config/ \
    --task gpt_pretraining  \
    --tokens-per-sample 2048 \
    --mlm-tokens-per-sample 512   \
    --criterion unigpt  \
    --arch elegptmodel_xl \
    --required-batch-size-multiple 1 \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6  \
    --clip-norm 2.0 \
    --lr-scheduler polynomial_decay  \
    --weight-decay 0.01  \
    --lr 0.0006  \
    --warmup-updates 375  \
    --total-num-update 500000 \
    --max-update 500000 \
    --max-sentences 2 \
    --update-freq 1 \
    --log-format simple  \
    --log-interval 50 \
    --disable-validation \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --min-loss-scale 0.0001 \
    --seed 3 \
    --dict-path /mnt/msranlp/shumma/data/16g/dict.txt  \
    --spm-model /mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model \
    --ele-model-path /mnt/msranlp/shumma/exp/xd_exp/electra-24L-1k-7e-4-30k-6L-8k/checkpoint_1_150000.pt  \
    --save-dir /mnt/msranlp/shaohanh/exp/unitask_exp/new-unitask-xl-simpleconnector-continue-lr6e-4-bs512-reset-wd0.01-mlmdrop-warm10k-fxeos-freeze-electra-fxfinal-mask-gpt-scratch-update-ele-lasttwolayer-no_aphla-nogithub-noarvix-nopubmed-cont280k/ \
    --tensorboard-logdir /mnt/msranlp/shaohanh/exp/unitask_exp/new-unitask-xl-simpleconnector-continue-lr6e-4-bs512-reset-wd0.01-mlmdrop-warm10k-fxeos-freeze-electra-fxfinal-mask-gpt-scratch-update-ele-lasttwolayer-no_aphla-nogithub-noarvix-nopubmed-cont280k/tb-logs \
    --ddp-backend=no_c10d \
    --distributed-no-spawn \
    --batch-read-ahead 100 \
    --rel-pos-buckets 32 \
    --max-rel-pos 128 \
    --update-last-two-layers \
    --connector-type simple \
    --reset-dataloader \
    --deepnorm-encoder \
    --deep-net-decoder \
    --reset-optimizer --reset-dataloader --reset-meters 

# --last-ln-scale \
# --all-gather-list-size 102400000 \
# --checkpoint-activations \

# --checkpoint-activations \

