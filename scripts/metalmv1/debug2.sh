export batch_size=1
export update_freq=1
export save_dir=/mnt/tmp/ppp

python -m torch.distributed.launch --nproc_per_node=1 --master_port=1241 train.py /mnt/share/mydata/16g/ \
    --task gpt_pretraining  \
    --tokens-per-sample 32  \
    --criterion label_smoothed_cross_entropy  \
    --arch gptmodel_small  \
    --required-batch-size-multiple 1 \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6  \
    --clip-norm 2.0 \
    --lr-scheduler polynomial_decay  \
    --gpt2-vocab-bpe /mnt/share/mydata/16g/vocab.bpe \
    --gpt2-encoder-json /mnt/share/mydata/16g/encoder.json \
    --lr 0.0005  \
    --warmup-updates 10000  \
    --total-num-update 125000 \
    --max-update 125000 \
    --max-sentences $batch_size  \
    --update-freq $update_freq \
    --log-format simple  \
    --log-interval 4 \
    --disable-validation \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --min-loss-scale 0.0001 \
    --seed 1 \
    --save-dir $save_dir \
    --tensorboard-logdir $save_dir/tb-logs \
    --ddp-backend=no_c10d \
    --distributed-no-spawn \
    --reset-dataloader \
    --batch-read-ahead 10 \
    --gpt-model-path /mnt/share/mydata/gpt_model/en_dense_lm_125m/model.pt