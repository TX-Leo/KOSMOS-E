# fewshot debug

export batch_size=16
export update_freq=1
export save_dir=/mnt/tmp/kkk

# --pretrained-ckpt-path /mnt/share/data/nlpunilm/yaru/exp/unitask_exp/unitask-medium-lr3e-4-bs256-reset/checkpoint_1_250000.pt \
# --gpt-model-path /mnt/share/data/nlpunilm/yaru/data/gpt_model/en_dense_lm_355m/model.pt \

# --mlm-dict /mnt/share/mydata/roberta_model/roberta.base/dict.txt \

python -m torch.distributed.launch --nproc_per_node=1 --master_port=1241 validate.py "/mnt/share/data/nlpunilm/yaru/data/fewshot/glue-sst2/" \
    --task fewshotclassification \
    --tokens-per-sample 2048  \
    --criterion fewshotclassification \
    --arch unigptmodel_medium  \
    --pretrained-ckpt-path /mnt/share/data/nlpunilm/yaru/exp/unitask_exp/new-unitask-medium-newconnector-continue-lr2e-4-bs256-reset-wd0.01-mlmdrop-warm10k/checkpoint_1_100000-ft/SST-2/5-3-10-32-2e-5-1/checkpoint_best.pt \
    --gpt2-vocab-bpe /mnt/share/mydata/16g/vocab.bpe \
    --gpt2-encoder-json /mnt/share/mydata/16g/encoder.json \
    --batch-size $batch_size  \
    --log-format simple  \
    --log-interval 4 \
    --lr-scheduler polynomial_decay  \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6  \
    --clip-norm 2.0 \
    --warmup-updates 0  \
    --total-num-update 1 \
    --max-update 0 \
    --fp16 \
    --eval-data glue-sst2 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --min-loss-scale 0.0001 \
    --seed 13 \
    --fewshot-type 0 \
    --tensorboard-logdir $save_dir/tb-logs \
    --ddp-backend=no_c10d \
    --distributed-no-spawn \
    --reset-dataloader \
    --test-split "train" \
    --mlm-dict /mnt/share/mydata/roberta_model/roberta.base/dict.txt \
    --gpt-dict /mnt/share/mydata/gpt_model/en_dense_lm_125m/dict.txt 