

python train.py /mnt/share/data/nlpunilm/yaru/data/ft/glue/RTE-bin --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --restore-file /mnt/share/data/nlpunilm/yaru/exp/unitask_exp/new-unitask-medium-newconnector-continue-lr8e-5-bs256-reset-wd0.01-mlmdrop-warm30k/checkpoint_1_110000.pt \
    --max-positions 2048 \
    --max-sentences 8 \
    --max-tokens 2200 \
    --update-freq 1 \
    --task glue \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 8 \
    --init-token 0 --separator-token 2 \
    --arch unigptmodel_medium \
    --criterion glue  \
    --num-classes 2 \
    --eval-data RTE \
    --dropout 0.1 --attention-dropout 0.1  --pooler-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9,0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0001 --total-num-update 10 --warmup-updates 2  \
    --max-update 10 --seed 1 --save-dir /mnt/tmp/kkk --no-progress-bar --log-interval 4 --no-epoch-checkpoints --no-last-checkpoints \
    --find-unused-parameters --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric --validate-interval-updates 500 \
    --mlm-dict /mnt/share/mydata/roberta_model/roberta.base/dict.txt --ft-type 1 \
    --gpt-dict /mnt/share/data/nlpunilm/yaru/data/openwebtext2/dict.txt \
    --mlm-model-path /mnt/share/data/nlpunilm/yaru/data/roberta_model/roberta.base/model.pt
    # 