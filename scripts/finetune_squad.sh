#!/usr/bin/env bash

set -ex

TASK=$1
BERT_MODEL_PATH=$2
DATA_DIR=$3
OUTPUT_PATH=$4
ARCH=$5
N_EPOCH=$6
WARMUP_RATIO=$7
BSZ=$8
LR=$9
SEED=${10}
extra_args=${11}

# lrs = ['0.00002', '0.00003', '0.00004', '0.00005']
# seeds = ['1', '2', '3', '4', '5']

BETAS="(0.9,0.98)"
CLIP=0.0
WEIGHT_DECAY=0.01

TASK_DATA_DIR=$DATA_DIR/$TASK

if [ ! -e $BERT_MODEL_PATH ]; then
    echo "Checkpoint doesn't exist"
    exit 0
fi

OPTION=""

if [ "$TASK" = "squad1" ]
then
EPOCH_ITER=5520
fi

if [ "$TASK" = "squad2" ]
then
EPOCH_ITER=8218
OPTION="--version-2-with-negative"
fi

BSZ_EXPAND=$((BSZ/16))
EPOCH_ITER=$((EPOCH_ITER/BSZ_EXPAND))

TOTAL_STEPS=$((EPOCH_ITER*N_EPOCH))
WARMUP_STEPS=$((TOTAL_STEPS/WARMUP_RATIO))
VALIDATE_INTERVAL=$((EPOCH_ITER/2))

echo $EPOCH_ITER
echo $TOTAL_STEPS
echo $WARMUP_STEPS
echo $BSZ

OUTPUT_PATH=$OUTPUT_PATH/$TASK/$N_EPOCH-$WARMUP_RATIO-$BSZ-$LR-$SEED
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
if [ -e $OUTPUT_PATH/train_log.txt ]; then
    if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'Loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
        echo "Training log existed"
        exit 0
    fi
fi

CUDA_VISIBLE_DEVICES=0 python train.py $TASK_DATA_DIR --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --restore-file $BERT_MODEL_PATH \
    --max-sentences $BSZ \
    --update-freq 1 \
    --task squad \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 8 \
    --arch $ARCH \
    --criterion squad $OPTION \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "$BETAS" --adam-eps 1e-06 \
    --clip-norm $CLIP \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_STEPS --warmup-updates $WARMUP_STEPS  \
    --max-update $TOTAL_STEPS --seed $SEED --save-dir /tmp/ --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-save \
    --find-unused-parameters --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric loss --maximize-best-checkpoint-metric \
    --spm-model $TASK_DATA_DIR/sp.model --validate-interval-updates $VALIDATE_INTERVAL ${extra_args} |& tee $OUTPUT_PATH/train_log.txt
