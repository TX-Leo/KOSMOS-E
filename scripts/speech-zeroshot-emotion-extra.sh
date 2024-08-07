set -ex

TASK="emotion"
BRANCH=$1
CKPT=$2
OUTPUT_PATH=$3
BEAM=$4
INPUT_FILE_ID=$5
extra_args=$6

INPUT_FILE_ID=0

mkdir -p $OUTPUT_PATH

cat /mnt/unilm/shaohanh/exp/unigpt_exp/data/speech/emotion/test.s1.out | python eval/speech_zero_shot.py /mnt/unilm/shaohanh/data/tnlg_config/ \
    --task generation \
    --path /mnt/unilm/shaohanh/exp/unigpt_exp/${BRANCH}/${CKPT}.pt \
    --model-overrides "{'visual_pretrained':'/mnt/readincu/t-schen/models/mn_hubert_pretrain_base_iter2_mnpx_rep_fb40un_mbc_mp08_ml10_vggb/checkpoint_last.pt.for_inference','dict_path':'/mnt/unilm/shumma/data/16g/dict.txt'}" \
    --dict-path /mnt/unilm/shumma/data/16g/dict.txt \
    --required-batch-size-multiple 1 --remove-bpe=sentencepiece \
    --max-len-b 900 --add-bos-token \
    --no-repeat-ngram-size 3 \
    --beam ${BEAM} --batch-size 1  --buffer-size 16 ${extra_args} \
    > $OUTPUT_PATH/${TASK}_log_beam${BEAM}_template${INPUT_FILE_ID}.txt

# python evaluation/xsum.py --pred $OUTPUT_PATH/output.txt --gold $REFERENCE --split test