set -ex

TASK="coco"
BRANCH=$1
CKPT=$2
OUTPUT_PATH=$3
BEAM=$4
INPUT_FILE_ID=$5

mkdir -p $OUTPUT_PATH

cat /mnt/unilm/shaohanh/exp/unigpt_exp/data/COCO/coco/unigpt/coco_test.json.template_${INPUT_FILE_ID}.out | python eval/caption_zero_shot.py /mnt/unilm/shaohanh/data/tnlg_config/ \
    --task generation \
    --path /mnt/unilm/shaohanh/exp/unigpt_exp/${BRANCH}/${CKPT}.pt \
    --model-overrides "{'visual_pretrained':'/mnt/unilm/zechi/models/clip/vit_b_16-laion400m_e32-55e67d44.pt','dict_path':'/mnt/unilm/shumma/data/16g/dict.txt'}" \
    --dict-path /mnt/unilm/shumma/data/16g/dict.txt \
    --required-batch-size-multiple 1 --remove-bpe=sentencepiece \
    --max-len-b 300 --add-bos-token \
    --image-feature-length 197 --no-repeat-ngram-size 3 \
    --beam ${BEAM} --batch-size 32  --buffer-size 512 \
    > $OUTPUT_PATH/${TASK}_log_beam${BEAM}_template${INPUT_FILE_ID}.txt

python eval/vl_coco/eval_coco.py /mnt/unilm/shaohanh/exp/unigpt_exp/data/COCO/coco/coco_test.json  $OUTPUT_PATH/${TASK}_log_beam${BEAM}_template${INPUT_FILE_ID}.txt > $OUTPUT_PATH/${TASK}_log_beam${BEAM}_template${INPUT_FILE_ID}.txt.eval
# python evaluation/xsum.py --pred $OUTPUT_PATH/output.txt --gold $REFERENCE --split test