set -ex

TASK="story_cloze"
BRANCH=$1
CKPT=$2
OUTPUT_PATH=$3

mkdir -p $OUTPUT_PATH


cat /mnt/unilm/shaohanh/exp/unigpt_exp/data/gpt/story_cloze/test_data.txt \
 |  python eval/text_zero_shot.py /mnt/unilm/shaohanh/data/tnlg_config/ --task generation \
     --path /mnt/unilm/shaohanh/exp/unigpt_exp/${BRANCH}/${CKPT}.pt \
     --model-overrides "{'visual_pretrained':'/mnt/readincu/t-schen/models/mn_hubert_pretrain_base_iter2_mnpx_rep_fb40un_mbc_mp08_ml10_vggb/checkpoint_last.pt.for_inference','dict_path':'/mnt/unilm/shumma/data/16g/dict.txt'}" \
     --dict-path /mnt/unilm/shumma/data/16g/dict.txt \
     --required-batch-size-multiple 1 --remove-bpe=sentencepiece --no-repeat-ngram-size 3 \
     --max-len-b 256 --add-bos-token --buffer-size 256 --batch-size 16 \
     > $OUTPUT_PATH/${TASK}_log.txt

# python evaluation/xsum.py --pred $OUTPUT_PATH/output.txt --gold $REFERENCE --split test