python eval/text_zero_shot.py /mnt/localdata/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix/checkpoint_1_145000.pt --model-overrides "{'visual_pretrained':'/mnt/localdata/msranlp/zechi/models/clip/vit_b_16-laion400m_e32-55e67d44.pt','dict_path':'/mnt/localdata/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/localdata/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece --no-repeat-ngram-size 3 --max-len-b 60 --add-bos-token

python eval/caption_zero_shot.py /mnt/localdata/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix/checkpoint_1_145000.pt --model-overrides "{'visual_pretrained':'/mnt/localdata/msranlp/zechi/models/clip/vit_b_16-laion400m_e32-55e67d44.pt','dict_path':'/mnt/localdata/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/localdata/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece  --max-len-b 300 --add-bos-token --image-feature-length 197

python eval/caption_zero_shot.py /mnt/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/msranlp/shaohanh/exp/unigpt_exp/vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix/checkpoint_1_240000.pt --model-overrides "{'visual_pretrained':'/mnt/msranlp/zechi/models/clip/vit_b_16-laion400m_e32-55e67d44.pt','dict_path':'/mnt/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece  --max-len-b 300 --add-bos-token --image-feature-length 197 --no-repeat-ngram-size 3 --beam 6

 python eval/speech_zero_shot.py /mnt/localdata/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/only-speech-unigpt-small/checkpoint_1_45000.pt  --model-overrides "{'speech_model_path':'/mnt/localdata/readincu/t-schen/models/mn_hubert_pretrain_base_iter2_mnpx_rep_fb40un_mbc_mp08_ml10_vggb/checkpoint_last.pt.for_inference','dict_path':'/mnt/localdata/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/localdata/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece  --max-len-b 900 --add-bos-token  --no-repeat-ngram-size 3 --beam 1

cat test.txt | python eval/caption_zero_shot.py /mnt/localdata/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix/checkpoint_1_500000.pt --model-overrides "{'visual_pretrained':'/mnt/localdata/msranlp/zechi/models/clip/vit_b_16-laion400m_e32-55e67d44.pt','dict_path':'/mnt/localdata/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/localdata/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece  --max-len-b 300 --add-bos-token --image-feature-length 197 --no-repeat-ngram-size 3 --beam 5 --batch-size 2  --buffer-size 2000

python eval/text_zero_shot.py /mnt/localdata/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/speech-unigpt-medium-lr1e-3-bs512-reset-wd0.01-mlmdrop-warm500-fxeos-fxfinal-mask-gpt-scratch/checkpoint_1_135000.pt --model-overrides "{'speech_model_path':'/mnt/localdata/readincu/t-schen/models/mn_hubert_pretrain_base_iter2_mnpx_rep_fb40un_mbc_mp08_ml10_vggb/checkpoint_last.pt.for_inference','dict_path':'/mnt/localdata/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/localdata/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece --no-repeat-ngram-size 3 --max-len-b 60 --add-bos-token

python eval/speech_zero_shot.py /mnt/localdata/msranlp/shaohanh/data/tnlg_config/ --task generation --path /mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/speech-unigpt-medium-lr1e-3-bs512-reset-wd0.01-mlmdrop-warm500-fxeos-fxfinal-mask-gpt-scratch/checkpoint_1_135000.pt  --model-overrides "{'speech_model_path':'/mnt/localdata/readincu/t-schen/models/mn_hubert_pretrain_base_iter2_mnpx_rep_fb40un_mbc_mp08_ml10_vggb/checkpoint_last.pt.for_inference','dict_path':'/mnt/localdata/msranlp/shumma/data/16g/dict.txt'}" --dict-path /mnt/localdata/msranlp/shumma/data/16g/dict.txt --required-batch-size-multiple 1 --remove-bpe=sentencepiece  --max-len-b 900 --add-bos-token  --no-repeat-ngram-size 3 --beam 5


# coco eval
python eval/vl_coco/eval_coco_all.py &
python eval/vl_coco/get_coco_res.py