import json, os
from tqdm import tqdm, trange

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


if __name__ == "__main__":
    # for _file in ['vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix','vl_base_20m_mtnlg_6e4_256-2048_no_deepnorm_fix', 'vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M']:
    for _file in ['vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M_xcon']:
        for f in os.listdir(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}'):
            if f.endswith('vl_zero-shot'):
                best_blue = 0
                best_template = ''
                for f2 in os.listdir(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}'):
                    if 'eval' in f2:
                        # read json file f2
                        try:                            
                            obj = json.load(open(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}/{f2}', 'r', encoding='utf-8'))
                            # print(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}/{f2}', obj['Bleu_4'], obj['METEOR'], obj['CIDEr'])
                            if obj['Bleu_4'] > best_blue:
                                best_blue = obj['Bleu_4']
                                best_template = f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}/{f2}'
                        except:
                            pass
                print(best_template, best_blue)
