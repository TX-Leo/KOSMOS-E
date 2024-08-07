import json
from tqdm import tqdm, trange

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def eval(input_file, hypo_file, output_file, split_str='summary this image:'):
    # ['▁summary ▁this ▁image :', '▁describe ▁this ▁image :', '▁a ▁image ▁of', '▁a ▁picture ▁of', '▁a ▁photo ▁of']
    # if 'template0' in hypo_file:
    # get idx from hypo_file 
    idx = int(hypo_file.split('.')[0][-1])
    split_str = ['summary this image:', 'describe this image:', 'a image of', 'a picture of', 'a photo of'][idx]
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE"),
    ]

    ptb_tokenizer =  PTBTokenizer()

    # read image

    all_refs = {}
    print("collect ref data")
    obj = json.load(open(input_file, 'r', encoding='utf-8'))
    for i in tqdm(range(len(obj))):    
        if str(i) not in all_refs:
            all_refs[str(i)] = []
        _temp_list = list(obj[i]['caption'])
        for item in _temp_list:
            all_refs[str(i)].append({"caption": item.lower()})

    all_hypos = {}
    # hypo_file = r"C:\Users\shaohanh\Downloads\coco_caption_karpathy_test.arrow.jsonl.0shot"
    with open(hypo_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith('H-'):
                idx = line.split('\t')[0][2:]
                caption = line.split(split_str)[1].strip()
                all_hypos[idx] = [{"caption": caption}]
    
    if len(all_hypos) != len(all_refs):
        print("Warning: hypo_file and input_file have different length.")
        return 
    
    all_refs = ptb_tokenizer.tokenize(all_refs)
    all_hypos = ptb_tokenizer.tokenize(all_hypos)

    final_scores = {}
    for scorer, method in scorers:
        print('computing %s score with COCO-EVAL...' % (scorer.method()))
        score, scores = scorer.compute_score(all_refs, all_hypos)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    writer = open(output_file, 'w', encoding='utf-8')
    print(final_scores)
    writer.write(json.dumps(final_scores, indent=4))
    print("bleu_4", final_scores["Bleu_4"])
    print("meteor", final_scores["METEOR"])
    print("CIDEr", final_scores["CIDEr"])
    # print("SPICE", final_scores["SPICE"])

import sys

if __name__ == "__main__":
    # input_file = sys.argv[1]
    # hypo_file = sys.argv[2]
    input_file = '/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/data/COCO/coco/coco_test.json'
    import os
    # for _file in ['vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix','vl_base_20m_mtnlg_6e4_256-2048_no_deepnorm_fix', 'vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M']:
    for _file in ['vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M_xcon']:
        for f in os.listdir(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}'):
            if f.endswith('vl_zero-shot'):
                for f2 in os.listdir(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}'):
                    print(f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}')
                    if 'coco_log_beam5_template' in f2 and 'eval' not in f2:
                        hypo_file = f'/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/{_file}/{f}/{f2}'
                        import sys
                        # print(hypo_file)
                        # with open(hypo_file + '.eval', 'w') as sys.stdout:
                            # try:
                        output_file = hypo_file + '.eval'
                        eval(input_file, hypo_file, output_file)
                            # except:
                            #     pass
