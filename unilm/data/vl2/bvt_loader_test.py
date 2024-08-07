import sys
sys.path.append('.')

from typing import NamedTuple
import os
import argparse
import json

import sentencepiece as spm
from fairseq.data.dictionary import Dictionary
from unilm.data.vl2.bvt_loader import BVTLoader
import tqdm

def to_word(item, dictionary):
    print(dictionary.string(item['net_input']['src_tokens'][0]))
    print(dictionary.string(item['target'][0]))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data', type=str, default='/mnt/unilm/shumma/data/16g')
    parser.add_argument('--spm_path', type=str, default='/mnt/unilm/shumma/data/16g/sentencepiece.bpe.model')
    parser.add_argument('--tokens_per_sample', type=int, default=256)
    parser.add_argument('--sample_break_mode', type=str, default='')
    parser.add_argument('--batch_read_ahead', type=int, default=1)
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--span_length', type=int, default=3)
    parser.add_argument('--dynamic_mask', default=True)
    parser.add_argument('--max_sentences', type=int, default=1) # batch size

    args = parser.parse_args()
    
    Dataset = NamedTuple('Dataset', [('data', str), ('data_dir', str), ('shuffle', bool)])
    dataset = Dataset(json.load(open(f'{args.data}/json/train.json')), args.data, True)
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    dictionary.add_symbol('<mask>')
    for i in range(100):
        dictionary.add_symbol(f"<mask_{i}>")

    tokenizer = spm.SentencePieceProcessor(model_file=args.spm_path)

    mlm_loader = BVTLoader(
            args,
            dataset,
            dictionary,
            tokenizer,
            max_tokens=args.tokens_per_sample,
            max_sentences=args.max_sentences,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
        )

    num = 0
    i = 0
    # for item in mlm_loader:
    #     print(item)
    #     i += 1
    #     if i > num:
    #         break

    import datetime
    for item in tqdm.tqdm(mlm_loader):
        i += 1

def cook_json():
    data = []
    item = {
        "source": [],
        "source_lang": "bvt",
        "weight": 1.0,
        "name": "bvt"
    }
    for i in range(1000):
        item['source'].append("../../../wenwan/bvt/filtered_data/shard0/{:04d}_shard_0_parqChunk_0_filtered.jsonl".format(i))
    data.append(item)
    json.dump(data, open('train.json', 'w', encoding='utf-8'), indent=2)

if __name__ == '__main__':
    run()
    # cook_json()