import json
import os
import multiprocessing
import itertools

from infinibatch import iterators
from functools import partial

try:
    from fairseq.data.encoders.gpt2_bpe import GPT2BPE
except:
    print("GPT2BPE not found, please install fairseq first if you want to use GPT2BPE")

import glob
import os
import torch
import numpy as np
import time
import json
import random
import itertools
import hydra
import copy
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator
from unilm.data.vl2.vl_base_loader import VLBaseLoader

from PIL import Image
import base64
import io

ALT_KEY='MMAltTextWords'
CAPTION_KEY='MMCaptionWords'
CONTENT_KEY='Content'
IMAGE_KEY='MMImage'

class BVTLoader(VLBaseLoader):
    def _build_filter(self):
        def alt_score_filter(item):
            # TODO: add args
            return item['alt_score'] < 0.2
        def cap_score_filter(item):
            # TODO: add args
            return item['cap_score'] < 0.2
        return [alt_score_filter, cap_score_filter]

    def _build_image_transform(self):
        preprocess_image = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return preprocess_image
    
    def _build_text_transform(self):
        def text_transform(text):
            append_eos=True
            fs_dict = self.dictionary
            words = self.tokenizer.encode(text, out_type=str)
            ids = [fs_dict.bos_index]
            for i, word in enumerate(words):
                idx = fs_dict.index(word)
                ids.append(idx)
            if append_eos:
                ids.append(fs_dict.eos_index)
            return ids
        return text_transform

    def _batchify(self, lines):
        
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = iterators.BlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            # -
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)
            
            batches = iterators.BucketedReadaheadBatchIterator(
                    lines,
                    read_ahead=self.batch_read_ahead, 
                    key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
                    batch_size=dynamic_batch_size, 
                    shuffle=self.shuffle,
                    seed=self.seed,
            )

        def collate(batch):
            batch_size = len(batch)

            gpt_max_length = max([len(x[0]) for x in batch])
            image_shape = batch[0][1].shape # (3, 224, 224)

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            image_source_ids = np.full(shape=(batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32,
                                 fill_value=self.dictionary.pad())

            for i, (full_tokens, image_tokens) in enumerate(batch):
                gpt_source_ids[i, :len(full_tokens)-1] = full_tokens[:-1]
                gpt_target_ids[i, :len(full_tokens)-1] = full_tokens[1:]
                image_source_ids[i] = image_tokens
            
            ret_batch = {
                'vl':{
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                        'image_tokens': image_source_ids.astype(np.float32),
                    },
                    'target': gpt_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                }
            }

            return ret_batch

        padded_batches = iterators.MapIterator(
            batches, collate
        )

        return padded_batches

    def _prepare(self, _random, doc):
        """
        """
        
        text_tokens = None
        if doc[ALT_KEY] and doc[CAPTION_KEY]: # random choose one as text tokens
            r = _random.random()
            if r < 0.5:
                text_tokens = doc[ALT_KEY]
            else:
                text_tokens = doc[CAPTION_KEY]
        elif doc[ALT_KEY]:
            text_tokens = doc[ALT_KEY]
        elif doc[CAPTION_KEY]:
            text_tokens = doc[CAPTION_KEY]
        else:
            raise ValueError("No text tokens found in doc")
        image_tokens = doc[IMAGE_KEY]
        return text_tokens, image_tokens

    def _read_from_files(self, source_file):
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        for doc_jsonstr in lines:
            json_obj = json.loads(doc_jsonstr)
            
            # filter item based self.filter
            is_filter = False
            for filter in self.filters:
                if filter(json_obj):
                    is_filter = True
                    break
            if is_filter:
                continue
            
            if json_obj[ALT_KEY]:
                json_obj[ALT_KEY] = self.text_transform(json_obj[ALT_KEY])
            if json_obj[CAPTION_KEY]:
                json_obj[CAPTION_KEY] = self.text_transform(json_obj[CAPTION_KEY])
            pil_img = Image.open(io.BytesIO(base64.b64decode(str(json_obj[CONTENT_KEY])))).convert("RGB")
            torch_tensor = self.image_transform(pil_img)
            json_obj[IMAGE_KEY] = torch_tensor
            # TODO: torch tensor could block the data loader, need to change to other format
            del json_obj[CONTENT_KEY]
            yield json_obj
