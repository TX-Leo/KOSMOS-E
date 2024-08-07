import json
import os
import multiprocessing
import itertools
import ast

from infinibatch import iterators
from functools import partial
import re

try:
    from fairseq.data.encoders.gpt2_bpe import GPT2BPE
except:
    print("GPT2BPE not found, please install fairseq first if you want to use GPT2BPE")
from tiktoken.core import Encoding

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

import torchvision.transforms as T
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator
from unilm.data.vl2.vl_base_loader import VLBaseLoader
from unilm.data.vl2.laion2b_loader import Laion2BLoader, NumpyNormalize
from unilm.data.vl2.obj_utils import *
from unilm.data.vl2.laion2b_obj_loader import Laion2BObjLoader
from unilm.data.vl2.grounding_pron_prompts import (
    pron_templates, brief_caption_templates, text_templates, vqa_templates
)

from PIL import Image
import base64
import io

import logging
logger = logging.getLogger(__name__)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import pdb

ALT_KEY='MMAltTextWords'
CAPTION_KEY='MMCaptionWords'
CONTENT_KEY='Content'
IMAGE_KEY='MMImage'

BOI_SYMBOL="<image>"
EOI_SYMBOL="</image>"

GRD_SYMBOL="<grounding>"

# for objects
OBJ_KEY='Objects'
BOP_SYMBOL="<phrase>"
EOP_SYMBOL="</phrase>"
BOO_SYMBOL="<object>"
EOO_SYMBOL="</object>"
DOM_SYMBOL="</delimiter_of_multi_objects/>"

class Laion2BObjTuneLoader(Laion2BObjLoader):
    def _setup(self):
        self.max_image_num = self.args.max_image_num
        self.image_token_length = self.args.image_token_length
        self.input_resolution = self.args.input_resolution
        self.quantized_size = self.args.quantized_size
        self.quantized_num = self.quantized_size ** 2
        self.box_score_threshold = self.args.box_score_threshold
        self.mix_no_object_prob = self.args.mix_no_object_prob
        self.use_object_bbox_prob = self.args.use_object_bbox_prob
        self.use_locate_special_token = bool(self.args.locate_special_token)
        
        self.phrase_mode = self.args.phrase_mode
        assert self.phrase_mode in ['phrase', 'expression']
        
        self.laion_tune_min_box = self.args.laion_tune_min_box
        self.laion_tune_image_only_resize = bool(self.args.laion_tune_image_only_resize)
        self.laion_tune_use_single_box_prob = self.args.laion_tune_use_single_box_prob
        self.laion_tune_use_single_box_mode = self.args.laion_tune_use_single_box_mode
        self.laion_tune_use_caption_template = bool(self.args.laion_tune_use_caption_template)
        
        # statistic the number of vocab
        tokenizer_vocabs = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        self.tokenizer_vocab_num = len(tokenizer_vocabs)
        
        logger.info(f"Enabling {self.phrase_mode}-mode for phrase name")
        logger.info(f"Mixing prob {self.mix_no_object_prob} for using image-text pair without no grounding label")
        logger.info(f"Mixing prob {self.use_object_bbox_prob} for using image-text pair with grounding label")
        logger.info(f"Vocab length in tokenizer: {self.tokenizer_vocab_num}")
        logger.info(f"Vocab length in dictionary: {len(self.dictionary.symbols)}")
        logger.info(f"Only use resize transform during tuning: {self.laion_tune_image_only_resize}")
        logger.info(f"Min obj number during tuning: {self.laion_tune_min_box}")
        logger.info(f"Prob of use img-txt that below laion_tune_min_box during tuning: {self.laion_tune_use_single_box_prob}")
        logger.info(f"Gradient mode of use img-txt that below laion_tune_min_box during tuning: {self.laion_tune_use_single_box_mode}")
        logger.info(f"Use brief caption template during laion tuning: {self.laion_tune_use_caption_template}")
        
    def _build_filter(self):
        def width_height_filter(item):
            # judge item[3] and item[4] is interger
            if item[3].isdigit() and item[4].isdigit():
                return int(item[3]) < 200 or int(item[4]) < 200
            return True
        return [width_height_filter]
    
    def _build_image_transform(self):
        preprocess_image = Compose([
            Resize(self.input_resolution, interpolation=BICUBIC),
            CenterCrop(self.input_resolution),
            NumpyNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        if getattr(self, 'laion_tune_image_only_resize', False):
            return self._build_image_resize_transform()
        return preprocess_image
    
    def _build_image_resize_transform(self):
        # only perform resize transform 
        preprocess_image = Compose([
            Resize((self.input_resolution, self.input_resolution), interpolation=BICUBIC),
            NumpyNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return preprocess_image
    
    def _build_text_transform(self):
        def text_transform(text):
            append_eos=True
            fs_dict = self.dictionary
            if isinstance(self.tokenizer, Encoding):
                words = list(map(str, self.tokenizer.encode(text, allowed_special="all")))
            else:
                words = self.tokenizer.encode(text, out_type=str)
            
            # ids = [fs_dict.bos_index]
            ids = []
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
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)
            image_source_ids = np.full(shape=(batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32,
                                 fill_value=self.dictionary.pad())
            chunk_tokens_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            segment_tokens_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            for i, (full_tokens, image_tokens, text_input_mask, text_loss_mask, chunk_tokens, segment_tokens) in enumerate(batch):
                gpt_source_ids[i, :len(full_tokens)-1] = full_tokens[:-1]
                gpt_target_ids[i, :len(full_tokens)-1] = full_tokens[1:]
                gpt_input_mask_all[i, :len(full_tokens)-1] = text_input_mask[:-1]
                gpt_loss_mask_all[i, :len(full_tokens)-1] = text_loss_mask[:-1]
                chunk_tokens_all[i, :len(full_tokens)-1] = chunk_tokens[:-1]
                segment_tokens_all[i, :len(full_tokens)-1] = segment_tokens[:-1]
                image_source_ids[i] = image_tokens
            
            ret_batch = {
                'vl_laion':{
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                        'img_src_tokens': image_source_ids.astype(np.float32),
                        'img_gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
                        'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
                        'chunk_tokens': chunk_tokens_all.astype(np.int64),
                        'segment_tokens': segment_tokens_all.astype(np.int64),
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
        boi_id = self.dictionary.index(BOI_SYMBOL) 
        eoi_id = self.dictionary.index(EOI_SYMBOL)
        bos_id = self.dictionary.bos_index
        text_tokens = doc[CAPTION_KEY]
        image_tokens = doc[IMAGE_KEY]
        text_length = len(text_tokens)
        text_tokens = [bos_id] + [boi_id] * (self.image_token_length + 1) + [eoi_id] + text_tokens
        text_input_mask = [0]  + [0]  + [1] * (self.image_token_length) + [0] + [0] * text_length
        if 'text_loss_mask' in doc:
            text_loss_mask =  [0]  + [0]  + [0] * (self.image_token_length) + [1] + doc['text_loss_mask']
            # print(f"{text_length}-{len(doc['text_loss_mask'])}: {text_loss_mask}")
        else:
            text_loss_mask =  [0]  + [0]  + [0] * (self.image_token_length) + [1] + [1] * text_length
        chunk_tokens = [0]  + [1]  + [1] * (self.image_token_length) + [1] + [1] * text_length 
        segment_tokens = [0]  + [1]  + [1] * (self.image_token_length) + [1] + [0] * text_length
        return text_tokens, image_tokens, text_input_mask, text_loss_mask, chunk_tokens, segment_tokens

    def _read_from_files(self, source_file):
        file_path = os.path.join(self.data_dir, source_file)
        
        if 'laion2b_filtered_tsvs_v1' in file_path: 
            file_path = file_path.replace('laion2b_filtered_tsvs_v1', 'laion2b_filtered_tsvs_v1_obj_expression')
        elif 'coyo_filtered_tsvs_v1' in file_path: 
             file_path = file_path.replace('coyo_filtered_tsvs_v1', 'coyo_filtered_tsvs_v1_obj')
        else:
            print("Unsupport file: ", file_path)
            return iter([]) # skip bad file
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file
        print(file_path)
        
        for doc_str in lines:
            json_obj = {}
            item = doc_str.strip().split('\t')
                
            # filter item based self.filter
            if 'laion2b' in source_file: # filter out bad image on laion dataset
                try:
                    is_filter = False
                    for filter in self.filters:
                        if filter(item):
                            is_filter = True
                            break
                    if is_filter:
                        continue
                except Exception as e:
                    logger.warning(f" {e}")
                    continue
            
            try:
                caption = item[1]
                
                # read image and transform it
                pil_img = Image.open(io.BytesIO(base64.b64decode(item[2]))).convert("RGB")
                ori_img_w, ori_img_h = pil_img.size
                torch_tensor = self.image_transform(pil_img)
                json_obj[IMAGE_KEY] = torch_tensor
                
                # mix_no_object_prob is the hyper to control whether using data without bbox labels
                if len(item) < 6 and random.random() < self.mix_no_object_prob:
                    # images without bboxes
                    # print(f"No object from {file_path} with mixing prob {self.mix_no_object_prob}")
                    json_obj[CAPTION_KEY] = self.text_transform(item[1])
                else:
                    filter_this_image = False
                    single_obj = False
                    # mode: 'expression', 'phrase'
                    cluster_obj_dict = process_grounding_data(self, item, 
                                                              (ori_img_w, ori_img_h), 
                                                              mode='expression', 
                                                              mode_switch_prob=0.5,
                                                              drop_crop_thr=0.2,
                                                              perform_centercrop=(not self.laion_tune_image_only_resize))

                    filter_cluster_obj_dict = {}
                    box_cnt = 0
                    for start_end in cluster_obj_dict.keys():
                        expression = caption[start_end[0]: start_end[1]]
                        # contain number
                        number = find_first_number(expression)
                        if number is not None:
                            # pdb.set_trace()
                            if number == len(cluster_obj_dict[start_end][0]):
                                filter_cluster_obj_dict[start_end] = cluster_obj_dict[start_end]
                                box_cnt += number
                                continue
                            else:
                                # we discard this image because its unmatch
                                filter_this_image = True
                                break
                            
                        if starts_with_demonstrative(expression):
                            if len(cluster_obj_dict[start_end][0]) == 1:
                                # we discard this image because its unmatch
                                filter_this_image = True
                                break
                            
                        # filter the box overlap the whole image
                        _grd = [[], [], []]
                        for grd_index in range(len(cluster_obj_dict[start_end][-1])):
                            box = cluster_obj_dict[start_end][-1][grd_index]
                            if (box[2] - box[0]) * (box[3] - box[1]) > 0.9 * 0.9:
                                # sometimes, the box covering the whole image is also right
                                single_obj = True
                                continue
                            _grd[0].append(cluster_obj_dict[start_end][0][grd_index])
                            _grd[1].append(cluster_obj_dict[start_end][1][grd_index])
                            _grd[2].append(cluster_obj_dict[start_end][2][grd_index])
                            
                        if len(_grd[0]) > 0:
                            filter_cluster_obj_dict[start_end] = _grd
                            box_cnt += len(_grd[0])
                        
                    if len(filter_cluster_obj_dict.keys()) == 0 or filter_this_image:
                        continue
                    
                    if box_cnt < self.laion_tune_min_box:
                        if random.random() > self.laion_tune_use_single_box_prob:
                            continue
                        single_obj = True
                    # box_cnt < self.laion_tune_min_box
                    # visualize the transformed image and box to check
                    # visualize_normed_img_with_bbox(torch_tensor, filter_cluster_obj_dict, self.quantized_size, caption, item[0])
                    
                    if random.random() < self.use_object_bbox_prob:
                        if single_obj and self.laion_tune_use_single_box_mode == 'box':
                            # pdb.set_trace()
                            tokenized_id_list = self.text_transform(item[1])
                            # random choose one expression, just one
                            random_key = random.sample(filter_cluster_obj_dict.keys(), 1)[0]
                            new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, {random_key: filter_cluster_obj_dict[random_key]})
                            bop_start_index = new_tokenized_id_list.index(self.dictionary.index(BOP_SYMBOL))
                            eoo_end_index = new_tokenized_id_list.index(self.dictionary.index(EOO_SYMBOL))
                            new_tokenized_id_list = new_tokenized_id_list[bop_start_index:eoo_end_index+1] + [self.dictionary.eos_index]
                            if self.use_locate_special_token:
                                new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                                
                            boo_start_index = new_tokenized_id_list.index(self.dictionary.index(BOO_SYMBOL))
                            text_loss_mask = [0] * len(new_tokenized_id_list[:boo_start_index]) + [1] * len(new_tokenized_id_list[boo_start_index:])
                            text_loss_mask[-1] = 0
                            text_loss_mask[-2] = 0
                            json_obj[CAPTION_KEY] = new_tokenized_id_list
                            json_obj['text_loss_mask'] = text_loss_mask
                            # visualize_normed_img_with_bbox(torch_tensor, {random_key: filter_cluster_obj_dict[random_key]}, self.quantized_size, caption, item[0])
                        else:
                            # we use the sample for caption here, like the pretraining paradigm
                            tokenized_id_list = self.text_transform(item[1])
                            new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, filter_cluster_obj_dict)
                            if self.use_locate_special_token:
                                new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                                # print(f"tokenized code with location start code {self.dictionary.index(GRD_SYMBOL)}: {new_tokenized_id_list}")
                            json_obj[CAPTION_KEY] = new_tokenized_id_list
                            
                            if self.laion_tune_use_caption_template:
                                # pdb.set_trace()
                                caption_template = random.sample(brief_caption_templates, 1)[0]
                                caption_template_id_list = self.text_transform(caption_template)[:-1] # exclue the <eos>
                                if self.use_locate_special_token:
                                    text_loss_mask = [0] + [0] * len(caption_template_id_list) + [1] * (len(new_tokenized_id_list)-1)
                                    new_tokenized_id_list = [new_tokenized_id_list[0]] + caption_template_id_list + new_tokenized_id_list[1:]
                                else:
                                    text_loss_mask =[0] * len(caption_template_id_list) + [1] * len(new_tokenized_id_list)
                                    new_tokenized_id_list = caption_template_id_list + new_tokenized_id_list
                                json_obj[CAPTION_KEY] = new_tokenized_id_list
                                json_obj['text_loss_mask'] = text_loss_mask
                            # if single_obj:
                            #     visualize_normed_img_with_bbox(torch_tensor, filter_cluster_obj_dict, self.quantized_size, caption, item[0])
                    else:
                        # filter all objects
                        # print(f"Filter all objects from {file_path}")
                        json_obj[CAPTION_KEY] = self.text_transform(item[1])
                        
                yield json_obj
            # except KeyboardInterrupt as e:
            #     raise ValueError
            except Exception as e:
                # raise
            #     # logger.warning(f"{e}, skip this image-text pair data")
                continue

def find_first_number(string):  
    arabic_numbers = "123456789"  
    english_numbers = {  
        # "zero": 0,  
        "one": 1,  
        "two": 2,  
        "three": 3,  
        "four": 4,  
        "five": 5,  
        "six": 6,  
        "seven": 7,  
        "eight": 8,  
        "nine": 9,
        "ten": 10,  
    }  
  
    for char in string:  
        if char in arabic_numbers:  
            return int(char)  
  
    lower_case_string = string.lower()  
    words = re.findall(r'\w+', lower_case_string)  
  
    for word in words:  
        if word in english_numbers:  
            return english_numbers[word]  
  
    return None  


def starts_with_demonstrative(string):  
  
    demonstratives = ["these", "those", 'all', 'their', 
                      'our', 'several', 'some',
                      'many', 'multiple', 'hundreds', 'thoundreds', 
                      'little', 'multi', 'any', '©']  
    lower_case_string = string.lower()  
      
    first_word = re.findall(r'\w+', lower_case_string)[0]  
  
    return first_word in demonstratives