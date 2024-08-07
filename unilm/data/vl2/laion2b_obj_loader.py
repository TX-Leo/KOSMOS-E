import json
import os
import multiprocessing
import itertools
import ast
import re

from infinibatch import iterators
from functools import partial

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

# ===BEGIN===
CAPTION_KEY_NUMBERS = 'NUMBERS'
NUMBERS_LOSS_ONLY = False
a_to_a_encoded = True
DEBUG = True
GEN_PC = False
# ===END===

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
class Laion2BObjLoader(Laion2BLoader):
    def _setup(self):
        self.number = 0
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
        
        if getattr(self.args, 'training_image_only_resize', False):
            self.training_image_only_resize = bool(self.args.training_image_only_resize)
        else:
            self.training_image_only_resize = False
            
        # **** add special tokens, have done it at unilm/data/utils.py ****
        # self.dictionary.add_symbol(BOI_SYMBOL)
        # self.dictionary.add_symbol(EOI_SYMBOL)
        # self.dictionary.add_symbol(BOO_SYMBOL)
        # self.dictionary.add_symbol(EOO_SYMBOL)
        # self.dictionary.add_symbol(BOP_SYMBOL)
        # self.dictionary.add_symbol(EOP_SYMBOL)
        # self.dictionary.add_symbol(DOM_SYMBOL)
        # # add special location tokens
        # for i in range(self.quantized_num):
        #     token_name = f"<patch_index_{str(i).zfill(4)}>"
        #     self.dictionary.add_symbol(token_name)
        
        # statistic the number of vocab
        tokenizer_vocabs = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        # dictionary_vocabs = self.dictionary.symbols
        self.tokenizer_vocab_num = len(tokenizer_vocabs)
        
        logger.info(f"Enabling {self.phrase_mode}-mode for phrase name")
        logger.info(f"Mixing prob {self.mix_no_object_prob} for using image-text pair without no grounding label")
        logger.info(f"Mixing prob {self.use_object_bbox_prob} for using image-text pair with grounding label")
        logger.info(f"Vocab length in tokenizer: {self.tokenizer_vocab_num}")
        logger.info(f"Vocab length in dictionary: {len(self.dictionary.symbols)}")
        logger.info(f"Only use resize transform during pretraining: {self.training_image_only_resize}")
        
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
        if getattr(self, 'training_image_only_resize', False):
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
        
        # ===BEGIN===
        if NUMBERS_LOSS_ONLY:
            numbers_loss_mask = doc[CAPTION_KEY_NUMBERS]
            text_loss_mask =  [0]  + [0]  + [0] * (self.image_token_length) + [0] + numbers_loss_mask
            if DEBUG:
                # print(f'numbers_loss_mask:{numbers_loss_mask}')
                print(f'text_loss_mask:{text_loss_mask}')
        else:
            text_loss_mask =  [0]  + [0]  + [0] * (self.image_token_length) + [1] + [1] * text_length
        # ===END===
        
        chunk_tokens = [0]  + [1]  + [1] * (self.image_token_length) + [1] + [1] * text_length 
        segment_tokens = [0]  + [1]  + [1] * (self.image_token_length) + [1] + [0] * text_length
        return text_tokens, image_tokens, text_input_mask, text_loss_mask, chunk_tokens, segment_tokens

    def _find_number_locations(self,s):
        # s = "Grasp center point coordinates: x_95,y_56 Grasp rotation angle in radians: a_-0.327 "  --> "x_95,y_56" + "a_-0.327"
        import re
        pattern = r'\b[a-zA-Z]+\s*_[+-]?\d+(?:\.\d+)?(?:,[a-zA-Z]+\s*_[+-]?\d+(?:\.\d+)?)*\b'
        matches = re.findall(pattern, s)
        number_locations = []
        for match in matches:
            start = s.index(match)
            end = start + len(match)
            number_locations.append([start, end])
        return number_locations
    
    def encode_value(self, value, bins=256, lower_limit=-np.pi/2, upper_limit=np.pi/2):
        range_width = upper_limit - lower_limit
        normalized_value = value - lower_limit
        scaled_value = normalized_value / range_width
        encoded_value = max(0, min(round(scaled_value * bins), 255))
        return encoded_value
    
    def _read_from_files(self, source_file):
        if DEBUG:
            print()
        file_path = os.path.join(self.data_dir, source_file)
        
        # ============= Note:Comment out this part of code ===============
        # if 'laion2b_filtered_tsvs_v1' in file_path: 
        #     file_path = file_path.replace('laion2b_filtered_tsvs_v1', 'laion2b_filtered_tsvs_v1_obj_expression')
        # elif 'coyo_filtered_tsvs_v1' in file_path: 
        #      file_path = file_path.replace('coyo_filtered_tsvs_v1', 'coyo_filtered_tsvs_v1_obj')
        # else:
        #     print("Unsupport file: ", file_path)
        #     return iter([]) # skip bad file
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file
         # print(file_path)
        
        for doc_str in lines:
            json_obj = {}
            item = doc_str.strip().split('\t')
            # delete the blank at the string end
            item[1] = item[1].rstrip()
            if GEN_PC:
                # delete the pointcloud item
                item = item[:-1]
             # print(f'len:{len(item)}')
             
            if a_to_a_encoded:
                data = item[1]
                if DEBUG:
                    print(f'data:{data}')
                a = float(re.findall(r'a_(-?[\d.]+)', data)[0])
                if DEBUG:
                    print(f'a:{a}')
                a_encoded = self.encode_value(a)
                if DEBUG:
                    print(f'a_encoded:{a_encoded}')
                new_data = data.replace(str(a), str(a_encoded))
                item[1] = new_data
                if DEBUG:
                    print(f'item[1]:{item[1]}')

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
                    # if DEBUG:
                    #     print('model-1')
                else:
                    # mode: 'expression', 'phrase'
                    cluster_obj_dict = process_grounding_data(self, item, 
                                                                (ori_img_w, ori_img_h), 
                                                                mode=self.phrase_mode, 
                                                                mode_switch_prob=0.5,
                                                                drop_crop_thr=0.2,
                                                                perform_centercrop=(not self.training_image_only_resize))
                    # print(cluster_obj_dict)
                    # visualize the transformed image and box to check
                    # print(file_path)
                    # print(caption)
                    # print(obj_lists)
                    # pdb.set_trace()
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0])
                    
                    if len(cluster_obj_dict) > 0 and random.random() < self.use_object_bbox_prob:
                        tokenized_id_list = self.text_transform(item[1])
                        new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict)
                        if self.use_locate_special_token:
                            new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                            # print(f"tokenized code with location start code {self.dictionary.index(GRD_SYMBOL)}: {new_tokenized_id_list}")
                        json_obj[CAPTION_KEY] = new_tokenized_id_list
                        
                        # print(new_tokenized_id_list)
                        # print(f"Some objects from {file_path}")
                        # print some information
                        # pdb.set_trace()
                        # print(obj_lists)
                        # rich_caption = insert_bounding_boxes(caption, dict(sorted(cluster_obj_dict.items(), key=lambda x: x[0][0])))
                        # print('\n')
                        # print(caption)
                        # print(rich_caption)
                        # print(self._decode_id_to_piece(new_tokenized_id_list))
                        # print(self._decode_id_to_str(new_tokenized_id_list))
                        # print(self._decode_id_to_str_with_location(new_tokenized_id_list))
                        # if DEBUG:
                        #     print('model-2')
                        
                    else:
                        # filter all objects
                        # print(f"Filter all objects from {file_path}")
                        json_obj[CAPTION_KEY] = self.text_transform(item[1])
                        # if DEBUG:
                        #     print('model-3')

                # ===BEGIN===           
                if NUMBERS_LOSS_ONLY:
                    if DEBUG:
                        print(f'====== item[1]:\n{item[1]}')
                        print(f'number:{self.number}')
                        print(f'json_obj[CAPTION_KEY]:{json_obj[CAPTION_KEY]}')
                        
                    s = str(item[1])
                    number_locations = self._find_number_locations(s)
                    texts = []
                    texts_tokens = []
                    numbers_loss_mask_locations = []
                    for number_location in number_locations:
                        # get the text
                        text_without_number = s[:number_location[0]]
                        text_with_number = s[:number_location[1]]
                        # get the text tokens
                        text_tokens_without_number = self.text_transform(text_without_number)
                        text_tokens_with_number = self.text_transform(text_with_number)
                        # get the number loss mask index
                        number_loss_mask_start = len(text_tokens_without_number)-1
                        number_loss_mask_end = len(text_tokens_with_number)-1
                        # update texts
                        texts.append(text_without_number)
                        texts.append(text_with_number)
                        # update texts_tokens
                        texts_tokens.append(text_tokens_without_number)
                        texts_tokens.append(text_tokens_with_number)
                        # update numbers_loss_mask_locations
                        numbers_loss_mask_locations.append([number_loss_mask_start,number_loss_mask_end])
                    # update numbers_loss_mask
                    numbers_loss_mask = [0] * len(json_obj[CAPTION_KEY])
                    for start, end in numbers_loss_mask_locations:
                        numbers_loss_mask[start:end] = [1] * (end - start)
                    json_obj[CAPTION_KEY_NUMBERS] = numbers_loss_mask
                    
                    if DEBUG:
                        [print(number_location) for number_location in number_locations]
                        [print(text) for text in texts]
                        [print(text_tokens) for text_tokens in texts_tokens]    
                        [print(numbers_loss_mask_location) for numbers_loss_mask_location in numbers_loss_mask_locations]
                        print(f'numbers_loss_mask:{numbers_loss_mask}')
                    
                    self.number += 1
                    # ===END===
                    
                yield json_obj
            except KeyboardInterrupt as e:
                raise ValueError
            except Exception as e:
                # raise
            #     # logger.warning(f"{e}, skip this image-text pair data")
                continue
            
    def _embed_box_after_phrase(self, caption, tokenized_id_list, cluster_obj_dict, has_eos=True):
        # get the word based on the tokenized id
        tokenized_list = self._decode_id_to_piece(tokenized_id_list) # exclude the eos token
        if has_eos:
            tokenized_list = tokenized_list[:-1]
        # phrase_positions = find_substring_pairs(caption, tokenized_list, cluster_obj_dict.keys())
        phrase_positions = find_substring_pairs(caption, cluster_obj_dict.keys(), self.tokenizer)
        
        for i, k in enumerate(cluster_obj_dict.keys()):
            ids = phrase_positions[i]
            cluster_obj_dict[k].append([ids[0], ids[-1]+1])
        # print("clustering the object", cluster_obj_dict)

        sorted_cluster_obj_dict = dict(sorted(cluster_obj_dict.items(), key=lambda x: x[0][0]))
        # print("clustering the object and sort", sorted_cluster_obj_dict)

        new_sorted_cluster_obj_dict = {}
        for k, v in sorted_cluster_obj_dict.items():
            new_k = tuple(v[-1])
            new_v = [v[0], v[1], list(k)]
            new_sorted_cluster_obj_dict[new_k] = new_v
            
        sorted_cluster_obj_dict = new_sorted_cluster_obj_dict
        # print("clustering the object and sort", sorted_cluster_obj_dict)

        # inset the box index into tokeinzed code
        new_tokenized_id_list = []

        phrase_index = 0
        phrase_start = list(sorted_cluster_obj_dict.keys())[phrase_index][0]
        phrase_end = list(sorted_cluster_obj_dict.keys())[phrase_index][1]

        for i, token_id in enumerate(tokenized_list):
            # print(f"{i} -> {tokenized_id_list[i]} -> {tokenized_list[i]}")
            if i == phrase_start:
                new_tokenized_id_list.append(self.dictionary.index(BOP_SYMBOL))

            new_tokenized_id_list.append(tokenized_id_list[i])
            
            if i + 1 == phrase_end:
                new_tokenized_id_list.append(self.dictionary.index(EOP_SYMBOL))
                new_tokenized_id_list.append(self.dictionary.index(BOO_SYMBOL))
                
                # add patch index tokens
                obj_lists = list(sorted_cluster_obj_dict.values())[phrase_index]
                for k, (ul_index, lr_index) in enumerate(obj_lists[0]):
                    token_name = f"<patch_index_{str(ul_index).zfill(4)}>"
                    new_tokenized_id_list.append(self.dictionary.index(token_name))
                    
                    token_name = f"<patch_index_{str(lr_index).zfill(4)}>"
                    new_tokenized_id_list.append(self.dictionary.index(token_name))
                    
                    if k < len(obj_lists[0]) - 1:
                        new_tokenized_id_list.append(self.dictionary.index(DOM_SYMBOL))
                    
                new_tokenized_id_list.append(self.dictionary.index(EOO_SYMBOL))
                
                # update phrase flag
                if phrase_index < len(sorted_cluster_obj_dict) - 1:
                    phrase_index += 1
                    phrase_start = list(sorted_cluster_obj_dict.keys())[phrase_index][0]
                    phrase_end = list(sorted_cluster_obj_dict.keys())[phrase_index][1]
                    
        # add eos
        if has_eos:
            new_tokenized_id_list.append(tokenized_id_list[-1])
        # print("New new_tokenized_id_list: ", new_tokenized_id_list)

        return new_tokenized_id_list
        
    def _decode_id_to_piece(self, tokenized_id_list):
        return [self.dictionary[idx] for idx in tokenized_id_list]
    
    def _decode_id_to_str(self, tokenized_id_list):
        decode_tokenized_id_list = [idx for idx in tokenized_id_list if idx < self.tokenizer_vocab_num]
        decode_tokenized_list = [self.dictionary[idx] for idx in decode_tokenized_id_list]
        decode_str = self.tokenizer.decode(decode_tokenized_list)
        
        return decode_str

    def _decode_id_to_str_with_location(self, tokenized_id_list):
        
        decode_results = []
        to_decode_piece = []
        i, j = 0, 0
        
        # pdb.set_trace()
        while True:
            if i >= len(tokenized_id_list):
                if len(to_decode_piece) > 0:
                    decode_results.append(self.tokenizer.decode(to_decode_piece)) 
                break
            
            current_id = tokenized_id_list[i]
            current_word = self.dictionary[tokenized_id_list[i]]
            
            # print(decode_results)
            # print(to_decode_piece)
            # print(f"id {current_id} -> {current_word}")
            
            # meet one special token
            if current_id >= self.tokenizer_vocab_num:
                # decode previous words
                if len(to_decode_piece) > 0:
                    decode_results.append(self.tokenizer.decode(to_decode_piece))
                    to_decode_piece = []
                
                # decode current special token 
                if 'patch_index_' in current_word:
                    # current token for bounding box coordinates
                    patch_index = int(current_word[1:-1].split('_')[-1])
                    cell_size = 1.0 / self.quantized_size
                    x = (patch_index % self.quantized_size) * cell_size + cell_size / 2
                    y = (patch_index // self.quantized_size) * cell_size + cell_size / 2
                    decode_results.append(f"({x}, {y})")
                    i += 1
                else:
                    # not for coordinates, we just directly decode now (<phrase>, </phrase>, ...)
                    decode_results.append(current_word)
                    i += 1
            
            # normal token, store it for decode
            else:
                i += 1
                to_decode_piece.append(current_word)
        
        return " ".join(decode_results)
    


def find_substring_pairs(input_str, pos_list, tokenizer):
    substring_positions = []
    for (pos_start, pos_end) in pos_list:
        before_pos_string = input_str[:pos_start]
        before_pos_string_tokenized_list = tokenizer.encode(before_pos_string, out_type=str)
        # print(f"before_pos_string_tokenized_list: ", before_pos_string_tokenized_list)
        
        after_pos_string = input_str[:pos_end]
        after_pos_string_tokenized_list = tokenizer.encode(after_pos_string, out_type=str)
        # print(f"after_pos_string_tokenized_list: ", after_pos_string_tokenized_list)
        
        before_length = len(before_pos_string_tokenized_list)
        # assert before_pos_string_tokenized_list == after_pos_string_tokenized_list[:before_length], \
        #     f"{before_pos_string_tokenized_list} is not contained in {after_pos_string_tokenized_list} when the pos_list is [{pos_start}, {pos_end}]"
        if before_pos_string_tokenized_list == after_pos_string_tokenized_list[:before_length]:
            substring_positions.append([before_length, len(after_pos_string_tokenized_list)-1])
        elif before_pos_string_tokenized_list[:-1] == after_pos_string_tokenized_list[:before_length-1]:
            substring_positions.append([before_length-1, len(after_pos_string_tokenized_list)-1])
        else:
            raise AssertionError(f"{before_pos_string_tokenized_list} is not contained in {after_pos_string_tokenized_list} when the pos_list is [{pos_start}, {pos_end}]")
        
    return substring_positions

def insert_bounding_boxes(text, cluster_obj_dict):
    phrase_start_end_list = list(cluster_obj_dict.keys())
    phrase_bbox_score_list = list(cluster_obj_dict.values())
    result = []
    i = 0
    j = 0
    while True:
        if i >= len(text):
            break
        
        if j == len(phrase_start_end_list):
            result.append(f"{text[i:]}")
            break
        
        if i == phrase_start_end_list[j][0]:
            phrase_name = text[phrase_start_end_list[j][0]:phrase_start_end_list[j][1]]
            box = phrase_bbox_score_list[j][0]
            result.append(f"[{phrase_name}]({str(box)})")
            
            i = phrase_start_end_list[j][1]
            j += 1
        else:
           result.append(text[i])
           i += 1
    return ''.join(result)

def test_laion2b_obj_loader():
    dataloader = Laion2BObjLoader()
    s = "Grasp center point coordinates: x_109,y_111 "
    number_locations = dataloader._find_number_locations(s)
    [print(number_location) for number_location in number_locations]
    texts = []
    texts_tokens = []
    numbers_loss_mask_locations = []
    for number_location in number_locations:
        # get the text
        text_without_number = s[:number_location[0]]
        text_with_number = s[:number_location[1]]
        # get the text tokens
        text_tokens_without_number = dataloader.text_transform(text_without_number)
        text_tokens_with_number = dataloader.text_transform(text_with_number)
        # get the number loss mask index
        number_loss_mask_start = len(text_tokens_without_number)-1
        number_loss_mask_end = len(text_tokens_with_number)-1
        # update texts
        texts.append(text_without_number)
        texts.append(text_with_number)
        # update texts_tokens
        texts_tokens.append(text_tokens_without_number)
        texts_tokens.append(text_tokens_with_number)
        # update numbers_loss_mask_locations
        numbers_loss_mask_locations.append([number_loss_mask_start,number_loss_mask_end])
    # update numbers_loss_mask
    tokenized_text = dataloader.text_transform(s)
    numbers_loss_mask = [0] * len(tokenized_text)
    for start, end in numbers_loss_mask_locations:
        numbers_loss_mask[start:end] = [1] * (end - start)
    
    if DEBUG:
        [print(number_location) for number_location in number_locations]
        [print(text) for text in texts]
        [print(text_tokens) for text_tokens in texts_tokens]    
        [print(numbers_loss_mask_location) for numbers_loss_mask_location in numbers_loss_mask_locations]
        print(f'numbers_loss_mask:{numbers_loss_mask}')
                        
if __name__ == "__main__":
    test_laion2b_obj_loader()