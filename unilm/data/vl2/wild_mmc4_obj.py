import json
import os
import multiprocessing
import itertools
import ast

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
from unilm.data.vl2.laion2b_loader import NumpyNormalize
from unilm.data.vl2.laion2b_obj_loader import Laion2BObjLoader, find_substring_pairs, insert_bounding_boxes
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
CONTENT_KEY='Content'
IMAGE_KEY='MMImage'
TEXT_KEY="Extracted"

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

class Mmc4ObjLoader(Laion2BObjLoader):
    def _setup(self):
        self.max_image_num = self.args.max_image_num
        self.image_token_length = self.args.image_token_length
        self.input_resolution = self.args.input_resolution
        self.quantized_size = self.args.quantized_size
        self.quantized_num = self.quantized_size ** 2
        self.box_score_threshold = self.args.box_score_threshold
        self.mix_no_object_prob = self.args.mix_no_object_prob
        self.use_locate_special_token = bool(self.args.locate_special_token)
        
        self.phrase_mode = self.args.phrase_mode
        assert self.phrase_mode in ['phrase', 'expression']
        
        # mmc4 parameters
        self.mmc4_min_ground_labels = self.args.mmc4_min_ground_labels
        self.mmc4_clean_before_first_ground = self.args.mmc4_clean_before_first_ground
        
        # statistic the number of vocab
        tokenizer_vocabs = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        # dictionary_vocabs = self.dictionary.symbols
        self.tokenizer_vocab_num = len(tokenizer_vocabs)
        
        logger.info(f"Enabling {self.phrase_mode}-mode for phrase name")
        logger.info(f"Mixing prob {self.mix_no_object_prob} for using image-text pair without no grounding label")
        logger.info(f"Vocab length in tokenizer: {self.tokenizer_vocab_num}")
        logger.info(f"Min grounded box num: {self.mmc4_min_ground_labels}")
        logger.info(f"Crop to the first image with box: {self.mmc4_clean_before_first_ground}")
        logger.info(f"Vocab length in tokenizer: {self.tokenizer_vocab_num}")
        logger.info(f"Vocab length in dictionary: {len(self.dictionary.symbols)}")
        
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
        return preprocess_image
    
    def _build_text_transform(self):
        def text_transform(text):
            append_eos=False
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
            image_shape = batch[0][1][0].shape # (3, 224, 224)
            image_num = sum([len(x[1]) for x in batch])
            
            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)
            chunk_tokens_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            segment_tokens_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            
            image_source_ids = np.full(shape=(image_num, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32,
                                 fill_value=self.dictionary.pad())
            all_image_tokens = []
            
            for i, (full_tokens, image_tokens, text_input_mask, text_loss_mask, chunk_tokens, segment_tokens) in enumerate(batch):
                gpt_source_ids[i, :len(full_tokens)-1] = full_tokens[:-1]
                gpt_target_ids[i, :len(full_tokens)-1] = full_tokens[1:]
                gpt_input_mask_all[i, :len(full_tokens)-1] = text_input_mask[:-1]
                gpt_loss_mask_all[i, :len(full_tokens)-1] = text_loss_mask[:-1]
                chunk_tokens_all[i, :len(full_tokens)-1] = chunk_tokens[:-1]
                segment_tokens_all[i, :len(full_tokens)-1] = segment_tokens[:-1]
                # image_source_ids[i] = image_tokens
                all_image_tokens.extend(image_tokens)
            
            # pdb.set_trace()
            image_source_ids = np.stack(all_image_tokens)
            ret_batch = {
                'vl_mmc4':{
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
        text_tokens = doc[TEXT_KEY]
        image_tokens = doc[IMAGE_KEY]
        text_input_mask = doc['input_mask']
        text_loss_mask = doc['loss_mask']
        chunk_tokens = doc['chunk_tokens']
        segment_tokens = doc['segment_tokens']
        return text_tokens, image_tokens, text_input_mask, text_loss_mask, chunk_tokens, segment_tokens
    
    def _read_from_files(self, source_file):
        file_path = os.path.join(self.data_dir, source_file)

        # print(file_path)
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        for i, doc_str in enumerate(lines):
            
            item = doc_str.strip().split('\t')
            instruction = item[0]
        
            # skip the sample without grounding labels if prob > mix_no_object_prob
            if 'g' not in instruction and random.random() > self.mix_no_object_prob:
                # not visualize
                continue
            
            # grounding data numbers
            try:
                grounding_dict = ast.literal_eval(item[-1])
            except:
                # the last one is not grounding data
                continue
            
            if len(grounding_dict) < self.mmc4_min_ground_labels:
                continue
            
            # all grounding box are cover the entire image
            all_grounding_lists = [grounding_dict[a_i]['expression_v1'] for a_i in grounding_dict.keys()]
            isfilter = True
            for all_grounding_list in all_grounding_lists:
                for grounding_list in all_grounding_list:
                    phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score = grounding_list
                    if (x2_norm - x1_norm) * (y2_norm - y1_norm) < 0.9 * 0.9:
                        isfilter = False
            if isfilter:
                continue
            
            if self.mmc4_clean_before_first_ground:
                # crop image or text without grouding data, put the 1-st ground image & text as the begain
                min_grounded_text_index = min(grounding_dict.keys()) # can not be 0
                act_text_index = -1
                for ins_i, ins in enumerate(list(instruction)):
                    if ins == 't':
                        act_text_index += 1
                        if act_text_index == min_grounded_text_index:
                            # pdb.set_trace()
                            break
                instruction_start = ins_i - 1 # should be an image 
                item_start = instruction_start + 1
                text_index = min_grounded_text_index - 1
            else:
                instruction_start = 0
                item_start = 1
                text_index = -1

            if len(instruction[instruction_start:-1]) != len(item[item_start:-1]):
                # the sample length is wrong
                continue
            
            bos_id = self.dictionary.bos()
            boi_id = self.dictionary.index(BOI_SYMBOL) 
            eoi_id = self.dictionary.index(EOI_SYMBOL)
                
            # prepare text and image tokens
            doc = [bos_id]
            image_tokens = []
            doc_input_mask = [0]
            doc_loss_mask = [0] if self.mmc4_clean_before_first_ground else [1]
            chunk_tokens = [0]
            segment_tokens = [0]
            sample_is_valid = True
            
            for instrc, item_e in zip(list(instruction)[instruction_start:-1], item[item_start:-1]):
                image_is_valid = True
                if instrc == 'i':
                    try:
                        pil_img = Image.open(io.BytesIO(base64.b64decode(item_e))).convert("RGB")
                    except Exception as e:
                        print(f"loading image from {item_e} arise error: {e}")
                        image_is_valid = False
                        continue
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_transform(pil_img)
                    
                    # append image token
                    doc.extend([boi_id] * (self.image_token_length + 1) + [eoi_id])
                    doc_input_mask.extend([0] + [1] * self.image_token_length + [0])
                    doc_loss_mask.extend([0] + [0] * self.image_token_length + [1])
                    chunk_id = chunk_tokens[-1] + 1
                    chunk_tokens.extend([chunk_id] * (self.image_token_length + 2))
                    segment_tokens.extend([1] * (self.image_token_length + 2))
                    image_tokens.append(torch_tensor)
                    
                elif instrc == 't':
                    text_index += 1
                    if text_index in grounding_dict.keys():
                        # have grounding data
                        caption = item_e
                        
                        if not image_is_valid:
                            text_token = self.text_transform(caption)
                        else:
                            try:
                                p_item = [None, caption, None, None, None, str(grounding_dict[text_index])]
                                cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), 
                                                                    mode=self.phrase_mode, mode_switch_prob=0.5)
                                if len(cluster_obj_dict) > 0:
                                    # pdb.set_trace()
                                    # visualize for check
                                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, f"test_item_{i}_{text_index}")
                                    tokenized_id_list = self.text_transform(caption)
                                    new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                                    if self.use_locate_special_token:
                                        new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                                    text_token = new_tokenized_id_list
                                else:
                                    text_token = self.text_transform(caption)
                            except:
                                sample_is_valid = False
                                text_token = self.text_transform(caption)
                                continue
                    else:
                        text_token = self.text_transform(item_e)
                    
                    # append text token
                    doc.extend(text_token)
                    doc_input_mask.extend([0] * len(text_token))
                    doc_loss_mask.extend([1] * len(text_token))
                    chunk_id = chunk_tokens[-1]
                    chunk_tokens.extend([chunk_id] * len(text_token))
                    segment_tokens.extend([0] * len(text_token))

            # print(self._decode_id_to_piece(doc))
            # pdb.set_trace()
            if not sample_is_valid or len(doc) > self.tokens_per_sample:
                continue 
            
            # append eos token at the end
            doc.append(self.dictionary.eos())
            doc_input_mask.append(0)
            doc_loss_mask.append(1)
            chunk_tokens.append(chunk_tokens[-1])
            segment_tokens.append(0)
            yield {
                TEXT_KEY: doc,
                IMAGE_KEY: image_tokens,
                'input_mask': doc_input_mask,
                'loss_mask': doc_loss_mask,
                'chunk_tokens': chunk_tokens,
                'segment_tokens': segment_tokens,
            }