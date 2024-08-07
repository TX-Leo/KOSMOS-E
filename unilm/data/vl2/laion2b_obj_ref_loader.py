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
import re

import torchvision.transforms as T
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator
from unilm.data.vl2.vl_base_loader import VLBaseLoader
from unilm.data.vl2.laion2b_loader import Laion2BLoader, NumpyNormalize
from unilm.data.vl2.laion2b_obj_loader import Laion2BObjLoader
from unilm.data.vl2.obj_utils import *

from unilm.data.vl2.grounding_pron_prompts import pron_templates, simple_pron_templates

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

class Laion2BObjRefLoader(Laion2BObjLoader):
    def _setup(self):
        self.max_image_num = self.args.max_image_num
        self.image_token_length = self.args.image_token_length
        self.input_resolution = self.args.input_resolution
        self.quantized_size = self.args.quantized_size
        self.quantized_num = self.quantized_size ** 2
        self.box_score_threshold = self.args.box_score_threshold
        self.use_locate_special_token = bool(self.args.locate_special_token)
        
        self.phrase_mode = self.args.phrase_mode
        assert self.phrase_mode in ['phrase', 'expression']
        
        # referring setttings
        self.refer_image_only_resize = bool(self.args.refer_image_only_resize)
        self.refer_use_single_obj_prob = self.args.refer_use_single_obj_prob
        self.refer_ignore_eos_grad = bool(self.args.refer_ignore_eos_grad)
        self.refer_use_simple_template = bool(self.args.refer_use_simple_template)
        self.refer_use_short_expression_prob = self.args.refer_use_short_expression_prob
        self.refer_use_exp_w_multibox = bool(self.args.refer_use_exp_w_multibox)
        self.refer_use_exp_start_w_a_the = bool(self.args.refer_use_exp_start_w_a_the)
        
        # statistic the number of vocab
        tokenizer_vocabs = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        # dictionary_vocabs = self.dictionary.symbols
        self.tokenizer_vocab_num = len(tokenizer_vocabs)
        
        # logger.info(f"Vocab length in tokenizer: {self.tokenizer_vocab_num}")
        # logger.info(f"Vocab length in dictionary: {len(self.dictionary.symbols)}")
        logger.info(f"Only use resize transform for refer data: {self.refer_image_only_resize}")
        logger.info(f"refer use single obj prob: {self.refer_use_single_obj_prob}")
        logger.info(f"refer use short expression prob: {self.refer_use_short_expression_prob}")
        logger.info(f"refer use simple template: {self.refer_use_simple_template}")
        logger.info(f"refer ignore eos grad: {self.refer_ignore_eos_grad}")
        logger.info(f"refer use the expression for multiboxes: {self.refer_use_exp_w_multibox}")
        logger.info(f"refer use the expression start with a/the: {self.refer_use_exp_start_w_a_the}")
        
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
        if getattr(self, 'refer_image_only_resize', False):
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
            
            # pdb.set_trace()
            ret_batch = {
                'vl_tune':{ 
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
        text_tokens = doc[CAPTION_KEY]['input'] + doc[CAPTION_KEY]['output']
        image_tokens = doc[IMAGE_KEY]
        text_length = len(text_tokens)
        text_tokens = [bos_id] + [boi_id] * (self.image_token_length + 1) + [eoi_id] + text_tokens
        text_input_mask = [0]  + [0]  + [1] * (self.image_token_length) + [0] + [0] * text_length
        # Note: the last token of inputs has gradients (the target token is the first output token)
        text_loss_mask =  [0]  + [0]  + [0] * (self.image_token_length) + [0] + [0] * (len(doc[CAPTION_KEY]['input'])-1) + [1] * (len(doc[CAPTION_KEY]['output']) + 1)
        if doc[CAPTION_KEY].get('ignore_eos_gradient', False):
            text_loss_mask[-2] = 0 # the last vaild input token (not <eos>, but it's target token is eos)
            text_loss_mask[-1] = 0 # the last token is <eos>, may this step is not necessary here
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
                if len(item) < 6:
                    # no box, we drop
                    continue
                
                # pdb.set_trace()
                grounding_list = ast.literal_eval(item[-1])['expression_v1']
                
                cluster_obj_dict = {}
                for (phrase_start, phrase_end, x1, y1, x2, y2, score) in grounding_list:
                    # filter the low-score
                    if self.box_score_threshold > score: 
                        continue
                    
                    # filter the too large box
                    if (x2 - x1) * (y2 - y1) > 0.9 * 0.9:
                        continue
                    
                    expression = caption[phrase_start: phrase_end]
                    if expression not in cluster_obj_dict.keys():
                        cluster_obj_dict[expression] = [[], [], []]
                    cluster_obj_dict[expression][0].append([phrase_start, phrase_end])
                    cluster_obj_dict[expression][1].append([x1, y1, x2, y2])
                    cluster_obj_dict[expression][2].append([score])
            
                # random using the data that only contains one expression/box
                if len(list(cluster_obj_dict.keys())) == 1:
                    if random.random() > self.refer_use_single_obj_prob:
                        # print("fliter because contain single obj")
                        continue
                # add some rules to exclude some expression-box pairs
                # 1. filter the expression contains number
                # 2. remove the expression that is a single world
                filter_cluster_obj_dict = {}
                for expression in cluster_obj_dict.keys():
                    number = find_first_number(expression)
                    if number is not None and number > 1:
                        continue
                    if is_single_word(expression):
                        # a single world always be person name or nouns
                        continue
                    if starts_with_demonstrative(expression):
                        # start with some prons
                        continue
                    if first_word_plural(expression):
                        continue
                    if len(re.findall(r'\w+', expression)) < 4:
                        if random.random() > self.refer_use_short_expression_prob:
                            continue
                    if len(cluster_obj_dict[expression][0]) > 1:
                        # filter the expression correspondding to multiple boxes if not refer_use_exp_w_multibox
                        if not self.refer_use_exp_w_multibox:
                            continue
                    if self.refer_use_exp_start_w_a_the:
                        if re.findall(r'\w+', expression.lower())[0] not in ['a', 'an', 'the', 'one']:
                            continue
                    
                    filter_cluster_obj_dict[expression] = cluster_obj_dict[expression]
                
                grounding_list_fliter = []
                for _, grd_list in filter_cluster_obj_dict.items(): 
                    assert len(grd_list[0]) == len(grd_list[1]) == len(grd_list[2])
                    for grd_index in range(len(grd_list[0])):
                        _grd = []
                        _grd.extend(grd_list[0][grd_index])
                        _grd.extend(grd_list[1][grd_index])
                        _grd.extend(grd_list[2][grd_index])
                        grounding_list_fliter.append(_grd)
                
                p_item = [0, caption, None, None, None, str(grounding_list_fliter)]
                cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase',
                                                            drop_crop_thr=0.3, perform_centercrop=(not self.refer_image_only_resize))
                
                if len(cluster_obj_dict) == 0:
                    continue
                
                # random choose a expression
                random_key = random.choice(list(cluster_obj_dict.keys()))  
                phrase_start, phrase_end = random_key
                common_index = random.randrange(len(cluster_obj_dict[random_key][0]))  
                ul_index, lr_index = cluster_obj_dict[random_key][0][common_index]  
                score = cluster_obj_dict[random_key][1][common_index]  
                croped_box = cluster_obj_dict[random_key][2][common_index] 
                expression = caption[phrase_start: phrase_end]
                
                # random choose a template
                if self.refer_use_simple_template:
                    region_caption_template = random.sample(simple_pron_templates, 1)[0]
                else:
                    region_caption_template = random.sample(pron_templates, 1)[0]
                # qa_string = region_caption_template['Question'] + region_caption_template['Answer']
                
                ul_token_name = f"<patch_index_{str(ul_index).zfill(4)}>"
                lr_token_name = f"<patch_index_{str(lr_index).zfill(4)}>"
                coord_string = BOO_SYMBOL + ul_token_name + lr_token_name + EOO_SYMBOL
                
                if not first_two_words_uppercase(expression):
                    expression = expression[0].lower() + expression[1:]
                
                if region_caption_template['Question']:
                    q_string = region_caption_template['Question'].format(x1y1x2y2=coord_string)
                    question_string_encode = self._encode_a_string_w_special_tokens(q_string)
                    if self.use_locate_special_token:
                        question_string_encode = [self.dictionary.index(GRD_SYMBOL),] + question_string_encode
                        
                    a_string = region_caption_template['Answer'].format(expression=expression)
                    answer_string_encode = self.text_transform(a_string)
                else:
                    # no question
                    qa_string = region_caption_template['Answer'].format(x1y1x2y2=coord_string, expression=expression)
                    qa_string_wo_exp = qa_string[:qa_string.find(expression)]
                            
                    # just calculate the gradients on the expression here
                    question_string_encode = self._encode_a_string_w_special_tokens(qa_string_wo_exp)
                    if self.use_locate_special_token:
                        question_string_encode = [self.dictionary.index(GRD_SYMBOL),] + question_string_encode
                    answer_string_encode = self.text_transform(expression) + [self.dictionary.eos_index]
                
                # show_cluster_obj_dict = {(phrase_start, phrase_end): [[(ul_index, lr_index)], [score], [croped_box]]}
                # visualize_normed_img_with_bbox(torch_tensor, show_cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                # pdb.set_trace()
                # print(f"{expression} {self._decode_id_to_piece(question_string_encode + answer_string_encode)}")
                if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                    print(f"Note, the eos index appear in the meidum sentence (encode: {question_string_encode + answer_string_encode})")
                
                json_obj[CAPTION_KEY] = {
                    'input': question_string_encode,
                    'output': answer_string_encode,
                    'ignore_eos_gradient': self.refer_ignore_eos_grad,
                }  
                                        
                yield json_obj
            # except KeyboardInterrupt as e:
            #     raise ValueError
            except Exception as e:
                # raise
            #     # logger.warning(f"{e}, skip this image-text pair data")
                continue
            
    def _split_string(self, string, separators):
        """
        Function to split a given string based on a list of separators.
        """
        pattern = "|".join(re.escape(separator) for separator in separators) 
        result = re.split(f'({pattern})', string)  
        return [elem for elem in result if elem] 
    
    def _encode_a_string_w_special_tokens(self, caption):
        special_tokens = [self.dictionary[idx] for idx in range(self.dictionary.index(BOI_SYMBOL), len(self.dictionary))]
        split_resutls = self._split_string(caption, special_tokens)
        tokenized_codes = []
        for string in split_resutls:
            if string in special_tokens:
                tokenized_codes.append(self.dictionary.index(string))
            else:
                encode_tokens = self.text_transform(string)
                tokenized_codes.extend(encode_tokens)
        return tokenized_codes

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
        10: 10,
        100: 100,
        1000: 1000,
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


def is_single_word(string):  
    if not isinstance(string, str) or len(string) == 0:  
        return False  
  
    words = re.findall(r'\w+', string)  
    return len(words) == 1 

def starts_with_demonstrative(string):  
  
    demonstratives = ["these", "this", "that", "those", 'other', 
                      'another', 'all', 'their', 'her', 'his',
                      'my', 'our', "it", 'its', 'several', 'some',
                      'many', 'multiple', 'hundreds', 'thoundreds', 
                      'half', 'little', 'multi', 'any', '©']  
    lower_case_string = string.lower()  
      
    first_word = re.findall(r'\w+', lower_case_string)[0]  
  
    return first_word in demonstratives

def first_two_words_uppercase(string):  
    if not isinstance(string, str) or len(string) == 0:  
        return False  
  
    words = re.findall(r'\w+', string)  
    if len(words) < 2:  
        return False  
    
    if words[0].isupper():
        return True
    
    return words[0][0].isupper() and words[1][0].isupper()

def is_plural(word):  
    irregular_plurals = {  
        "men": "man",  
        "women": "woman",  
        "children": "child",  
        "feet": "foot",  
        "teeth": "tooth",  
        "mice": "mouse",  
        "geese": "goose",  
        "lives": "life",  
        "wives": "wife",  
        "knives": "knife",  
        "leaves": "leaf",  
        "loaves": "loaf",  
        "halves": "half",  
        "scarves": "scarf",  
        "sheep": "sheep",  
        "fish": "fish",  
        "deer": "deer",  
        "aircraft": "aircraft",  
        "series": "series",  
        "species": "species",  
        "people": "person",  
        "oxen": "ox",  
        "dice": "die",  
        "pence": "penny",  
        "indices": "index",  
        "appendices": "appendix",  
        "theses": "thesis",  
        "analyses": "analysis",  
        "diagnoses": "diagnosis",  
        "phenomena": "phenomenon"  
    }  
  
    if word in irregular_plurals:  
        return True  
    elif word.endswith("s"):  
        return True  
    elif word.endswith("ies") and len(word) > 3:  
        return True  
    elif word.endswith("ves") and len(word) > 3:  
        return True  
    else:  
        return False  
  
def first_word_plural(sentence):  
    words = re.findall(r'\w+', sentence)
    return is_plural(words[0])  
