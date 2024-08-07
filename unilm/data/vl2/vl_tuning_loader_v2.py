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
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator
from unilm.data.vl2.vl_base_loader import VLBaseLoader
from unilm.data.vl2.laion2b_loader import Laion2BLoader, NumpyNormalize
from unilm.data.vl2.laion2b_obj_loader import Laion2BObjLoader
from unilm.data.vl2.grounding_pron_prompts import (
    pron_templates, brief_caption_templates, text_templates, vqa_templates
)
from unilm.data.vl2.obj_utils import *

import logging
logger = logging.getLogger(__name__)

from PIL import Image
import base64
import io

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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

# /mnt/msranlp/zliang/data/tuning/LLaVA-Instruct-150K

class InstructVLLoader(Laion2BObjLoader):
    def _setup(self):
        self.max_image_num = self.args.max_image_num
        self.image_token_length = self.args.image_token_length
        self.input_resolution = self.args.input_resolution
        self.quantized_size = self.args.quantized_size
        self.quantized_num = self.quantized_size ** 2
        self.box_score_threshold = self.args.box_score_threshold
        self.vl_instru_dataset = self.args.vl_instru_dataset
        self.simplest_grounding_prompt = self.args.simplest_grounding_prompt
        self.use_locate_special_token = bool(self.args.locate_special_token)
        self.region_caption_template_prob = self.args.region_caption_template_prob
        self.region_caption_template_data = self.args.region_caption_template_data.split(',')
        self.flickr_tuning_mode = self.args.flickr_tuning_mode.split(',')
        self.flickr_tuning_mode_prob = list(map(float, self.args.flickr_tuning_mode_prob.split(',')))
        self.flickr_caption_template = bool(self.args.flickr_caption_template)
        self.flickr_caption_ignore_eos_gra = bool(self.args.flickr_caption_ignore_eos_gra)
        self.text_tuning_template = bool(self.args.text_tuning_template)
        self.llava_tuning_splits = self.args.llava_tuning_splits.split(',')
        self.llava_conversation_multiturn = bool(self.args.llava_conversation_multiturn)
        self.llava_question_template = bool(self.args.llava_question_template)
        self.vcr_tuning_mode = self.args.vcr_tuning_mode
        
        # this image transform is just used in vcr dataset
        self.image_resize_transform = self._build_image_resize_transform()
        # overwrite the image transform if tuning_image_only_resize is enable
        # need to ablate to determine the process method: centercrop or resize
        self.tuning_image_only_resize = bool(self.args.tuning_image_only_resize)
        
        self.dictionary.add_symbol(BOI_SYMBOL)
        self.dictionary.add_symbol(EOI_SYMBOL)

        logger.info(f"VL tuning dataset : {self.vl_instru_dataset}")
        logger.info(f"Using pron-template prob: {self.region_caption_template_prob}")
        logger.info(f"Using pron-template data: {self.region_caption_template_data}")
        logger.info(f"Flickr tuning type: {self.flickr_tuning_mode}")
        logger.info(f"Flickr tuning type prob: {self.flickr_tuning_mode_prob}")
        logger.info(f"Flickr tuning using caption template: {self.flickr_caption_template}")
        logger.info(f"Flickr caption tuning ignore <eos> gradient : {self.flickr_caption_ignore_eos_gra}")
        logger.info(f"Text using template : {self.text_tuning_template}")
        logger.info(f"VCR tuning mode: {self.vcr_tuning_mode}")
        
        logger.info(f"Tuning only use resize image transform: {self.tuning_image_only_resize}")
        
        
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
    
    def _build_image_transform(self):
        # the default setting as pretraining
        preprocess_image = Compose([
            Resize(self.input_resolution, interpolation=BICUBIC),
            CenterCrop(self.input_resolution),
            NumpyNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        if getattr(self, 'tuning_image_only_resize', False):
            return self._build_image_resize_transform()
        return preprocess_image
    
    def _build_image_resize_transform(self):
        # only perform resize transform 
        preprocess_image = Compose([
            Resize((self.input_resolution, self.input_resolution), interpolation=BICUBIC),
            NumpyNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return preprocess_image

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
            # self._decode_id_to_piece(gpt_source_ids[0][gpt_loss_mask_all[0].nonzero()[0]])
            # self._decode_id_to_piece(gpt_target_ids[0][gpt_loss_mask_all[0].nonzero()[0]])
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
        # print(file_path)
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        if 'LLaVA' in file_path and 'llava' in self.vl_instru_dataset:
            if os.path.basename(file_path).split('_')[0] not in self.llava_tuning_splits:
                print(f"Unchoiced splits in {self.llava_tuning_splits} for {file_path}")
                return iter([])
            print(file_path)
            for doc_str in lines:
                json_obj = {}
                item = doc_str.strip().split('\t')
                # tsv_str = "{item['id']}\t{encoded_image}\t{conversations}\n
                try:
                    # json_obj[CAPTION_KEY] = self.text_transform(item[1])
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    torch_tensor = self.image_transform(pil_img)
                    json_obj[IMAGE_KEY] = torch_tensor
                    
                    if self.llava_question_template:
                        qa_template = random.sample(vqa_templates, 1)[0]
                    else:
                        qa_template = "{question}"
                    # encode the conversations data
                    conversations = ast.literal_eval(item[2])
                    if len(conversations) > 2:
                        # pdb.set_trace()
                        # multi-turn conversation data
                        if is_valid_conversation(conversations):  
                            if self.llava_conversation_multiturn:
                                groups = group_conversations(conversations)  
                                num_groups = random.randint(1, len(groups))  
                                random_groups = get_random_groups(groups, num_groups)  
                                flattened_groups = [chat for group in random_groups for chat in group]
                            else:
                                groups = group_conversations(conversations)
                                index = random.randint(0, len(groups))
                                flattened_groups = groups[index]
                        else:
                            print(f"Unknow conversations length{len(conversations)}:\n {conversations}")
                            continue
                        
                        human_input_string = ""
                        for i in range(len(flattened_groups)-1):
                            string_clean = conversations[i]['value'].replace("<image>\n", "").replace("\n<image>", "").replace("following", "")
                            if conversations[i]['from'] == 'human':
                                string_clean = qa_template.format(question=string_clean)
                            human_input_string = human_input_string + string_clean + " "
                            # human_input_string = human_input_string + conversations[i]['value'].replace("<image>\n", "").replace("\n<image>", "") + " "
                        gpt_output_string = conversations[len(flattened_groups)-1]['value'].replace("<image>\n", "").replace("\n<image>", "")
                        # print(f"{human_input_string} {gpt_output_string}")
                        
                    elif len(conversations) == 2:
                        # complex reasoning and detail data
                        human_input = conversations[0]
                        if human_input['from'] == 'human':
                            string_clean = human_input['value'].replace("<image>\n", "").replace("\n<image>", "").replace("following", "")
                            human_input_string = qa_template.format(question=string_clean) + " "
                        else:
                            print(f"Unknow human input {human_input}")
                            continue
                            
                        gpt_output = conversations[1]
                        if gpt_output['from'] == 'gpt':
                            gpt_output_string = gpt_output['value'].replace("<image>\n", "").replace("\n<image>", "")
                        else:
                            print(f"Unknow gpt output {gpt_output}")  
                            continue
                        
                    else:
                        print(f"Unknow conversation length {len(conversations)} input \n {conversations}")
                        continue
                    
                    # print(human_input_string + gpt_output_string)
                    human_input_string_encode = self.text_transform(human_input_string)
                    gpt_output_string_encode = self.text_transform(gpt_output_string) + [self.dictionary.eos_index]
                    if self.dictionary.eos_index in (human_input_string_encode + gpt_output_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the medium of setence {item[2]} (encode: {human_input_string_encode + gpt_output_string_encode})")
                    json_obj[CAPTION_KEY] = {
                        'input': human_input_string_encode,
                        'output': gpt_output_string_encode,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise ValueError
                except:
                    continue
        elif ('refcoco' in file_path and 'refcoco' in self.vl_instru_dataset) or \
            ('clevr_ref' in file_path and 'clevr_ref' in self.vl_instru_dataset):
            print(file_path)
            for doc_str in lines:
                # pdb.set_trace()
                json_obj = {}
                item = doc_str.strip().split('\t')
                # tsv_str = "{image_name}\t{encoded_image}\t{caption}\t{str(convert_boxes)}\n"
                try:
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_transform(pil_img)
                    json_obj[IMAGE_KEY] = torch_tensor
                    
                    caption = item[2]
                    p_item = [0, caption, None, None, None, item[-1]]
                    cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase',
                                                              drop_crop_thr=0.1, perform_centercrop=(not self.tuning_image_only_resize))
                    
                    if len(cluster_obj_dict) == 0:
                        continue
                    elif len(cluster_obj_dict) == 1:
                        # pdb.set_trace()
                        # {(0, 63): [[(630, 927)], [1.0], [[0.7040312886238098, 0.620968759059906, 1.0, 0.9057187438011169]]]}
                        if random.random() < self.region_caption_template_prob and os.path.basename(file_path).split('_')[0] in self.region_caption_template_data:
                            region_caption_template = random.sample(pron_templates, 1)[0]
                        else:
                            region_caption_template = None
                            
                        if region_caption_template:
                            # pdb.set_trace()
                            qa_string = region_caption_template['Question'] + region_caption_template['Answer']
                            # pdb.set_trace()
                            ul_index, lr_index = list(cluster_obj_dict.values())[0][0][0]
                            ul_token_name = f"<patch_index_{str(ul_index).zfill(4)}>"
                            lr_token_name = f"<patch_index_{str(lr_index).zfill(4)}>"
                            coord_string = BOO_SYMBOL + ul_token_name + lr_token_name + EOO_SYMBOL
                            
                            expression=caption[:-1] if caption.endswith('.') else caption
                            qa_string = qa_string.format(x1y1x2y2=coord_string, expression=expression)
                            qa_string_wo_exp = qa_string[:qa_string.find(expression)]
                            
                            # just calculate the gradients on the expression here
                            question_string_encode = self._decode_a_string_w_special_tokens(qa_string_wo_exp)
                            if self.use_locate_special_token:
                                question_string_encode = [self.dictionary.index(GRD_SYMBOL),] + question_string_encode
                            answer_string_encode = self.text_transform(expression) + [self.dictionary.eos_index]
                            
                        else:
                            tokenized_id_list = self.text_transform(caption)
                            new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                            if self.use_locate_special_token:
                                new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                            question_string_encode = new_tokenized_id_list[:new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1]
                            answer_string_encode = new_tokenized_id_list[new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1:] + [self.dictionary.eos_index]
                    else:
                        continue
                    
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                    # pdb.set_trace()
                    # print(f"{item[0]} {self._decode_id_to_piece(question_string_encode + answer_string_encode)}")
                    if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the meidum sentence (encode: {question_string_encode + answer_string_encode})")
                    
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                        'ignore_eos_gradient': True,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except:
                    continue
        elif 'vcr' in file_path and 'vcr' in self.vl_instru_dataset:
            print(file_path)
            for doc_str in lines:
                # pdb.set_trace()
                json_obj = {}
                item = doc_str.strip().split('\t')
                try:
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_resize_transform(pil_img) # we only resize here, don't do crop
                    json_obj[IMAGE_KEY] = torch_tensor
                    
                    # pdb.set_trace()
                    qa_template = random.sample(vqa_templates, 1)[0]
                    gt_info_dict = ast.literal_eval(item[-1])
                    question = gt_info_dict['question_string'] # the encoded string, contain patch_index
                    answer = gt_info_dict['answer_choices_string'][gt_info_dict['answer_label']]
                    rationable = gt_info_dict['ration_choices_string'][gt_info_dict['ration_label']]
                    
                    if self.vcr_tuning_mode == 'qa':
                        question_string = qa_template.format(question=question)
                        question_string_encode = self._decode_a_string_w_special_tokens(question_string)
                        answer_string_encode = self._decode_a_string_w_special_tokens(answer)
                    elif self.vcr_tuning_mode == 'qar':
                        question_string = qa_template.format(question=question)
                        question_string_encode = self._decode_a_string_w_special_tokens(question_string)
                        answer_string_encode = self._decode_a_string_w_special_tokens(answer)
                        rationable_string_encode = self._decode_a_string_w_special_tokens(rationable)
                        answer_string_encode.extend(rationable_string_encode)
                    else:
                        raise NotImplementedError
                    
                    if self.use_locate_special_token:
                        question_string_encode = [self.dictionary.index(GRD_SYMBOL),] + question_string_encode
                    answer_string_encode += [self.dictionary.eos_index,]
                    # print(f"{item[0]} {self._decode_id_to_piece(question_string_encode + answer_string_encode)}")
                          
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                        'ignore_eos_gradient': False,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except NotImplementedError as e:
                    raise e
                except Exception as e:
                    # raise e
                    continue
                
        elif 'lvis' in file_path and 'lvis' in self.vl_instru_dataset:
            print(file_path)
            for doc_str in lines:
                # pdb.set_trace()
                json_obj = {}
                item = doc_str.strip().split('\t')
                # tsv_str = "{image_name}\t{encoded_image}\t{caption}\t{str(convert_boxes)}\n"
                # caption: [[single], [plur], [neg]]
                # convert_boxes: {'category 1': [[box 1], [box 2]]}
                try:
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_transform(pil_img)
                    json_obj[IMAGE_KEY] = torch_tensor
                    
                    caption_list = ast.literal_eval(item[2])
                    
                    # For simplify, we only choose one category and it's corresponding box annotations
                    caption = random.sample(caption_list[0], 1)[0]
                    all_anns = ast.literal_eval(item[-1])
                    choose_anns = all_anns[caption]
                    p_item = [0, caption, None, None, None, str(choose_anns)]
                    cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase',
                                                              drop_crop_thr=0.1, perform_centercrop=(not self.tuning_image_only_resize))
                    
                    # pdb.set_trace()
                    if len(cluster_obj_dict) == 0:
                        continue
                    
                    tokenized_id_list = self.text_transform(caption)
                    new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                    if self.use_locate_special_token:
                        new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                    question_string_encode = new_tokenized_id_list[:new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1]
                    answer_string_encode = new_tokenized_id_list[new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1:] + [self.dictionary.eos_index]
                    
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                    # pdb.set_trace()
                    # print(self._decode_id_to_piece(question_string_encode+answer_string_encode))
                    if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the meidum sentence (encode: {question_string_encode + answer_string_encode})")
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                        'ignore_eos_gradient': True,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except:
                    continue
        elif 'flickr' in file_path and 'flickr' in self.vl_instru_dataset:
            print(file_path)
            for doc_str in lines:
                # pdb.set_trace()
                json_obj = {}
                item = doc_str.strip().split('\t')
                # tsv_str = "{image_name}\t{encoded_image}\t{caption}\t{str(convert_boxes)}\n"
                try:
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_transform(pil_img)
                    json_obj[IMAGE_KEY] = torch_tensor
                    ignore_eos_gradient = True
                    caption = item[2]
                    
                    # exists many extra infomation in item[-1]
                    all_ann_info = ast.literal_eval(item[-1])
                    phrases_list = all_ann_info['phrases_list']
                    expressions_list = all_ann_info['expressions_list']
                    is_complete_sentence = not any(all_ann_info['extension_to_whole']) # the caption is a complete sentence or not
                    exist_extend = [p[0]!=e[0] or p[1] != e[1] for p, e in zip(phrases_list, expressions_list)]
                    # prob of exist_extend: 0.53; prob of complete_sentence: 0.815
                    
                    # pdb.set_trace()
                    choice_mode = random.choices(self.flickr_tuning_mode, weights=self.flickr_tuning_mode_prob, k=1)[0]
                    if not any(exist_extend):
                        if 'caption' in self.flickr_tuning_mode:
                            choice_mode = 'caption'
                        
                    if choice_mode == 'grounding':
                        # here, we prompt the model with a expression, and only calculate loss on
                        # patch index token. Ignore the gradient of <eos>.
                        
                        # p_item = [0, caption, None, None, None, str({'expression_v1': expressions_list})]
                        # cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='expression')
                        p_item = [0, caption, None, None, None, str([expressions for ei,expressions in enumerate(expressions_list) if exist_extend[ei]])]
                        cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase',
                                                                  drop_crop_thr=0.1, perform_centercrop=(not self.tuning_image_only_resize))
                        if len(cluster_obj_dict) == 0:
                            continue
                        choice_key, choice_value = random.choices(list(cluster_obj_dict.items()))[0]
                        expression_name = caption[choice_key[0]: choice_key[1]]
                        
                        tokenized_id_list = self.text_transform(expression_name)
                        new_tokenized_id_list = self._embed_box_after_phrase(expression_name, tokenized_id_list, {(0, len(expression_name)): choice_value}, has_eos=False)
                        if self.use_locate_special_token:
                            new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                        question_string_encode = new_tokenized_id_list[:new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1]
                        answer_string_encode = new_tokenized_id_list[new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1:] + [self.dictionary.eos_index]
                        ignore_eos_gradient = True
                        # visualize_normed_img_with_bbox(torch_tensor, {(0, len(expression_name)): choice_value}, self.quantized_size, expression_name, item[0].split('/')[-1].replace(".jpg", ""))
                    
                    elif choice_mode == 'caption':
                        # here, we prompt the model with caption-instruction, and calculate loss on
                        # the other tokens (including the <eos>)
                        
                        # pdb.set_trace()
                        if self.flickr_caption_template:
                            caption_template = random.choices(brief_caption_templates)[0]
                            caption_template_id_list = self.text_transform(caption_template)
                            
                            p_item = [0, caption, None, None, None, str({'expression_v1': expressions_list})]
                            cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='expression',
                                                                      drop_crop_thr=0.1, perform_centercrop=(not self.tuning_image_only_resize))
                            if len(cluster_obj_dict) == 0:
                                continue
                            
                            tokenized_id_list = self.text_transform(caption)
                            new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                            
                            question_string_encode = caption_template_id_list
                            if self.use_locate_special_token:
                                question_string_encode = [self.dictionary.index(GRD_SYMBOL),] + question_string_encode
                            answer_string_encode = new_tokenized_id_list + [self.dictionary.eos_index]
                            # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                        else:
                            p_item = [0, caption, None, None, None, str({'expression_v1': expressions_list})]
                            cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='expression',
                                                                      drop_crop_thr=0.1, perform_centercrop=(not self.tuning_image_only_resize))
                            if len(cluster_obj_dict) == 0:
                                continue
                            tokenized_id_list = self.text_transform(caption)
                            new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                            if self.use_locate_special_token:
                                new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                            question_string_encode = new_tokenized_id_list[:new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1]
                            answer_string_encode = new_tokenized_id_list[new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1:] + [self.dictionary.eos_index]
                            
                        if self.flickr_caption_ignore_eos_gra:
                            ignore_eos_gradient = True
                        else:
                            ignore_eos_gradient = False
                        # pdb.set_trace()
                    else:
                        continue
                    
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                    # pdb.set_trace()
                    # print(f"{item[0]} {self._decode_id_to_piece(question_string_encode + answer_string_encode)}")
                    if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the meidum sentence (encode: {question_string_encode + answer_string_encode})")
                    
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                        'ignore_eos_gradient': ignore_eos_gradient,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except Exception as e:
                    # print(e)
                    continue
                    # raise e
        elif ('cocotext' in file_path and 'cocotext' in self.vl_instru_dataset) or \
            ('totaltext' in file_path and 'totaltext' in self.vl_instru_dataset):
            print(file_path)
            for doc_str in lines:
                # pdb.set_trace()
                json_obj = {}
                item = doc_str.strip().split('\t')
                # tsv_str = "{image_name}\t{encoded_image}\t{str(convert_boxes)}\n"
                # convert_boxes: [{word: [box]}]
                try:
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_transform(pil_img)
                    json_obj[IMAGE_KEY] = torch_tensor
                    
                    caption_list = ast.literal_eval(item[2])
                    
                    # pdb.set_trace()
                    # For simplify, we only random choose one word from labels
                    word_ann = random.sample(caption_list, 1)[0]
                    caption = list(word_ann.keys())[0]
                    anns = [word_ann[caption]]
                    
                    if len(caption) == 0:
                        continue
                    
                    p_item = [0, caption, None, None, None, str(anns)]
                    cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase', 
                                                              drop_crop_thr=0.6, perform_centercrop=(not self.tuning_image_only_resize))
                    
                    # pdb.set_trace()
                    if len(cluster_obj_dict) == 0:
                        continue
                    
                    # filter too small box
                    filtered_box = []
                    word_key = list(cluster_obj_dict.keys())[0]
                    for box in cluster_obj_dict[word_key][-1]:
                        if (box[2] - box[0]) * self.input_resolution < 16 and (box[3] - box[1]) * self.input_resolution < 16:
                            continue
                        else:
                            filtered_box.append(box)
                    if len(filtered_box) == 0:
                        continue
                    cluster_obj_dict[word_key][-1] = filtered_box
                        
                    if self.text_tuning_template:
                        text_template = random.choices(text_templates)[0]
                        text_template_id_list = self.text_transform(text_template)
                        
                        tokenized_id_list = self.text_transform(caption)
                        new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                        new_tokenized_id_list = text_template_id_list + new_tokenized_id_list
                    else:
                        tokenized_id_list = self.text_transform(caption)
                        new_tokenized_id_list = self._embed_box_after_phrase(caption, tokenized_id_list, cluster_obj_dict, has_eos=False)
                        
                    if self.use_locate_special_token:
                        new_tokenized_id_list = [self.dictionary.index(GRD_SYMBOL),] + new_tokenized_id_list
                    question_string_encode = new_tokenized_id_list[:new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1]
                    answer_string_encode = new_tokenized_id_list[new_tokenized_id_list.index(self.dictionary.index(EOP_SYMBOL))+1:] + [self.dictionary.eos_index]
                            
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                    # pdb.set_trace()
                    # print(self._decode_id_to_piece(question_string_encode))
                    # print(self._decode_id_to_piece(answer_string_encode))
                    if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the meidum sentence (encode: {question_string_encode + answer_string_encode})")
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                        'ignore_eos_gradient': True,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except Exception as e:
                    # print(e)
                    continue
                    # raise e
        else:
            # print(f"Unkonwn file type or not in {self.vl_instru_dataset}")
            return iter([]) # skip bad file
    
    def _split_string(self, string, separators):
        """
        Function to split a given string based on a list of separators.
        """
        pattern = "|".join(re.escape(separator) for separator in separators) 
        result = re.split(f'({pattern})', string)  
        return [elem for elem in result if elem] 
    
    def _decode_a_string_w_special_tokens(self, caption):
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
                
def is_valid_conversation(conversation):  
    if len(conversation) % 2 != 0:  
        return False  
  
    for i, chat in enumerate(conversation):  
        if i % 2 == 0 and chat["from"] != "human":  
            return False  
        if i % 2 != 0 and chat["from"] != "gpt":  
            return False  
  
    return True  
  
def group_conversations(conversation):  
    groups = []  
    for i in range(0, len(conversation), 2):  
        groups.append([conversation[i], conversation[i + 1]])  
    return groups  
  
def get_random_groups(groups, num_groups):  
    return random.sample(groups, num_groups)