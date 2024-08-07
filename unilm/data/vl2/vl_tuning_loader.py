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
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator
from unilm.data.vl2.vl_base_loader import VLBaseLoader
from unilm.data.vl2.laion2b_loader import Laion2BLoader, NumpyNormalize
from unilm.data.vl2.grounding_prompts import (
    single_expression, plural_expressions, multi_plural_expressions, no_expression,
    simplest_single_expression, simplest_plural_expressions, simplest_no_expression
)
from unilm.data.vl2.obj_utils import *

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

class InstructVLLoader(Laion2BLoader):
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
        self.dictionary.add_symbol(BOI_SYMBOL)
        self.dictionary.add_symbol(EOI_SYMBOL)
    
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
        text_loss_mask =  [0]  + [0]  + [0] * (self.image_token_length) + [0] + [0] * len(doc[CAPTION_KEY]['input']) + [1] * len(doc[CAPTION_KEY]['output'])
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
                    
                    # encode the conversations data
                    conversations = ast.literal_eval(item[2])
                    if len(conversations) > 2:
                        # multi-turn conversation data
                        if is_valid_conversation(conversations):  
                            groups = group_conversations(conversations)  
                            num_groups = random.randint(1, len(groups))  
                            random_groups = get_random_groups(groups, num_groups)  
                            flattened_groups = [chat for group in random_groups for chat in group]
                        else:
                            print(f"Unknow conversations length{len(conversations)}:\n {conversations}")
                            continue
                        
                        human_input_string = ""
                        for i in range(len(flattened_groups)-1):
                            human_input_string = human_input_string + conversations[i]['value'].replace("<image>\n", "").replace("\n<image>", "") + " "
                        gpt_output_string = conversations[len(flattened_groups)-1]['value'].replace("<image>\n", "").replace("\n<image>", "")
                            
                    elif len(conversations) == 2:
                        # complex reasoning and detail data
                        human_input = conversations[0]
                        if human_input['from'] == 'human':
                            human_input_string = human_input['value'].replace("<image>\n", "").replace("\n<image>", "")
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
        elif 'refcoco' in file_path and 'refcoco' in self.vl_instru_dataset:
            print(file_path)
            for doc_str in lines:
                # pdb.set_trace()
                json_obj = {}
                item = doc_str.strip().split('\t')
                # tsv_str = "{image_name}\t{encoded_image}\t{caption}\t{str(convert_boxes)}\n"
                try:
                    # json_obj[CAPTION_KEY] = self.text_transform(item[1])
                    pil_img = Image.open(io.BytesIO(base64.b64decode(item[1]))).convert("RGB")
                    ori_img_w, ori_img_h = pil_img.size
                    torch_tensor = self.image_transform(pil_img)
                    json_obj[IMAGE_KEY] = torch_tensor
                    
                    caption = item[2]
                    p_item = [0, caption, None, None, None, item[-1]]
                    cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase')
                    if len(cluster_obj_dict) == 0:
                        no_object_template = random.sample(no_expression, 1)[0] if not self.simplest_grounding_prompt else random.sample(simplest_no_expression, 1)[0]
                        question = no_object_template['question'].format(expression=caption)
                        answer = no_object_template['answer'].format(expression=caption).capitalize()
                        question_string_encode = self.text_transform(question)
                        answer_string_encode = self.text_transform(answer) + [self.dictionary.eos_index]
                    elif len(cluster_obj_dict) == 1:
                        # {(0, 63): [[(630, 927)], [1.0], [[0.7040312886238098, 0.620968759059906, 1.0, 0.9057187438011169]]]}
                        single_object_template = random.sample(single_expression, 1)[0] if not self.simplest_grounding_prompt else random.sample(simplest_single_expression, 1)[0]
                        question = single_object_template['question'].format(expression=caption.replace(".", ""))
                        question_string_encode = self.text_transform(question)
                        
                        # pdb.set_trace()
                        ul_index, lr_index = list(cluster_obj_dict.values())[0][0][0]
                        ul_token_name = f"<patch_index_{str(ul_index).zfill(4)}>"
                        lr_token_name = f"<patch_index_{str(lr_index).zfill(4)}>"
                        template_string = single_object_template['answer'].format(expression=caption.replace(".", ""), x1y1x2y2=f"x1y1x2y2").capitalize()
                        template_string_list = template_string.split('x1y1x2y2')
                        template_string_encode_list = list(map(self.text_transform, template_string_list))
                        coord_encode = [
                            self.dictionary.index(BOO_SYMBOL),
                            self.dictionary.index(ul_token_name),
                            self.dictionary.index(lr_token_name),
                            self.dictionary.index(EOO_SYMBOL),
                        ]
                        
                        # 1 we calculate gradients at all answer token
                        final_encode_list = [template_string_encode_list[0], coord_encode, template_string_encode_list[1]]
                        answer_string_encode = [subsub for sub in final_encode_list for subsub in sub] + [self.dictionary.eos_index]
                        
                        # # 2 we only calculate gradients at coordients and the string like "in the picture"
                        # question_string_encode.extend(template_string_encode_list[0])
                        # final_encode_list = [coord_encode, template_string_encode_list[1]]
                        # answer_string_encode = [subsub for sub in final_encode_list for subsub in sub] + [self.dictionary.eos_index]
                        
                        # string_decode = self._decode_id_to_piece(question_string_encode)
                        
                    else:
                        continue
                    
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                    # pdb.set_trace()
                    # print(self._decode_id_to_piece(question_string_encode+answer_string_encode))
                    if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the meidum sentence {question+template_string} (encode: {question_string_encode + answer_string_encode})")
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except:
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
                    prob = random.random()
                    if len(caption_list[-1]) > 0 and prob < 0.3:
                        # negtive object
                        caption = random.sample(caption_list[-1])[0]
                        no_object_template = random.sample(no_expression, 1)[0]
                        question = no_object_template['question'].format(expression=caption)
                        answer = no_object_template['answer'].format(expression=caption).capitalize()
                        question_string_encode = self.text_transform(question)
                        answer_string_encode = self.text_transform(answer) + [self.dictionary.eos_index]
                    
                    elif prob < 1.0:
                        # we only choose one category and it's corresponding box annotations
                        caption = random.sample(caption_list[0], 1)[0]
                        all_anns = ast.literal_eval(item[-1])
                        choose_anns = all_anns[caption]
                        
                        p_item = [0, caption, None, None, None, str(choose_anns)]
                        cluster_obj_dict = process_grounding_data(self, p_item, (ori_img_w, ori_img_h), mode='phrase')
                        # pdb.set_trace()
                        if len(cluster_obj_dict) == 0:
                            no_object_template = random.sample(no_expression, 1)[0] if not self.simplest_grounding_prompt else random.sample(no_expression, 1)[0]
                            question = no_object_template['question'].format(expression=caption)
                            answer = no_object_template['answer'].format(expression=caption).capitalize()
                            question_string_encode = self.text_transform(question)
                            answer_string_encode = self.text_transform(answer) + [self.dictionary.eos_index]
                        elif len(cluster_obj_dict) == 1 and len(list(cluster_obj_dict.values())[0][0]) == 1:
                            # single object
                            # {(0, 63): [[(630, 927)], [1.0], [[0.7040312886238098, 0.620968759059906, 1.0, 0.9057187438011169]]]}
                            single_object_template = random.sample(single_expression, 1)[0] if not self.simplest_grounding_prompt else random.sample(simplest_single_expression, 1)[0]
                            question = single_object_template['question'].format(expression=caption.replace(".", ""))
                            question_string_encode = self.text_transform(question)
                            
                            # pdb.set_trace()
                            ul_index, lr_index = list(cluster_obj_dict.values())[0][0][0]
                            ul_token_name = f"<patch_index_{str(ul_index).zfill(4)}>"
                            lr_token_name = f"<patch_index_{str(lr_index).zfill(4)}>"
                            template_string = single_object_template['answer'].format(expression=caption.replace(".", ""), x1y1x2y2=f"x1y1x2y2").capitalize()
                            template_string_list = template_string.split('x1y1x2y2')
                            template_string_encode_list = list(map(self.text_transform, template_string_list))
                            coord_encode = [
                                self.dictionary.index(BOO_SYMBOL),
                                self.dictionary.index(ul_token_name),
                                self.dictionary.index(lr_token_name),
                                self.dictionary.index(EOO_SYMBOL),
                            ]
                            
                            # 1 we calculate gradients at all answer token
                            final_encode_list = [template_string_encode_list[0], coord_encode, template_string_encode_list[1]]
                            answer_string_encode = [subsub for sub in final_encode_list for subsub in sub] + [self.dictionary.eos_index]
                            
                            # # 2 we only calculate gradients at coordients and the string like "in the picture"
                            # question_string_encode.extend(template_string_encode_list[0])
                            # final_encode_list = [coord_encode, template_string_encode_list[1]]
                            # answer_string_encode = [subsub for sub in final_encode_list for subsub in sub] + [self.dictionary.eos_index]
                            
                            # string_decode = self._decode_id_to_piece(question_string_encode)
                        else:
                            # multiple objects
                            # pdb.set_trace()
                            plural_object_template = random.sample(plural_expressions, 1)[0] if not self.simplest_grounding_prompt else random.sample(simplest_plural_expressions, 1)[0]
                            plur_caption = caption_list[1][caption_list[0].index(caption)]
                            question = plural_object_template['question'].format(expressions=plur_caption.replace(".", ""))
                            question_string_encode = self.text_transform(question)
                            
                            template_string = plural_object_template['answer'].format(expressions=plur_caption.replace(".", ""), x1y1x2y2=f"x1y1x2y2").capitalize()
                            template_string_list = template_string.split('x1y1x2y2')
                            template_string_encode_list = list(map(self.text_transform, template_string_list))
                            coord_encode = [self.dictionary.index(BOO_SYMBOL),]
                            for (ul_index, lr_index) in list(cluster_obj_dict.values())[0][0]:
                                ul_token_name = f"<patch_index_{str(ul_index).zfill(4)}>"
                                lr_token_name = f"<patch_index_{str(lr_index).zfill(4)}>"
                                coord_encode.append(self.dictionary.index(ul_token_name))
                                coord_encode.append(self.dictionary.index(lr_token_name))
                                coord_encode.append(self.dictionary.index(DOM_SYMBOL))
                            coord_encode[-1] = self.dictionary.index(EOO_SYMBOL)
                            
                            # 1 we calculate gradients at all answer token
                            final_encode_list = [template_string_encode_list[0], coord_encode, template_string_encode_list[1]]
                            answer_string_encode = [subsub for sub in final_encode_list for subsub in sub] + [self.dictionary.eos_index]
                            
                            # # 2 we only calculate gradients at coordients and the string like "in the picture"
                            # question_string_encode.extend(template_string_encode_list[0])
                            # final_encode_list = [coord_encode, template_string_encode_list[1]]
                            # answer_string_encode = [subsub for sub in final_encode_list for subsub in sub] + [self.dictionary.eos_index]
                            
                    else:
                        continue
                    
                    # visualize_normed_img_with_bbox(torch_tensor, cluster_obj_dict, self.quantized_size, caption, item[0].split('/')[-1].replace(".jpg", ""))
                    # pdb.set_trace()
                    # print(self._decode_id_to_piece(question_string_encode+answer_string_encode))
                    if self.dictionary.eos_index in (question_string_encode + answer_string_encode)[:-1]:
                        print(f"Note, the eos index appear in the meidum sentence {question+template_string} (encode: {question_string_encode + answer_string_encode})")
                    json_obj[CAPTION_KEY] = {
                        'input': question_string_encode,
                        'output': answer_string_encode,
                    }  
                    
                    yield json_obj
                except KeyboardInterrupt as e:
                    raise KeyboardInterrupt
                except:
                    continue
        else:
            print(f"Unkonwn type or not in {self.vl_instru_dataset}")
            return iter([]) # skip bad file
    
    def _decode_id_to_piece(self, tokenized_id_list):
        return [self.dictionary[idx] for idx in tokenized_id_list]

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