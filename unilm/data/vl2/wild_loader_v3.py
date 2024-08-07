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
from azure.storage.blob import ContainerClient # pip install azure-storage-blob

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator
from unilm.data.vl2.vl_base_loader import VLBaseLoader

from PIL import Image
import base64
import io, re

from spacy.lang.en import English

import logging
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)

# UUID  PAGE_URL    IMAGE_URL   ALT_TEXT    IMAGE_UUID    BASE64_ENC  OG_HEIGH    OG_WIDTH    RESIZED_HEIGHT  RESIZED_WIDTH
BASE64_KEY=5
IMAGE_KEY="Images"
TEXT_KEY="Extracted"
AZURE_URL="https://turingdata2.blob.core.windows.net/ita"
CREDENTIAL="sv=2020-08-04&st=2022-11-07T17%3A55%3A08Z&se=2023-06-08T16%3A55%3A00Z&sr=c&sp=rl&sig=LQwL3DhQLAGfZ7EIV3X8JhnnCRpAK2k1nNJHFPAOf1c%3D"
RAW_IMAGE_ROOT="2022-05/downloaded_images"
BOI_SYMBOL="<image>"
EOI_SYMBOL="</image>"

class NumpyNormalize(torch.nn.Module):
    def __init__(self,  mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor).
        Returns:
        """
        image = np.array(img).transpose(2, 0, 1)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class WildLoader(VLBaseLoader):
    def _setup(self):
        self.nlp_sentencizer = English()
        self.nlp_sentencizer.add_pipe("sentencizer")

        self.max_image_num = self.args.max_image_num
        self.image_token_length = self.args.image_token_length
        self.dictionary.add_symbol(BOI_SYMBOL)
        self.dictionary.add_symbol(EOI_SYMBOL)
        
        self.container_client = ContainerClient.from_container_url(
            container_url=AZURE_URL,
            credential=CREDENTIAL
        )

    def _build_filter(self):
        def max_image_num_filter(item):
            # TODO: add args
            return len(item[IMAGE_KEY]) > self.max_image_num
        def clip_score_filter(item):
            threshold = 0.28
            filter_flag = True
            for image_item in item[IMAGE_KEY]:
                if 'CLIP' not in image_item:
                    return True
                if image_item["CLIP"][0] > threshold or image_item["CLIP"][1] > threshold:
                    filter_flag = False
                    break
            return filter_flag
        return [max_image_num_filter, clip_score_filter]

    def _build_image_transform(self):
        preprocess_image = Compose([
            Resize(224),
            CenterCrop(224),
            NumpyNormalize(0.5, 0.5)
            # ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return preprocess_image
    
    def _build_text_transform(self):
        def text_transform(text):
            append_eos=False
            fs_dict = self.dictionary
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
                all_image_tokens.extend(image_tokens)

            # image_source_ids = torch.stack(all_image_tokens).numpy()
            image_source_ids = np.stack(all_image_tokens)

            ret_batch = {
                'vl':{
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

    def _get_image_token(self, image_file_name, offset):
        # partition.000_0000000000_0005014222.tsv
        #  -> '2022-05/downloaded_images/partition.000/resized/partition.000_0000000000_0005014222.tsv'
        part_id = image_file_name.split('_')[0]
        blob_name = f"{RAW_IMAGE_ROOT}/{part_id}/resized/{image_file_name}"
        blob_obj = self.container_client.get_blob_client(blob_name)
        image_text = blob_obj.download_blob(offset=offset, length=1000000).content_as_text(encoding='UTF-8')
        base64_image = image_text.split('\n')[0].split('\t')[BASE64_KEY]
        pil_img = Image.open(io.BytesIO(base64.b64decode(str(base64_image)))).convert("RGB")
        torch_tensor = self.image_transform(pil_img)
        # TODO: need to change to other format if torch tensor blocks the data loader
        return torch_tensor

    def clean(self, text):
        # python re, remove html tags
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _read_from_files(self, source_file):
        
        # merged_content = self.container_client.get_blob_client(source_file)
        # while True: # read until EOF
        #     try:
        #         print(f"Reading {source_file}")
        #         lines = merged_content.download_blob().content_as_text(encoding='UTF-8').strip().split('\n')
        #         break
        #     except:
        #         print(f"Error reading {source_file}")
        file_path = os.path.join(self.data_dir, source_file)
        print(file_path)
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        for doc_jsonstr in lines:
            try:
                json_obj = json.loads(doc_jsonstr)
            except Exception as e:
                print(e)
                continue
            # filter item based self.filter
            is_filter = False
            for filter in self.filters:
                if filter(json_obj):
                    is_filter = True
                    break
            if is_filter:
                continue
            
            bos_id = self.dictionary.bos()
            # prepare text and image tokens
            doc = [bos_id]
            image_tokens = []
            doc_input_mask = [0]
            doc_loss_mask = [1]
            chunk_tokens = [0]
            segment_tokens = [0]
            start_idx = 0
            is_valid = True

            # image_num = len(json_obj[IMAGE_KEY])
            # if image_num == 1:
            #     r = random.random()
            #     if r > 0.5:
            #         continue

            for image_idx, image_item in enumerate(json_obj[IMAGE_KEY]):
                text_snippet = json_obj[TEXT_KEY][start_idx:image_item['Span'][0]-1]
                if len(text_snippet) != 0:
                    if image_idx == 0:
                        # crop 3 sentences before the first image
                        try:
                            sentences = list(self.nlp_sentencizer(text_snippet).sents)
                        except:
                            is_valid = False
                            break
                        text_snippet = ' '.join([str(sent) for sent in sentences[-3:]])
                    text_token = self.text_transform(text_snippet)
                    doc.extend(text_token)
                    doc_input_mask.extend([0] * len(text_token))
                    doc_loss_mask.extend([1] * len(text_token))
                    chunk_id = chunk_tokens[-1]
                    chunk_tokens.extend([chunk_id] * len(text_token))
                    segment_tokens.extend([0] * len(text_token))

                # print(text_snippet)
                boi_id = self.dictionary.index(BOI_SYMBOL) 
                eoi_id = self.dictionary.index(EOI_SYMBOL)

                doc.extend([boi_id] * (self.image_token_length + 1) + [eoi_id])
                doc_input_mask.extend([0] + [1] * self.image_token_length + [0])
                doc_loss_mask.extend([0] + [0] * self.image_token_length + [1])
                chunk_id = chunk_tokens[-1] + 1
                chunk_tokens.extend([chunk_id] * (self.image_token_length + 2))
                segment_tokens.extend([1] * (self.image_token_length + 2))

                start_idx = image_item['Span'][1]
                try:
                    image_token = self._get_image_token(image_item['ImageSubpartitionFilename'], image_item['ByteOffset'])
                    image_tokens.append(image_token)
                except Exception as e:
                    is_valid = False
                    break
            if start_idx < len(json_obj[TEXT_KEY]):
                text_snippet = json_obj[TEXT_KEY][start_idx:]
                # crop 3 sentences before the first image
                text_snippet = self.clean(text_snippet)
                try:
                    sentences = list(self.nlp_sentencizer(text_snippet).sents)
                except:
                    continue
                text_snippet = ' '.join([str(sent) for sent in sentences[:3]])
                
                text_token = self.text_transform(text_snippet)
                doc.extend(text_token)
                doc_input_mask.extend([0] * len(text_token))
                doc_loss_mask.extend([1] * len(text_token))
                chunk_id = chunk_tokens[-1]
                chunk_tokens.extend([chunk_id] * len(text_token))
                segment_tokens.extend([0] * len(text_token))
            
            if not is_valid or len(doc) > self.tokens_per_sample:
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
