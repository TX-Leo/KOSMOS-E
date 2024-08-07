import os
import json
import gzip
import io
import torch
import numpy as np
import logging
import binascii

from argparse import Namespace
from PIL import Image
from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.vl.clip_transform import image_transform
from unilm.data.vl.vlm_loader import fs_encode_line


logger = logging.getLogger(__name__)


_src_mask_cache = {
    "n_images": None,
    "img_len": None,
    "cache": {},}

def get_src_mask(n_images, img_len, img_id, device):
    """ To select which image to see. The last vector is a padding vector.
    Args:
        img_id: The img index to see, ranging from -1 to n_images-1,
                        -1 means only see the padding vector
    """
    cache = _src_mask_cache
    def _check_or_set(key, value):
        if cache[key] is not None:
            assert cache[key] == value, (cache[key], value)
        else:
            cache[key] = value
    _check_or_set("n_images", n_images)
    _check_or_set("img_len", img_len)
    assert img_id >= -1 and img_id < n_images
    if img_id in cache["cache"]:
        return cache["cache"][img_id]

    mask_len = n_images * img_len + 1
    # a True value indicates that the corresponding position is not allowed to attend
    # x = tensor4new.new(1, mask_len).fill_(True)
    x = torch.BoolTensor(1, mask_len, device=device)
    # padding should be always available
    x[0, -1] = False

    if n_images >= 0:
        offset = n_images * img_len
        x[offset: offset + img_len]= False

    cache["cache"][img_id] = x
    return x


def runtime_xattn_mask(xattn_args, n_heads, device):
    # NOTE https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    # xattn shape should be: N*n_heads, trg_len, src_len
    n_images = xattn_args.n_images
    img_len = xattn_args.per_img_len
    # ending positions for each text span
    all_end_positions = xattn_args.all_end_positions
    bsz = len(all_end_positions)
    src_mask_len = n_images * img_len + 1
    # assert all(len(span) == n_images + 1 for span in cut_spans), cut_spans

    max_trg_len = -1
    for end_positions in all_end_positions:
        # assert len(end_positions) == n_images + 1
        max_trg_len = max(max_trg_len, end_positions[-1])
    
    # each item should be shaped as 1, 1, trg_len, src_len
    all_mask = []
    for end_positions in all_end_positions:
        mask = []
        begin_pos = 0
        for img_id, end_pos in enumerate(end_positions + [max_trg_len]):
            if end_pos < begin_pos:
                raise ValueError
            elif end_pos > begin_pos:
                # TODO check
                _img_id = img_id % (n_images + 1) - 1
                src_mask = get_src_mask(n_images, img_len, _img_id, device)
                src_mask = src_mask.expand(end_pos - begin_pos, -1)
                mask.append(src_mask)
            begin_pos = end_pos
        mask = torch.cat(mask, dim=0)
        assert mask.size() == (max_trg_len, src_mask_len), (mask.size(), max_trg_len, src_mask_len)
        all_mask.append(mask[None, None, :, :])

    all_mask = torch.cat(all_mask, dim=0)
    all_mask = all_mask.expand(-1, n_heads, -1, -1)
    all_mask = all_mask.reshape(bsz * n_heads, max_trg_len, src_mask_len)
    return all_mask


class ItlvLoader(BaseBatchGen):

    def __init__(self, args, dataset, dictionary, tokenizer, max_tokens=None, max_sentences=None, max_positions=None, ignore_invalid_inputs=False, required_batch_size_multiple=1, seed=1, epoch=1, num_shards=1, shard_id=0, nested_batch_key=None, image_size=None, image_mean=None, image_std=None, ):

        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.dataset = dataset
        self.shuffle = dataset.shuffle
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample4vlm

        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = seed
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        self.nested_batch_key = nested_batch_key

        self.init_preprocess_img(image_size, mean=image_mean, std=image_std)
        self.images_per_sample = args.images_per_sample
        self.tokens_per_image = args.tokens_per_image
        self.decoder_attention_heads = args.decoder_attention_heads

        self._build_iter()
    
    def init_preprocess_img(self, image_size, mean=None, std=None):
        self._img_prep_img_sz = image_size
        self._img_prep_mean = mean
        self._img_prep_std = std
        self.img_prep_func = image_transform(image_size, is_train=False, mean=mean, std=std)
        pad_image = Image.fromarray(np.random.randint(255, size=(224, 224), dtype=np.uint8))
        self.pad_image = self.img_prep_func(pad_image)
    
    def _build_iter(self):
        samples = self._tokenize()
        self.padded_batches = self._batchify(samples)
        # prefetch_batches = iterators.PrefetchIterator(
        #     self.padded_batches, 
        #     buffer_size=100, 
        #     buffer_in_main_process=True, 
        #     log_empty_buffer_warning=True and self.shard_id == 0, )
        prefetch_batches = self.padded_batches
        prefetch_batches = iterators.MapIterator(prefetch_batches, self._move_to_tensor)
        self._iter = prefetch_batches
    
    def _tokenize(self):
        if self.shuffle:
            chunk_files = iterators.InfinitePermutationSourceIterator(
                self.data,
                seed=self.seed, 
                shuffle=self.shuffle, 
                num_instances=self.num_shards, 
                instance_rank=self.shard_id, )
        else:
            chunk_files = iterators.ChunkedSourceIterator(
                self.data,
                num_instances=self.num_shards, 
                instance_rank=self.shard_id, )
        # itlv_jsons = iterators.SelectManyIterator(chunk_files, lambda files: self._read_and_filter(*files))
        itlv_jsons = iterators.SelectManyIterator(chunk_files, self._read_and_filter)
        itlv_items = iterators.SamplingRandomMapIterator(itlv_jsons, self._prepare, self.seed)
        return itlv_items
    
    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _prepare(self, _random, itlv_jsn):
        tokenized_spans, images = self._split_and_tokenize(_random, itlv_jsn)
        tokens_itlv, img_indices = self._itlv_cut(_random, tokenized_spans)
        images = self._prepare_images(images, img_indices)
        tokens, end_positons = self._prepare_text(tokens_itlv)
        return images, tokens, end_positons
    
    def _split_and_tokenize(self, _random, itlv_jsn):
        doc, images = itlv_jsn["doc"], itlv_jsn["images"]
        text_spans = []
        cur_pos = 0
        for image_jsn in images:
            fr, to = image_jsn["Span"]
            # NOTE data example: xxxx xxx.\n![](xxx.com)\n
            # the span only labels [](xxx.cam)
            # so we extend the range of the span
            if fr > 0 and doc[fr - 1] == "!": fr -= 1
            text_spans.append(doc[cur_pos: fr].strip())
            cur_pos = to
        text_spans.append(doc[cur_pos:].strip())

        new_text_spans = [text_spans[0]]
        filtered_images = []
        for i, image_jsn in enumerate(images):
            if image_jsn["ImageSubpartitionFilename"] is None:
                new_text_spans[-1] += text_spans[i+1]
            else:
                new_text_spans.append(text_spans[i+1])
                filtered_images.append(image_jsn)
        text_spans = new_text_spans
        images = filtered_images
        assert len(new_text_spans) == len(filtered_images) + 1

        # NOTE image placement augmentation
        if len(text_spans) > 1 and _random.random() < 0.5:
            # moving the images forward for 1 span
            text_spans = [""] + text_spans
            text_spans[-2] += "\n" + text_spans[-1]
            text_spans.pop()

        # tokenize text spans
        # spans[span[line[token]]]
        # eg [[[1,2,3], [4,5,6]], [[1,2,3,4]]]
        tokenized_spans = []
        for i, text_span in enumerate(text_spans):
            text_span = text_span.strip()
            if text_span == "":
                tokenized_spans.append([])
                continue
            # prob=0.5 for prepend a space
            if _random.random() < 0.5:
                text_span = " " + text_span
            if i > 0:
                text_span = "<image>" + text_span
            tokenized_lines = []
            for line in text_span.split("\n"):
                line = line.strip()
                if line == "": continue
                tokens = self.tokenizer.encode(line)
                text_ids = fs_encode_line(self.dictionary, tokens.split(), append_eos=True)
                tokenized_lines.append(text_ids)
            tokenized_spans.append(tokenized_lines)
        
        assert len(tokenized_spans) == len(images) + 1
        return tokenized_spans, images

    def _prepare_images(self, images, img_indices):
        # TODO add try cache
        ret_images = []
        for i in img_indices:
            img_jsn = images[i]
            try:
                partition = img_jsn["ImageSubpartitionFilename"].split("_")[0]
            except Exception as e:
                logger.info("Image is not provided!")
                # TODO remove the image when it is not provided
                # raise e
                ret_images.append(None)
                continue
            file_path = os.path.join(
                self.data_dir, "downloaded_images", partition, "resized", img_jsn["ImageSubpartitionFilename"])
            offset = img_jsn["ByteOffset"]
            with open(file_path) as fp:
                fp.seek(offset)
                item = fp.readline().strip().split("\t")
            try:
                img = Image.open(io.BytesIO(binascii.a2b_base64(item[5]))).convert("RGB")
            except Exception as e:
                logger.info(str(e))
                ret_images.append(None)
                continue
            img = self.img_prep_func(img)
            ret_images.append(img)
        return ret_images
    
    def _prepare_text(self, tokens_itlv):
        # print("tokens_itlv=")
        # print(tokens_itlv)
        tokens = []
        end_positions = []
        for t in tokens_itlv:
            tokens.extend(t)
            end_positions.append(len(tokens) - 1)
            if len(tokens) > self.tokens_per_sample:
                tokens = tokens[:self.tokens_per_sample]
                end_positions[-1] = self.tokens_per_sample - 1
                break
            elif len(tokens) == self.tokens_per_sample:
                break
        # print("At _prepare_text tokens=")
        # print(tokens)
        # print("end_positions=")
        # print(end_positions)
        return tokens, end_positions
    
    def _itlv_cut(self, _random, tokenized_spans):
        end_pos_candidates = []
        the_last_position = (0, 0)
        tot_tokens = 0
        for sid, span in enumerate(tokenized_spans):
            if sid == 0: continue
            accumulated_distance = 0
            length_exceed = False
            for lid, line in enumerate(span):
                accumulated_distance += len(line)
                tot_tokens += len(line)
                the_last_position = (sid, lid)
                # NOTE when the accumulated length is larger then tokens_per_sample,
                # it is also included, we can use it by truncating its tail
                # -1 for bos
                if accumulated_distance >= self.tokens_per_sample - 1:
                    if not length_exceed:
                        end_pos_candidates.append((sid, lid))
                        length_exceed = True
                else:
                    if tot_tokens >= self.tokens_per_sample - 1:
                        end_pos_candidates.append((sid, lid))
                    
        if len(end_pos_candidates) == 0:
            end_pos_candidates.append(the_last_position)
        
        end_pos = _random.sample(end_pos_candidates, k=1)[0]
        # get token list interleaved with images by concating lines to a single line
        tokens_itlv = []
        img_indices = []
        end_sid, end_lid = end_pos
        tot_tokens = 0
        for sid in range(end_sid, -1, -1):
            tokens = []
            span = tokenized_spans[sid]
            _end_lid = end_lid if sid == end_sid else len(span) - 1
            for lid in range(_end_lid, -1, -1):
                tokens = span[lid] + tokens
                tot_tokens += len(span[lid])
                if len(img_indices) > 0 and tot_tokens >= self.tokens_per_sample - 1:
                    # should attend to last image
                    if lid == 0 and sid > 0:
                        img_indices.insert(0, sid - 1)
                    tokens_itlv.insert(0, tokens)
                    break                    
            # spans with sid=1 should attend to images with id=0
            if tot_tokens >= self.tokens_per_sample - 1:
                if len(img_indices) == 0:
                    tokens_itlv.insert(0, tokens)
                    img_indices.insert(0, sid - 1)
                break
            tokens_itlv.insert(0, tokens)
            img_indices.insert(0, sid - 1)
        
        if len(img_indices) == len(tokens_itlv):
            tokens_itlv.insert(0, [self.dictionary.bos()])
        elif len(img_indices) == len(tokens_itlv) - 1:
            tokens_itlv[0].insert(0, self.dictionary.bos())
        else:
            raise ValueError
        
        # remove the images when n > self.images_per_sample
        if len(img_indices) > self.images_per_sample:
            img_indices = img_indices[:self.images_per_sample]
            for i in range(self.images_per_sample + 1, len(tokens_itlv)):
                tokens_itlv[self.images_per_sample].extend(tokens_itlv[i])
            tokens_itlv = tokens_itlv[:self.images_per_sample + 1]

        assert len(tokens_itlv) == len(img_indices) + 1
        return tokens_itlv, img_indices
    
    def _read_and_filter(self, filename):
        data = []
        file_path = os.path.join(self.data_dir, "content_with_img_tags", filename)

        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        try:
            # with open(file_path, 'r', encoding='utf8') as f:
            #     lines = f.read().strip().split('\n')
            with gzip.open(file_path, "rb") as fp1:
                with io.TextIOWrapper(fp1, encoding="utf-8") as fp2:
                    for line in fp2:
                        try:
                            jsn = json.loads(line.strip())
                            data.append(jsn)
                        except:
                            pass

        except Exception as e:
            print("Exception at _read_from_files: %s" % str(e))
            return iter([]) # skip bad file
        
        # TODO maybe filter data

        itlv_jsons = []
        for jsn in data:
            images = jsn["Images"]
            have_images = False
            for img in images:
                if img["ImageSubpartitionFilename"] is None: continue
                have_images = True
                break
            if not have_images: continue
            sorted_images = sorted(images, key=lambda x: x["Span"])
            itlv_jsn = {"id": jsn["UUID"], "doc": jsn["Extracted"], "images": sorted_images}
            itlv_jsons.append(itlv_jsn)
        return itlv_jsons
    
    def _batchify(self, lines):
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = iterators.BlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else: raise NotImplementedError

        def collate(batch):
            batch_size = len(batch)
            all_end_positions = []
            images = []
            pad_id = self.dictionary.pad()
            gpt_max_length = max(len(x[1]) for x in batch)
            gpt_source_ids = np.full(
                shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=pad_id)
            gpt_target_ids = np.full(
                shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=pad_id)
            for i, (line_images, gpt_ids, end_positions) in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
                gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
                all_end_positions.append(end_positions)
                _images = line_images + (self.images_per_sample - len(line_images)) * [self.pad_image]
                _images_no_None = []
                for img in _images:
                    if img is None:
                        _images_no_None.append(self.pad_image)
                        print("[W] Image with val=None", flush=True)
                    else:
                        _images_no_None.append(img)
                images.extend(_images_no_None)
            
            img_batch = torch.stack(images, dim=0)
            # print("[DEBUG] img_batch.size()=")
            # print(img_batch.size())
            # TODO add decoder_attention_heads to task args
            xattn_args = Namespace(n_images=self.images_per_sample, per_img_len=self.tokens_per_image, all_end_positions=all_end_positions)
            xattn_mask = runtime_xattn_mask(xattn_args, self.decoder_attention_heads, img_batch.device)

            ret_batch = {
            "net_input": {
                "src_tokens": gpt_source_ids.astype(np.int64),
                "image": img_batch,
                "xattn_mask": xattn_mask,
                "is_itlv": True,
            },
            "target": gpt_target_ids.astype(np.int64),
            "nsentences": batch_size,
            "ntokens": sum([len(x[1]) for x in batch]),}

            if self.nested_batch_key is not None:
                ret_batch = {self.nested_batch_key: ret_batch}
            return ret_batch

        padded_batches = iterators.MapIterator(batches, collate)
        return padded_batches
