import torch
import numpy as np
import logging

from torch.utils.data import DataLoader, DistributedSampler
from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.iterators import InfinitePermutationSourceIteratorWithoutDeepCopy, NativeCheckpointableEpochStateIterator
from unilm.data.common_dataset import ShuffleDataset, ShardedDataset
from unilm.data.vl.arrow_dataset import load_vl_20m_dataset
from unilm.data.vl.clip_transform import image_transform
from unilm.data.vl.vl_loader import get_wds_dataset
from unilm.data.vl.wds import get_wds_dataset_20m
from fairseq.data.encoders.gpt2_bpe import GPT2BPE


logger = logging.getLogger(__name__)


def fs_encode_line(fs_dict, words, append_eos=True):
    assert not isinstance(words, str)
    ids = []
    for i, word in enumerate(words):
        idx = fs_dict.index(word)
        ids.append(idx)
    if append_eos:
        ids.append(fs_dict.eos_index)
    return ids


class VlmLoader(BaseBatchGen):

    def __init__(
            self,
            args,
            dataset,
            dictionary,
            tokenizer,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            epoch=1,
            num_shards=1,
            shard_id=0,
            nested_batch_key=None,
            image_size=None,
            image_mean=None,
            image_std=None):

        super().__init__()
        self.args = args
        # self.data = dataset.data
        # self.data_dir = dataset.data_dir
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
        self._build_iter()

        self.ending_punc = set(['.', '?', '!', ',', '"', "'", ';'])

    def init_preprocess_img(self, image_size, mean=None, std=None):
        self._img_prep_img_sz = image_size
        self._img_prep_mean = mean
        self._img_prep_std = std
        # self.img_prep_train = image_transform(image_size, is_train=True, mean=mean, std=std)
        # self.img_prep_val = image_transform(image_size, is_train=False, mean=mean, std=std)
        self.img_prep_func = image_transform(
            image_size, is_train=False, mean=mean, std=std)

    def _build_iter(self):
        self.padded_batches = self._tokenize()
        prefetch_batches = iterators.PrefetchIterator(
            self.padded_batches,
            buffer_size=10,
            buffer_in_main_process=True,
            log_empty_buffer_warning=True and self.shard_id == 0, )
        prefetch_batches = iterators.MapIterator(
            prefetch_batches, self._move_to_tensor)
        self._iter = prefetch_batches

    def _tokenize(self):
        # training:
        is_train = True
        img_prep = self.img_prep_func
        # validation
        # is_train = False
        # img_prep = self.img_prep_val

        assert self.max_sentences is not None
        wds_func = get_wds_dataset_20m if self.args.wds_format == "20m1k" else get_wds_dataset
        vl_dl = wds_func(self.args, img_prep, is_train, epoch=0, shard_id=self.shard_id,
                         num_shards=self.num_shards, max_sentences=self.max_sentences)

        def set_epoch_fn(epoch): vl_dl.set_epoch(epoch)
        batched_vl_pairs = NativeCheckpointableEpochStateIterator(
            vl_dl.dataloader, set_epoch_fn=set_epoch_fn)
        tokenized_vl_pairs = iterators.MapIterator(
            batched_vl_pairs, self._prepare)
        return tokenized_vl_pairs

    def _prepare(self, batched_vl_pair):
        # `<s> <image> image hidden </image> My cat looking very dignified.</s>`

        image_feature_length = self.args.image_feature_length
        bos_id = self.dictionary.bos()
        eos_id = self.dictionary.eos()
        boi_id = self.dictionary.index("<image>")
        eoi_id = self.dictionary.index("</image>")
        pad_id = self.dictionary.pad()

        img_batch, raw_text_batch = batched_vl_pair

        gpt_max_length = -1
        tokenized_ids = []
        gpt_input_masks = []
        gpt_loss_masks = []
        for text in raw_text_batch:
            text = text.strip()
            if len(text) == 0:
                text = ""
            else:
                if text[-1] not in self.ending_punc:
                    text = text + "."

            if isinstance(self.tokenizer, GPT2BPE):
                tokens = self.tokenizer.encode(text.strip()).split(' ')
            else:
                tokens = self.tokenizer.encode(text.strip(), out_type=str)
            text_ids = fs_encode_line(
                self.dictionary, tokens, append_eos=False)
            text_ids = [bos_id, boi_id] + [boi_id] * \
                image_feature_length + [eoi_id] + text_ids

            if len(text_ids) > self.tokens_per_sample - 2:
                text_ids = text_ids[:self.tokens_per_sample - 2]
            text_ids = text_ids + [eos_id]
            gpt_max_length = max(gpt_max_length, len(text_ids))

            tokenized_ids.append(text_ids)

            gpt_input_mask = [0] * 2 + [1] * image_feature_length + [0] * (len(text_ids) - 2 - image_feature_length)
            gpt_input_masks.append(gpt_input_mask)

            gpt_loss_mask = [0] * (2 + image_feature_length) + [1] * (len(text_ids) - 2 - image_feature_length)
            gpt_loss_masks.append(gpt_loss_mask)

            assert len(text_ids) == len(gpt_input_mask)
            assert len(text_ids) == len(gpt_loss_mask)

            # print(text_ids)

        # collate
        batch_size = len(raw_text_batch)
        gpt_source_ids = np.full(
            shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=self.dictionary.pad())
        gpt_target_ids = np.full(
            shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=self.dictionary.pad())
        gpt_input_mask_all = np.full(
            shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
        gpt_loss_mask_all = np.full(
            shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)

        for i, gpt_ids in enumerate(tokenized_ids):
            gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
            gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
            gpt_input_mask_all[i, :len(gpt_ids)-1] = gpt_input_masks[i][:-1]
            gpt_loss_mask_all[i, :len(gpt_ids)-1] = gpt_loss_masks[i][:-1]

        ret_batch = {
            "image": {
                "net_input": {
                    "src_tokens": gpt_source_ids.astype(np.int64),
                    "img_src_tokens": img_batch,
                    'img_gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
                    'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
                },
                "target": gpt_target_ids.astype(np.int64),
                "nsentences": batch_size,
                "ntokens": sum([len(x) for x in tokenized_ids]),
            }
        }

        if self.nested_batch_key is not None:
            ret_batch = {self.nested_batch_key: ret_batch}

        return ret_batch


class VlmLoader4ArrowDataset(VlmLoader):

    def _build_iter(self):
        tokenized_lines = self._tokenize()
        self.padded_batches = self._batchify(tokenized_lines)
        # TODO fix hanging bug
        # prefetch_batches = iterators.PrefetchIterator(
        #     self.padded_batches,
        #     buffer_size=10,
        #     buffer_in_main_process=True,
        #     log_empty_buffer_warning=True and self.shard_id == 0, )

        prefetch_batches = self.padded_batches

        prefetch_batches = iterators.MapIterator(
            prefetch_batches, self._move_to_tensor)
        self._iter = prefetch_batches

    def _tokenize(self):
        data_split = self.dataset.split
        img_prep = self.img_prep_train
        logger.info("loading vl_20m_dataset...")
        dataset = load_vl_20m_dataset(
            self.dataset.data_dir, transform=img_prep, split=data_split)

        # data_iterable = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)
        logger.info("vl_20m_dataset loaded")
        if data_split == "train":
            assert self.shuffle is True, self.shuffle
            # sharded_dataset = InfinitePermutationSourceIteratorWithoutDeepCopy(
            #     dataset,
            #     seed=self.seed,
            #     shuffle=self.shuffle,
            #     num_instances=self.num_shards,
            #     instance_rank=self.shard_id,)
            # shuffled_dataset = ShuffleDataset(dataset, self.seed, epoch=0)
            # sharded_dataset = ShardedDataset(shuffled_dataset, num_shards=self.num_shards, shard_id=self.shard_id)
            # def set_epoch_fn(epoch): sharded_dataset.set_epoch(epoch)
            sampler = DistributedSampler(dataset, num_replicas=self.num_shards,
                                         rank=self.shard_id, shuffle=True, seed=self.seed, drop_last=True)
            data_iterable = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                       persistent_workers=True, sampler=sampler, prefetch_factor=10)

            def set_epoch_fn(epoch): sampler.set_epoch(epoch)
            sharded_iter = NativeCheckpointableEpochStateIterator(
                data_iterable, set_epoch_fn=set_epoch_fn)

        elif data_split == "val" or data_split == "test":
            # sharded_dataset = iterators.ChunkedSourceIterator(
            #     dataset,
            #     num_instances=self.num_shards,
            #     instance_rank=self.shard_id,)
            raise NotImplementedError
        else:
            raise ValueError
        tokenized_sharded_vl_pairs = iterators.MapIterator(
            sharded_iter, self._prepare)
        return tokenized_sharded_vl_pairs

    def _prepare(self, vl_pair):
        bos_id = self.dictionary.bos()
        eos_id = self.dictionary.eos()
        image, text = vl_pair
        if isinstance(text, str):
            pass
        elif isinstance(text, tuple) and len(text) == 1:
            text = text[0]
        else:
            print("text: %s" % text, flush=True)
            raise ValueError
        tokens = self.tokenizer.encode("<image>" + text.strip())
        text_ids = fs_encode_line(
            self.dictionary, tokens.split(), append_eos=False)
        # TODO maybe replace eos,eos with eoc,eos
        text_ids = [bos_id] + text_ids
        if len(text_ids) > self.tokens_per_sample - 2:
            text_ids = text_ids[:self.tokens_per_sample - 2]
        text_ids = text_ids + [eos_id, eos_id]
        return image, text_ids

    def _collate_fn(self, batch):
        batch_size = len(batch)
        gpt_max_length = max([len(x[1]) for x in batch])

        gpt_source_ids = np.full(
            shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=self.dictionary.pad())
        gpt_target_ids = np.full(
            shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=self.dictionary.pad())
        images = []
        for i, (image, gpt_ids) in enumerate(batch):
            gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
            gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
            images.append(image)
        img_batch = torch.cat(images, dim=0)

        ret_batch = {
            "net_input": {
                "src_tokens": gpt_source_ids.astype(np.int64),
                "image": img_batch,
            },
            "target": gpt_target_ids.astype(np.int64),
            "nsentences": batch_size,
            "ntokens": sum([len(x[1]) for x in batch]), }
        if self.nested_batch_key is not None:
            ret_batch = {self.nested_batch_key: ret_batch}
        return ret_batch

    def _batchify(self, lines):
        if self.max_sentences is not None:
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            raise NotImplementedError
        padded_batches = iterators.MapIterator(batches, self._collate_fn)
        return padded_batches
