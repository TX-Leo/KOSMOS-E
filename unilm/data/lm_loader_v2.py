import json
import os
import multiprocessing
import numpy as np
import itertools

from infinibatch import iterators
from functools import partial
from fairseq.utils import safe_getattr
from unilm.data.lm_loader import LMLoader
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator


class MultiSourceLmLoader(LMLoader):

    def _tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize_foreach_lang(self, data):
        dataset = list(zip(data['source']))
        if self.shuffle:
            chunk_files = iterators.InfinitePermutationSourceIterator(
                dataset,
                seed=self.seed, 
                shuffle=self.shuffle, 
                num_instances=self.num_shards, 
                instance_rank=self.shard_id,)
        else:
            chunk_files = iterators.ChunkedSourceIterator(
                dataset,
                num_instances=self.num_shards, 
                instance_rank=self.shard_id,)
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines