import numpy as np

from fairseq.data.base_wrapper_dataset import BaseWrapperDataset, FairseqDataset
from fairseq.data import data_utils


class ToyFairseqDataset(FairseqDataset):

    @property
    def sizes(self):
        # WALKAROUND
        return [1] * len(self)

    def size(self, index):
        # WALKAROUND
        return 1
    
    def num_tokens(self, index):
        return 1

    def __len__(self):
        return self._len

    def collater(self, samples):
        raise NotImplementedError


class PairedDataset(ToyFairseqDataset):

    def __init__(self, src_dataset, trg_dataset):
        assert len(src_dataset) == len(trg_dataset)
        self._len = len(src_dataset)
        self.src = src_dataset
        self.trg = trg_dataset

    def __getitem__(self, index):
        return (self.src[index], self.trg[index])


class TupledDataset(ToyFairseqDataset):

    def __init__(self, *datasets):
        self._len = len(datasets[0])
        assert all(len(d) == self._len for d in datasets)
        self.datasets = datasets
        self.tuple_size = len(datasets)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)


class ShuffleDataset(BaseWrapperDataset):

    def __init__(self, dataset, seed, epoch=0):
        super().__init__(dataset)
        self.seed = seed
        self.indices_order = None
        self.set_epoch(epoch)
    
    def set_epoch(self, epoch, **kwargs):
        self.epoch = epoch
        with data_utils.numpy_seed(self.seed, self.epoch):
            self.indices_order = np.random.permutation(len(self.dataset))

    def __getitem__(self, index):
        idx = self.indices_order[index]
        return self.dataset[idx]


class RepeatDataset(ToyFairseqDataset):

    def __init__(self, item, size):
        self._len = size
        self.item = item
    
    def __getitem__(self, index):
        return self.item


# repeat more than one items
class RepeatItemsDataset(ToyFairseqDataset):

    def __init__(self, items, size, strict=True, repeat_locally=False):
        self._len = size
        self.items = items
        self._len_items = len(self.items)
        self.strict = strict
        # when strict is set True, the size must be a multiplier of len(items) 
        if strict:
            assert self._len % self._len_items == 0
        # e.g. items = [1, 2, 3], size = 6
        # repeat_locally is True -> 1, 1, 2, 2, 3, 3
        # repeat_locally is False -> 1, 2, 3, 1, 2, 3
        self.repeat_locally = repeat_locally
        if repeat_locally:
            assert self.strict
            self._n_repeat = self._len // self._len_items
    
    def __getitem__(self, index):
        if self.repeat_locally:
            return self.items[index // self._n_repeat]
        return self.items[index % self._len_items]
    

class RepeatTupledDataset(RepeatItemsDataset):

    def __init__(self, tup_ds: TupledDataset, size, strict=True, repeat_locally=False):
        super().__init__(tup_ds, size, strict=strict, repeat_locally=repeat_locally)
        self.tuple_size = tup_ds.tuple_size


class DictDataset(ToyFairseqDataset):

    def __init__(self, defn, sizes=None):
        self.defn = defn
        for v in self.defn.values():
            if not isinstance(v, (FairseqDataset, )):
                raise ValueError('Expected Dataset but found: {}'.format(v.__class__))
    
    def set_epoch(self, epoch):
        for ds in self.defn.values():
            ds.set_epoch(epoch)
    
    def __getitem__(self, index):
        ret = {key:self.defn[key][index] for key in self.defn}
        return ret
    
    def __len__(self):
        return min(len(v) for v in self.defn.values())

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        sample = {}
        for k, ds in self.defn.items():
            sample[k] = ds.collater([s[k] for s in samples])

        # DEBUG
        # print(sample)
        return sample


class ShardedDataset(ToyFairseqDataset):

    def __init__(self, dataset, num_shards=1, shard_id=0):
        assert len(dataset) > num_shards
        self.dataset = dataset
        self._len = len(dataset) // num_shards
        self.num_shards = num_shards
        self.shard_id = shard_id

    def __getitem__(self, index):
        index = index * self.num_shards + self.shard_id
        return self.dataset[index]
    
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch