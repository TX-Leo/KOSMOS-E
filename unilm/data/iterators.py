import multiprocessing

from infinibatch import iterators
from infinibatch.iterators import MapIterator, MultiplexIterator
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator


class FilterIterator(MapIterator):

    def __next__(self):
        should_skip = True
        next_item = None
        while should_skip:
            next_item = next(self._source_iterator)
            should_skip = self._transform(next_item)
        return next_item


class RepeatIndexIterator(object):

    def __init__(self, n_index):
        self.n_index = n_index
        self._cur_index = 0
        self.setstate(None)
            
    def __iter__(self):
        return self
    
    def getstate(self):
        return {"_cur_index": self._cur_index}

    def setstate(self, checkpoint):
        if checkpoint and "_cur_index" in checkpoint:
            self._cur_index = checkpoint["_cur_index"]
        else:
            self._cur_index = 0
            
    def __next__(self):
        ret = self._cur_index
        self._cur_index = (self._cur_index + 1) % self.n_index
        return ret
    
    def close(self):
            pass


def _get_parallel_loader(loader_cls, *loader_args, seed=1, n_loader=1, **loader_kwargs):
    if n_loader == 1:
        return loader_cls(*loader_args, seed=seed, **loader_kwargs)
    loaders = [loader_cls(*loader_args, seed=seed + loader_id, **loader_kwargs) for loader_id in range(n_loader)]

    control_iterator = NativeCheckpointableIterator(RepeatIndexIterator(n_loader))
    ret = MultiplexIterator(control_iterator, loaders)
    return ret


class ParallelLoader(BaseBatchGen):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._iter = _get_parallel_loader(*args, **kwargs)


class NativeCheckpointableEpochStateIterator(iterators.CheckpointableIterator):

    def __init__(self, iterable, set_epoch_fn=None):
        self._input_iterable = iterable
        self.set_epoch_fn = set_epoch_fn
        self.setstate(None)

    def getstate(self):
        return {'num_items_yielded': self._num_items_yielded, "itr_cur_epoch": self._cur_epoch}
    
    def setstate(self, checkpoint):
        self._iterator = iter(self._input_iterable)
        self._cur_epoch = checkpoint["itr_cur_epoch"] if checkpoint is not None else 0
        self.set_epoch_fn(self._cur_epoch)
        self._num_items_yielded = iterators._advance_iterator(self._iterator, checkpoint['num_items_yielded']) if checkpoint is not None else 0
    
    def __next__(self):
        try:
            item = next(self._iterator)
            self._num_items_yielded += 1
        except StopIteration:
            self._iterator = iter(self._input_iterable)
            item = next(self._iterator)
            self._num_items_yielded = 1
            self._cur_epoch += 1
        return item
    
    def close(self): pass


class InfinitePermutationSourceIteratorWithoutDeepCopy(iterators.InfinitePermutationSourceIterator):

    def __init__(self, source_items, seed: int = 0, shuffle: bool = True, num_instances: int = 1, instance_rank: int = 0,):
        if not source_items:
            raise ValueError("source must not be empty")
        if instance_rank >= num_instances:
            raise ValueError("invalid instance_rank")
        self._source_items = source_items
        self._shuffle = shuffle
        self._seed = seed
        self._num_instances = num_instances
        self._instance_rank = instance_rank
        self.setstate(None)