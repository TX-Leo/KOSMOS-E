import math
import webdataset as wds
import os

from argparse import Namespace
from unilm.data.vl.clip_dl import (
    DataInfo,
    detshuffle2,
    filter_no_caption,
    get_dataset_size,
    log_and_continue,
    ResampledShards2,
    _SAMPLE_SHUFFLE_SIZE,
    _SAMPLE_SHUFFLE_INITIAL,
    _SHARD_SHUFFLE_SIZE,
    _SHARD_SHUFFLE_INITIAL,
    SharedEpoch, 
    tarfile_to_samples_nothrow, )


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess_txt(text):
    return str(text).strip()


def filter_no_caption(sample):
        return 'txt' in sample


_datashards = {
        "vg": "vg-train-{000000..000540}.tar",
        "coco": "coco-train-{000000..000056}.tar",
        # "coco": "coco-train-{000000..000015}.tar",
        "gcc": "gcc-train-{000000..000304}.tar",
        "sbu": "sbu-train-{000000..000086}.tar",
        # "10m": "10m-train-{000000..000988}.tar",
        "10m": "10m-train-{000000..000063}.tar",
        "10m1k": "10m1k-train-{000000..009882}.tar",
        "20m1k": "20m1k-train-{000000..021234}.tar",
        "laion": "{00000..20000}.tar",
}


_datasize = {
        "vg": 5408689,
        "coco": 566747,
        # "coco": 160000,
        "gcc": 3046646,
        "sbu": 860471,
        # "10m": 9882553,
        "10m": 640000,
        "10m1k": 9882553,
        "20m1k": 21235000,
        "laion": 200010000,
}


def get_wds_dataset_20m(args, preprocess_img, is_train, epoch=0, floor=False, shard_id=0, num_shards=1, max_sentences=None):
    wds_args = Namespace(
        # train_data=args.wds_train_data,
        # val_data=args.wds_val_data,
        # dataset_resampled=args.wds_dataset_resampled,
        # train_num_samples=args.wds_train_num_samples,
        # val_num_samples=args.wds_val_num_samples,
        seed=args.seed,
        batch_size=max_sentences,
        workers=1,
        world_size=num_shards, )
    shard_key = "20m1k"
    input_shards = os.path.join(args.wds_train_data, _datashards[shard_key])
    num_input_samples = _datasize[shard_key]
    num_input_shards = 21235
    return _get_wds_dataset_20m(wds_args, input_shards, num_input_samples, num_input_shards, preprocess_img, is_train, epoch=epoch, floor=floor)


def _get_wds_dataset_20m(wds_args, input_shards, num_samples, num_shards, preprocess_img, is_train, epoch=0, floor=False):
    # NOTE debug
    # from webdataset.utils import pytorch_worker_info
    # print("DEBUG!")
    # print(pytorch_worker_info())

    args = wds_args

    shared_epoch = SharedEpoch(epoch=epoch)
    pipeline = [wds.SimpleShardList(input_shards)]

    if is_train:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=args.seed,
                epoch=shared_epoch, ),
            wds.split_by_node,
            wds.split_by_worker,])
        pipeline.extend([
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL, ), ])
        # pipeline.extend([
        #     tarfile_to_samples_nothrow,])
    else:
        # pipeline.extend([wds.split_by_worker])
        raise NotImplementedError
    pipeline.extend([
        # wds.decode("pil"),
        wds.select(filter_no_caption),
        wds.decode("pil", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg", text="txt"),
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),])
    
    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)    # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)    # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)