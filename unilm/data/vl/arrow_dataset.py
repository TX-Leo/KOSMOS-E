import random
import torch
import io
try:
    import pyarrow as pa
except:
    pass
import os
import numpy as np
import re
import logging

from PIL import Image
from torch.utils.data.dataset import ConcatDataset, Dataset


logger = logging.getLogger(__name__)


class ArrowDataset(Dataset):

    def __init__(self, data_dir, ds_names=None, transform=None, text_column_name=""):
        super().__init__()
        self.data_dir = data_dir
        self.ds_names = ds_names
        self.transform = transform
        self.text_column_name = text_column_name

        self.all_texts = []
        if len(ds_names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/{name}.arrow", "r")).read_all()
                for name in ds_names if os.path.isfile(f"{data_dir}/{name}.arrow")]
            self.table_names = []
            for i, name in enumerate(ds_names):
                self.table_names += [name] * len(tables[i])
            
            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = [list(set(texts)) for texts in self.all_texts]
        
        self.index_mapper = {}
        if text_column_name != "":
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)
        
        self.ending_punc = set(['.','?','!',',','"',"'",';'])
        
    def __len__(self): return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")
    
    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index)
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_raw_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]
        try:
            text = text.strip()
            if len(text) == 0: text = ""
            else:
                if text[-1] not in self.ending_punc: text = text + "."
        except Exception as e:
            logger.info(f"Error at ArrowDataset.get_raw_text idx {index} text {text} exception: {e}")
            text = ""
        return text
    
    def __getitem__(self, index):
        image = self.get_image(index)
        text = self.get_raw_text(index)
        return image, text
    

class CocoCaptionDataset(ArrowDataset):
    def __init__(self, data_dir, transform=None, split=""):
        assert split in ["train", "val", "test"]
        self.split = split
        if split == "train":
            ds_names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            ds_names = ["coco_caption_karpathy_val"]
        elif split == "test":
            ds_names = ["coco_caption_karpathy_test"]
        super().__init__(data_dir, ds_names=ds_names, transform=transform, text_column_name="caption")


class SbuCaptionDataset(ArrowDataset):
    def __init__(self, data_dir, transform=None, split=""):
        assert split in ["train", "val", "test"]
        self.split = split
        names = [f"sbu_{i}" for i in range(8)] if split == "train" else ["sbu_8"]
        super().__init__(data_dir, ds_names=names, transform=transform, text_column_name="caption")


class ConceptualCaptionDataset(ArrowDataset):
    def __init__(self, data_dir, transform=None, split=""):
        assert split in ["train", "val", "test"]
        self.split = split
        names = [f"conceptual_caption_train_{i}" for i in range(30)] if split == "train" else ["conceptual_caption_val_0"]
        super().__init__(data_dir, ds_names=names, transform=transform, text_column_name="caption")


class VisualGenomeCaptionDataset(ArrowDataset):
    def __init__(self, data_dir, transform=None, split=""):
        assert split in ["train", "val", "test"]
        names = ["vg"]
        super().__init__(data_dir, ds_names=names, transform=transform, text_column_name="caption")


_VL20M_DATASET_CLS = [CocoCaptionDataset, SbuCaptionDataset, ConceptualCaptionDataset, VisualGenomeCaptionDataset]
# # NOTE DEBUG TODO remove
# _VL20M_DATASET_CLS = [CocoCaptionDataset]


def load_vl_20m_dataset(data_dir, transform=None, split="train"):
    datasets = [ds_cls(data_dir, transform=transform, split=split) for ds_cls in _VL20M_DATASET_CLS]
    dataset = ConcatDataset(datasets)
    return dataset
