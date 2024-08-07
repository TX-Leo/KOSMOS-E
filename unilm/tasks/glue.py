import logging
import os

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from argparse import Namespace

from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionConfig, SentencePredictionTask
from fairseq.data import Dictionary, BaseWrapperDataset
from omegaconf import II

from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.data.shorten_dataset import maybe_shorten_dataset

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    BaseWrapperDataset,
    IdDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumelDataset,
    NestedDictionaryDataset,
    SortDataset,
    NumelDataset,
    RightPadDataset,
    RawLabelDataset,
    FairseqDataset,
    PrependTokenDataset,
    ConcatSentencesDataset,
)
from unilm.data.spm_lm_loader import EOL_SYMBOL

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"

@dataclass
class GlueConfig(SentencePredictionConfig):
    ft_type: int = field(
        default=3,
        metadata={
            "help": "fine-tuning type: \
            1: gpt only \
            2: roberta only \
            3: roberta + gpt \
            4: roberta(freeze) + gpt \
            5: roberta + gpt(freeze) "
        },
    )

    seed: int = II("common.seed")
    eval_data: str = field(default="", metadata={"help": "dataset name"})
    test_split: str = field(default="test", metadata={"help": "test data split"})
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")

    tokens_per_sample: int = field(
        default=2048,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )

    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )

    gpt_dict: str = field(
        default="", metadata={"help": "gpt dict file"}
    )

    num_classes: int = field(
        default=2,
        metadata={"help": "num classes"},
    )

temp_dict = {
    "SST-2": ["sentence:"],
    "MNLI": ["premise:", "hypothesis:"],
    "MRPC": ["sentence 1:", "sentence 2:"],
    "RTE": ["sentence 1:", "sentence 2:"]
}


@register_task("glue", dataclass=GlueConfig)
class GlueTask(SentencePredictionTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    def __init__(self, cfg, data_dict, label_dict, all_dict, tokenizer):
        super().__init__(cfg, data_dict, label_dict)
        self.cfg = cfg
        self.all_dict = all_dict
        self.seed = cfg.seed
        self.tokenizer = tokenizer
        self.num_classes = self.cfg.num_classes

    @classmethod
    def load_dictionary(cls, filename, extra_mask_tokens=False, required_batch_size_multiple=1):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        
        if extra_mask_tokens:
            dictionary.add_symbol("<mask>")
            for i in range(100):
                dictionary.add_symbol(f"<mask_{i}>")

        dictionary.pad_to_multiple_(required_batch_size_multiple)

        return dictionary
    
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        # load gpt dict
        all_dict = Dictionary.load(cfg.gpt_dict)

        all_dict.add_symbol(EOL_SYMBOL)

        all_dict.pad_to_multiple_(cfg.required_batch_size_multiple)
        logger.info("all dictionary: {} types".format(len(all_dict)))

        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(cfg.data, "input0", "dict.txt"),
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not cfg.regression_target:
            label_dict = cls.load_dictionary(
                os.path.join(cfg.data, "label", "dict.txt"),
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict

        tokenizer = GPT2BPE(Namespace(
            gpt2_vocab_bpe=cfg.gpt2_vocab_bpe,
            gpt2_encoder_json=cfg.gpt2_encoder_json))

        return cls(cfg, data_dict, label_dict, all_dict, tokenizer)

    def build_model(self, cfg, from_checkpoint=False):
        from fairseq import models

        model = models.build_model(cfg, self, from_checkpoint)

        if self.cfg.ft_type == 4:
            for p in model.text_model.parameters():
                p.requires_grad = False
        elif self.cfg.ft_type == 5:
            for p in model.gpt_model.parameters():
                p.requires_grad = False

        if self.cfg.ft_type == 2:
            model.register_classification_head(
                self.cfg.classification_head_name,
                num_classes=self.cfg.num_classes,
            )
            # unfreeze all parameters in model
            for p in model.parameters():
                p.requires_grad = True

        return model

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(key, split):
            return os.path.join(self.cfg.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = data_utils.load_indexed_dataset(
                    split_path,
                    dictionary,
                    combine=combine,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset

        temp_list = []
        if self.cfg.eval_data in temp_dict:
            for temp in temp_dict[self.cfg.eval_data]:
                temp_ids = self.all_dict.encode_line(self.tokenizer.encode(temp), add_if_not_exist=False)[0:-1]
                temp_list.append(temp_ids.tolist())

        label_token_list = []
        for i in range(self.num_classes):
            label_token_ids = self.all_dict.encode_line(self.tokenizer.encode(self.label_dictionary[4+i]), add_if_not_exist=False)
            label_token_list.append(label_token_ids)

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        # remove </s>
        input0 = RemoveTokenDataset(input0, remove_index=-1)

        if len(temp_list) >= 1:
            temp_ids = temp_list[0]
            for temp_token in temp_ids[::-1]:
                input0 = PrependTokenDataset(input0, temp_token)

        input1 = make_dataset("input1", self.source_dictionary)

        if len(temp_list) >= 2:
            temp_ids = temp_list[1]
            for temp_token in temp_ids[::-1]:
                input1 = PrependTokenDataset(input1, temp_token)

        if self.cfg.init_token is not None:
            input0 = PrependTokenDataset(input0, self.cfg.init_token)

        if input1 is None:
            mlm_src_tokens = input0
        else:
            # remove </s>
            input1 = RemoveTokenDataset(input1, remove_index=-1)

            if self.cfg.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.cfg.separator_token)

            mlm_src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(mlm_src_tokens))

        mlm_src_tokens = maybe_shorten_dataset(
            mlm_src_tokens,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.max_positions(),
            self.cfg.seed,
        )

        if self.cfg.ft_type != 2 and ("valid" in split or "test" in split):
            mlm_src_tokens_valid = []
            for i in range(len(mlm_src_tokens)):
                mlm_src_tokens_valid.extend([mlm_src_tokens[i] for j in range(self.num_classes)])
            mlm_src_tokens = RawArrayDataset(mlm_src_tokens_valid)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "mlm_src_tokens": RightPadDataset(
                    mlm_src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(mlm_src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(mlm_src_tokens, reduce=True),
        }

        if self.cfg.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(mlm_src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset["net_input"].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.cfg.regression_target:
            label_dataset = make_dataset("label", self.label_dictionary)
            if self.cfg.ft_type != 2 and ("valid" in split or "test" in split):
                label_dataset_all = [label_token_list[i] for i in range(self.num_classes)] * len(label_dataset)

                label_dataset_tmp = []
                for i in range(len(label_dataset)):
                    label_dataset_tmp.extend([label_dataset[i]]*self.num_classes)
                label_dataset = label_dataset_tmp
            else:
                label_dataset_all = []
                for i in range(len(label_dataset)):
                    label_dataset_all.append(label_token_list[label_dataset[i][0].item()-4])
            label_dataset_all = RawArrayDataset(label_dataset_all)

            src_tokens_all = ConcatSentencesDataset(mlm_src_tokens, label_dataset_all)
            src_tokens = RemoveTokenDataset(src_tokens_all, remove_index=-1)
            gpt_targets = RemoveTokenDataset(src_tokens_all, remove_index=0)

            dataset["net_input"].update(
                src_tokens=RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                )
            )
            dataset.update(
                gpt_targets=RightPadDataset(
                    gpt_targets,
                    pad_idx=self.source_dictionary.pad(),
                )
            )

            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dictionary.eos(),
                        ),
                        offset=-self.label_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path("label", split))
            if os.path.exists(label_path):

                def parse_regression_target(i, line):
                    values = line.split()
                    assert (
                        len(values) == self.cfg.num_classes
                    ), f'expected num_classes={self.cfg.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                with open(label_path) as h:
                    dataset.update(
                        target=RawLabelDataset(
                            [
                                parse_regression_target(i, line.strip())
                                for i, line in enumerate(h.readlines())
                            ]
                        )
                    )
        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.cfg.no_shuffle or "valid" in split or "test" in split:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]


class RemoveTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, remove_index=0):
        super().__init__(dataset)
        self.remove_index = remove_index

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.remove_index == 0:
            item = item[1:]
        elif self.remove_index == -1:
            item = item[:-1]

        return item


class RawArrayDataset(FairseqDataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        if hasattr(dataset, 'sizes'):
            self._sizes = dataset.sizes
        else:
            try:
                self._sizes = np.array([len(x) for x in self.dataset])
            except:
                self._sizes =  np.array([1 for x in self.dataset])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)
