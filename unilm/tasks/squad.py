import os
import pickle
import torch
import numpy as np
from argparse import Namespace

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    BaseWrapperDataset,
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    NestedDictionaryDataset,
    SortDataset,
    NumelDataset,
    RightPadDataset,
    RawLabelDataset,
    FairseqDataset,
)

from fairseq.tasks import register_task, FairseqDataclass, FairseqTask
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from dataclasses import dataclass, field
from omegaconf import II, MISSING

@dataclass
class SquadConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    seed: int = II("common.seed")
    spm_model: str = field(
        default="",
        metadata={
            "help": "sentencepice model to tokenize the data"
        },
    )
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")

@register_task('squad', dataclass=SquadConfig)
class SQuADTask(FairseqTask):

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.args = args
        self.dictionary = dictionary
        self.seed = args.seed
        self.tokenizer = SentencepieceBPE(Namespace(sentencepiece_model=args.spm_model))
        assert self.tokenizer is not None
        # self.dictionary.add_symbol('[MASK]')

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
    def setup_task(cls, args, **kwargs):
        dictionary = cls.load_dictionary(
            os.path.join(args.data, 'dict.txt'),
            extra_mask_tokens=True,
            required_batch_size_multiple=args.required_batch_size_multiple
        )
        print('| Dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        features_file_path = os.path.join(self.args.data, "{}_features.pkl".format(split))
        examples_file_path = os.path.join(self.args.data, "{}_examples.pkl".format(split))

        if os.path.exists(features_file_path) and os.path.exists(examples_file_path):
            examples = pickle.load(open(examples_file_path, 'rb'))
            features = pickle.load(open(features_file_path, 'rb'))
        else:
            raise FileNotFoundError("cannot find {} or {}".format(features_file_path, examples_file_path))

        if split == 'valid':
            # save for eval
            self.eval_examples = examples
            self.eval_features = features

        src_tokens = RawArrayDataset([torch.from_numpy(np.array(f.input_ids)) for f in features])
        p_mask = RawArrayDataset([torch.from_numpy(np.array(f.p_mask)).bool() for f in features])
        if split == 'train':
            starts = RawLabelDataset([int(f.start_position) for f in features])
            ends = RawLabelDataset([int(f.end_position) for f in features])
            is_impossible = RawLabelDataset([int(f.is_impossible) for f in features])
        else:
            starts = ends = is_impossible = None
        #sizes = np.array([len(f.input_ids) for f in features])

        '''
            Input format: <s> question here ? </s> Passage </s>
        '''
        dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': RightPadDataset(
                        src_tokens,
                        pad_idx=self.dictionary.pad(),
                    ),  
                    'src_lengths': NumelDataset(src_tokens, reduce=False),
                },
                'targets': {
                    'starts': starts,
                    'ends': ends,
                    'is_impossible': is_impossible,
                    'p_mask': RightPadDataset(p_mask, pad_idx=1),
                },
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_tokens, reduce=True),
            },
            sizes=[src_tokens.sizes],
        )

        if split == 'train':
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_tokens))
            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_question_answering_head(
            'question_answering_head',
            num_classes=2,
        )
        return model
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        criterion.context_metrics(logging_outputs)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


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