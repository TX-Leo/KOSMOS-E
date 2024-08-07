import json
import os
import multiprocessing
import itertools
import ast

from infinibatch import iterators
from functools import partial
from .lm_loader import LMLoader
from .utils import NativeCheckpointableIterator, WeightIterator, EOL_SYMBOL
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from tiktoken.core import Encoding
import numpy as np

class SpmLmLoader(LMLoader):
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
    
    @staticmethod
    def fs_encode_line(fs_dict, words, append_eos=True):
        ids = []
        for i, word in enumerate(words):
            idx = fs_dict.index(word)
            ids.append(idx)
        if append_eos:
            ids.append(fs_dict.eos_index)
        return ids

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
            mlm_batch_size = sum([len(x[2]) for x in batch]) 

            gpt_max_length = max([len(x[0]) for x in batch])

            mlm_max_length = 0
            mlm_ntokens = 0
            for x in batch:
                for y in x[2]:
                    mlm_max_length = max(mlm_max_length, len(y))
                    mlm_ntokens += len(y)

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            mlm_source_ids = np.full(shape=(mlm_batch_size, mlm_max_length), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)
            mlm_mask_all = np.full(shape=(mlm_batch_size, mlm_max_length), dtype=np.int32, fill_value=0)

            mlm_index = 0
            for i, (gpt_ids, gpt_input_mask, mlm_ids_list, mlm_mask_list, gpt_loss_mask) in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
                gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
                gpt_input_mask_all[i, :len(gpt_ids)-1] = gpt_input_mask[:-1]
                gpt_loss_mask_all[i, :len(gpt_ids)-1] = gpt_loss_mask[1:]
                
                for j, (mlm_ids, mlm_mask) in enumerate(zip(mlm_ids_list, mlm_mask_list)):
                    mlm_source_ids[mlm_index, :len(mlm_ids)] = mlm_ids
                    mlm_mask_all[mlm_index, :len(mlm_mask)] = mlm_mask
                    mlm_index += 1
            
            ret_batch = {
                'text':{
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                        'mlm_src_tokens': mlm_source_ids.astype(np.int64) if mlm_batch_size !=0 else None,
                        'gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
                        'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
                        'mlm_mask': mlm_mask_all.astype(np.bool_) if mlm_batch_size !=0 else None
                    },
                    'target': gpt_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                    'mlm_ntokens': mlm_ntokens
                }
            }

            return ret_batch

        def collate_for_gpt(batch):
            batch_size = len(batch)
            gpt_max_length = max([len(x[0]) for x in batch])

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)
            chunk_tokens_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            segment_tokens_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            
            for i, (gpt_ids, gpt_input_mask, mlm_ids_list, mlm_mask_list, gpt_loss_mask, chunk_tokens, segment_tokens) in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
                gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
                gpt_input_mask_all[i, :len(gpt_ids)-1] = gpt_input_mask[:-1]
                gpt_loss_mask_all[i, :len(gpt_ids)-1] = gpt_loss_mask[1:]
                chunk_tokens_all[i, :len(gpt_ids)-1] = chunk_tokens[:-1]
                segment_tokens_all[i, :len(gpt_ids)-1] = segment_tokens[:-1]
                
            ret_batch = {
                'gpt_tune':{
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                        'mlm_src_tokens': None,
                        'gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
                        'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
                        'chunk_tokens': chunk_tokens_all.astype(np.int64),
                        'segment_tokens': segment_tokens_all.astype(np.int64),
                        'mlm_mask': None
                    },
                    'target': gpt_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                    'mlm_ntokens': 0
                }
            }

            return ret_batch

        if self.mlm_tokens_proportion == 0:
            padded_batches = iterators.MapIterator(
                batches, collate_for_gpt
            )
        else:
            padded_batches = iterators.MapIterator(
                batches, collate
            )

        return padded_batches
    
    def _mlm_cut(self, _random, doc):
        # doc format: {'input': XXXX, 'output': XXXXX}
        
        if self.mlm_tokens_proportion == 0:
            mlm_tokens = []
            mlm_mask = []
            full_doc = doc['input'] + doc['output']
            gpt_input_mask = [0] * len(full_doc)
            gpt_loss_mask = [0] * len(doc['input']) + [1] * len(doc['output'])
            chunk_tokens = [0] * len(full_doc)
            segment_tokens = [0] * len(full_doc)
            return mlm_tokens, mlm_mask, gpt_input_mask, gpt_loss_mask, chunk_tokens, segment_tokens
        else:
            assert NotImplementedError

    def _gpt(self, doc):
        return doc['input'] + doc['output']

    @staticmethod
    def _doc_jsonstr_to_ids(doc_jsonstr, spm_tokenizer=None, fs_dict=None):
        def _encode_line(line):
            if isinstance(spm_tokenizer, GPT2BPE):
                tokens = spm_tokenizer.encode(line).split(' ')
            elif isinstance(spm_tokenizer, Encoding):
                tokens = list(map(str, spm_tokenizer.encode(line + '\n', allowed_special="all")))
            else:
                tokens = spm_tokenizer.encode(line, out_type=str)
            tokenized_tokens = SpmLmLoader.fs_encode_line(fs_dict, tokens, append_eos=False)
            return tokenized_tokens

        assert EOL_SYMBOL in fs_dict.indices
        eol_index = fs_dict.indices[EOL_SYMBOL]
        bos_index = fs_dict.bos_index
        eos_index = fs_dict.eos_index
        try:
            obj = json.loads(doc_jsonstr)
        except Exception as e:
            obj = ast.literal_eval(doc_jsonstr)
        input_line = obj['input']
        output_line = obj['output']
        
        doc = {
            'input': [bos_index] + _encode_line(input_line),
            'output': _encode_line(output_line) + [eos_index]
        }
        
        if eos_index in (doc['input'] + doc['output'])[:-1]:
            print(f"Note, the eos index appear in the medium of setence {obj} (encode: {doc['input'] + doc['output']})")

        return doc

    def _read_from_files(self, source_file):
        data = []
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file
        print(file_path)
        # NOTE #### simple spm implementation ###############
        data = []
        for doc_jsonstr in lines:
            ret = SpmLmLoader._doc_jsonstr_to_ids(doc_jsonstr, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
            data.append(ret)

        return data