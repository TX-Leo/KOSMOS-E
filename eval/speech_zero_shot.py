#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""
import sys
sys.path.append( '.' )

import unilm

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

from unilm.data.speech.audio_utils import get_fbank, get_waveform, _get_torchaudio_fbank


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints aud_src_tokens aud_gpt_input_mask aud_masks")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def get_interactive_tokens_and_lengths(self, lines, encode_fn):
    """
    line format: [image]path<tab>text<tab>[image]path
    model input: `<s> <image> image hidden </image> My cat looking very dignified.</s>`
    """
    image_feature_length = self.args.image_feature_length
    bos_id = self.dictionary.bos()
    eos_id = self.dictionary.eos()
    boi_id = self.dictionary.index("<image>")
    eoi_id = self.dictionary.index("</image>")
    
    def convert_one_line(input_str):
        # TODO: input interleave image and text
        token = []
        img_src_token = []
        img_gpt_input_mask = []
        audio_masks = []
        segments = input_str.split('<tab>')
        token.append(bos_id)
        img_gpt_input_mask.append(0)
        for i, segment in enumerate(segments):
            if segment.startswith('[image]'):
                wav_path = segment[7:]
                # read audio and transform to tensor
                n_bins=40
                fbank = get_fbank(wav_path, output_sample_rate=16000, n_bins=n_bins)
                audio_downsample_rate = 2
                max_audio_token_length = 800
                fbank_len = fbank.shape[0]
                audio_token = torch.zeros(max_audio_token_length * audio_downsample_rate, n_bins)
                if fbank_len > max_audio_token_length * audio_downsample_rate:
                    fbank = fbank[:max_audio_token_length * audio_downsample_rate, :]
                    fbank_len = fbank.shape[0]
                audio_token[:fbank_len] = fbank
                img_src_token.append(audio_token)
                audio_mask = torch.zeros(max_audio_token_length)
                audio_hidden_size = fbank_len // audio_downsample_rate
                audio_mask[audio_hidden_size:] = 1
                audio_masks.append(audio_mask.bool())
                # token.extend([boi_id] + [boi_id] * image_feature_length + [eoi_id])
                token.extend([boi_id] + list(range(4, audio_hidden_size+4)) + [eoi_id])
                
                img_gpt_input_mask.extend([0] + [1] * audio_hidden_size + [0])
            else:
                text_tokens = self.source_dictionary.encode_line(
                    encode_fn(segment), add_if_not_exist=False
                ).tolist()
                token.extend(text_tokens[:-1]) # remove eos
                img_gpt_input_mask.extend([0] * (len(text_tokens) - 1)) # </s> in token
        assert len(token) == len(img_gpt_input_mask)
        token.append(eos_id)
        token = torch.LongTensor(token)
        img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)

        return token, img_src_token[0], img_gpt_input_mask, audio_masks[0]
    
    tokens = []
    img_src_tokens = []
    img_gpt_input_masks = []
    audio_masks = []
    for src_str in lines:
        token, img_src_token, img_gpt_input_mask, audio_mask = convert_one_line(src_str)
        tokens.append(token)
        img_src_tokens.append(img_src_token)
        img_gpt_input_masks.append(img_gpt_input_mask)
        audio_masks.append(audio_mask)
    lengths = [t.numel() for t in tokens]
    
    return tokens, lengths, img_src_tokens, img_gpt_input_masks, audio_masks


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths, aud_src_tokens, aud_gpt_input_mask, audio_masks = get_interactive_tokens_and_lengths(task, lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_speech_inference(
            tokens, lengths, aud_src_tokens, aud_gpt_input_mask, audio_masks, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        aud_src_tokens = batch["net_input"]["aud_src_tokens"]
        aud_gpt_input_mask = batch["net_input"]["aud_gpt_input_mask"]
        aud_masks = batch["net_input"]["aud_masks"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            aud_src_tokens=aud_src_tokens,
            aud_gpt_input_mask=aud_gpt_input_mask,
            aud_masks=aud_masks,
            constraints=constraints,
        )


def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            aud_src_tokens = batch.aud_src_tokens
            aud_gpt_input_mask = batch.aud_gpt_input_mask
            aud_masks = batch.aud_masks
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                    "aud_src_tokens": aud_src_tokens,
                    "aud_gpt_input_mask": aud_gpt_input_mask,
                    "aud_masks": aud_masks,
                },
            }
            translate_start_time = time.time()
            translations = task.inference_step(
                generator, models, sample, constraints=constraints
            )
            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            if cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            src_str = ""
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                print("S-{}\t{}".format(id_, src_str))
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    print(
                        "C-{}\t{}".format(
                            id_,
                            tgt_dict.string(constraint, cfg.common_eval.post_process),
                        )
                    )

            # Process top predictions
            for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print("H-{}\t{}\t{}".format(id_, score, hypo_str))
                # detokenized hypothesis
                print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
                print(
                    "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                )
                if cfg.generation.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    print("A-{}\t{}".format(id_, alignment_str))

        # update running id_ counter
        start_id += len(inputs)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
