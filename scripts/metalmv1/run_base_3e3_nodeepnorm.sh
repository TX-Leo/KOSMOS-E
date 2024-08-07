echo 'OMPI_COMM_WORLD_RANK: ' $OMPI_COMM_WORLD_RANK
echo 'MASTER_IP: ' $MASTER_IP
echo 'MASTER_PORT: ' $MASTER_PORT

# Follow https://github.com/donglixp/unitask/blob/lm-freezeele/yamls-unitask/elefrz-gpt-scr-updatelast.yaml to cook 
# Encoder: 
# electra discriminator 24L, 1024; 400M
# GPT: 
# Position: absolute position + relative position?
# Preln deepnorm
# Bsz: 512; Step 1M
# Sequence length 2k
# LR: 2e-4 or 2.5e-4
# Data: Turing
# Tokenization: 64K

# https://github.com/shumingma/DeepNet/commit/02e4279e4a141c3bf051fb66acf3c219d6e2b857

# fairseq model cfg
# {'common': {'_name': None, 'no_progress_bar': False, 'log_interval': 25, 'log_format': 'json', 'wandb_project': None, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': True, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': True, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'user_dir': None, 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False}, 'common_eval': {'_name': None, 'path': None, 'post_process': None, 'quiet': False, 'model_overrides': '{}', 'results_path': None}, 'distributed_training': {'_name': None, 'distributed_world_size': 64, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'device_id': 0, 'local_rank': 0, 'distributed_no_spawn': False, 'ddp_backend': 'c10d', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'fast_stat_sync': False, 'broadcast_buffers': False, 'distributed_wrapper': 'DDP', 'slowmo_momentum': None, 'slowmo_algorithm': 'LocalSGD', 'localsgd_frequency': 3, 'nprocs_per_node': 8, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': 'never', 'zero_sharding': 'os', 'tpu': False, 'distributed_num_procs': 8}, 'dataset': {'_name': None, 'num_workers': 2, 'skip_invalid_size_inputs_valid_test': False, 'max_tokens': None, 'batch_size': 1, 'required_batch_size_multiple': 1, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'validate_interval': 1, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': False, 'max_tokens_valid': None, 'batch_size_valid': 1, 'curriculum': 0, 'gen_subset': 'test', 'num_shards': 1, 'shard_id': 0}, 'optimization': {'_name': None, 'max_epoch': 0, 'max_update': 572204, 'stop_time_hours': 0.0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [4], 'lr': [0.0003], 'min_lr': -1.0, 'use_bmuf': False}, 'checkpoint': {'_name': None, 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 1, 'save_interval_updates': 10000, 'keep_interval_updates': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': True, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'model_parallel_size': 1, 'distributed_rank': 0}, 'bmuf': {'_name': None, 'block_lr': 1.0, 'block_momentum': 0.875, 'global_sync_iter': 50, 'warmup_iterations': 500, 'use_nbm': False, 'average_sync': False, 'distributed_world_size': 64}, 'generation': {'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': False, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False}, 'eval_lm': {'_name': None, 'output_word_probs': False, 'output_word_stats': False, 'context_window': 0, 'softmax_batch': 9223372036854775807}, 'interactive': {'_name': None, 'buffer_size': 0, 'input': '-'}, 'model': {'_name': 'transformer_lm_gpt2_small', 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'relu_dropout': 0.0, 'decoder_embed_dim': 1024, 'decoder_output_dim': 1024, 'decoder_input_dim': 1024, 'decoder_ffn_embed_dim': 4096, 'decoder_layers': 24, 'decoder_attention_heads': 16, 'decoder_normalize_before': True, 'no_decoder_final_norm': False, 'adaptive_softmax_cutoff': None, 'adaptive_softmax_dropout': 0.0, 'adaptive_softmax_factor': 4.0, 'no_token_positional_embeddings': False, 'share_decoder_input_output_embed': True, 'character_embeddings': False, 'character_filters': '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]', 'character_embedding_dim': 4, 'char_embedder_highway_layers': 2, 'adaptive_input': False, 'adaptive_input_factor': 4.0, 'adaptive_input_cutoff': None, 'tie_adaptive_weights': False, 'tie_adaptive_proj': False, 'decoder_learned_pos': False, 'decoder_layerdrop': 0.0, 'decoder_layers_to_keep': None, 'layernorm_embedding': False, 'no_scale_embedding': False, 'checkpoint_activations': False, 'quant_noise_pq': 0.0, 'quant_noise_pq_block_size': 8, 'quant_noise_scalar': 0.0, 'add_bos_token': False, 'tokens_per_sample': 2048, 'max_target_positions': 2048, 'tpu': False}, 'task': {'_name': 'language_modeling', 'data': '.', 'sample_break_mode': 'none', 'tokens_per_sample': 2048, 'output_dictionary_size': -1, 'self_target': False, 'future_target': False, 'past_target': False, 'add_bos_token': False, 'max_source_positions': None, 'max_target_positions': None, 'shorten_method': 'none', 'shorten_data_split_list': '', 'seed': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'tpu': False}, 'criterion': {'_name': 'cross_entropy', 'sentence_avg': False}, 'optimizer': {'_name': 'adam', 'adam_betas': '(0.9, 0.98)', 'adam_eps': 1e-08, 'weight_decay': 0.01, 'use_old_adam': False, 'tpu': False, 'lr': [0.0003]}, 'lr_scheduler': {'no_progress_bar': False, 'log_interval': 25, 'log_format': 'json', 'wandb_project': None, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': True, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': True, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'user_dir': None, 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'criterion': 'cross_entropy', 'tokenizer': None, 'bpe': None, 'optimizer': 'adam', 'lr_scheduler': 'polynomial_decay', 'scoring': 'bleu', 'task': 'language_modeling', 'num_workers': 2, 'skip_invalid_size_inputs_valid_test': False, 'max_tokens': None, 'batch_size': 1, 'required_batch_size_multiple': 1, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'validate_interval': 1, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': False, 'max_tokens_valid': None, 'batch_size_valid': 1, 'curriculum': 0, 'gen_subset': 'test', 'num_shards': 1, 'shard_id': 0, 'distributed_world_size': 64, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'device_id': 0, 'local_rank': 0, 'distributed_no_spawn': False, 'ddp_backend': 'c10d', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'fast_stat_sync': False, 'broadcast_buffers': False, 'distributed_wrapper': 'DDP', 'slowmo_momentum': None, 'slowmo_algorithm': 'LocalSGD', 'localsgd_frequency': 3, 'nprocs_per_node': 8, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': 'never', 'zero_sharding': 'os', 'arch': 'transformer_lm_gpt2_small', 'max_epoch': 0, 'max_update': 572204, 'stop_time_hours': 0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [4], 'lr': [0.0003], 'min_lr': -1.0, 'use_bmuf': False, 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 1, 'save_interval_updates': 10000, 'keep_interval_updates': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': True, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'sample_break_mode': 'none', 'tokens_per_sample': 2048, 'output_dictionary_size': -1, 'self_target': False, 'future_target': False, 'past_target': False, 'add_bos_token': False, 'max_source_positions': None, 'max_target_positions': None, 'shorten_method': 'none', 'shorten_data_split_list': '', 'adam_betas': '(0.9, 0.98)', 'adam_eps': 1e-08, 'weight_decay': 0.01, 'use_old_adam': False, 'force_anneal': None, 'warmup_updates': 715, 'end_learning_rate': 0.0, 'power': 1.0, 'total_num_update': 572204, 'pad': 1, 'eos': 2, 'unk': 3, 'share_decoder_input_output_embed': True, 'dropout': 0.0, 'attention_dropout': 0.0, 'no_seed_provided': False, 'decoder_embed_dim': 1024, 'decoder_ffn_embed_dim': 4096, 'decoder_layers': 24, 'decoder_attention_heads': 16, 'activation_fn': 'gelu', 'adaptive_softmax_cutoff': None, 'adaptive_softmax_dropout': 0, 'adaptive_softmax_factor': 4, 'decoder_learned_pos': False, 'decoder_layerdrop': 0, 'decoder_layers_to_keep': None, 'quant_noise_pq': 0, 'quant_noise_pq_block_size': 8, 'quant_noise_scalar': 0, 'no_token_positional_embeddings': False, 'character_embeddings': False, 'decoder_output_dim': 1024, 'decoder_input_dim': 1024, 'decoder_normalize_before': True, 'no_decoder_final_norm': False, 'adaptive_input': False, 'adaptive_input_factor': 4, 'adaptive_input_cutoff': None, 'tie_adaptive_weights': False, 'tie_adaptive_proj': False, 'no_scale_embedding': False, 'layernorm_embedding': False, 'checkpoint_activations': False, '_name': 'polynomial_decay'}, 'scoring': {'_name': 'bleu', 'pad': 1, 'eos': 2, 'unk': 3}, 'bpe': None, 'tokenizer': None}

export branch=gptmodel_base-lr3e-3-bs512-reset-wd0.01-mlmdrop-warm500-fxeos-fxfinal-mask-gpt-scratch-nodeepnorm-no_aphla-nogithub-noarvix-nopubmed-mtnlg

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2  \
    --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP --master_port=$MASTER_PORT train.py /mnt/unilm/shaohanh/data/tnlg_config/ \
    --task gpt_pretraining  \
    --tokens-per-sample 1024 \
    --criterion unigpt  \
    --arch gptmodel_small \
    --required-batch-size-multiple 1 \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6  \
    --clip-norm 2.0 \
    --lr-scheduler polynomial_decay  \
    --weight-decay 0.01  \
    --lr 0.003  \
    --warmup-updates 500  \
    --total-num-update 500000 \
    --max-update 500000 \
    --max-sentences 8 \
    --update-freq 4 \
    --log-format simple  \
    --log-interval 50 \
    --disable-validation \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --min-loss-scale 0.0001 \
    --fp16-no-flatten-grads \
    --seed 1 \
    --dict-path /mnt/unilm/shumma/data/16g/dict.txt  \
    --spm-model /mnt/unilm/shumma/data/16g/sentencepiece.bpe.model \
    --save-dir /mnt/unilm/shaohanh/exp/unitask_exp/${branch}/ \
    --tensorboard-logdir /mnt/unilm/shaohanh/exp/unitask_exp/${branch}/tb-logs \
    --ddp-backend=legacy_ddp \
    --distributed-no-spawn \
    --batch-read-ahead 5000 \
    --reset-dataloader \
    --mlm-tokens-proportion 0 --mlm-cut-length 0

# --deep-net-decoder \
# --last-ln-scale \

# --all-gather-list-size 102400000 \
# --checkpoint-activations \
# --checkpoint-activations \

