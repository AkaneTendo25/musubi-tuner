# Copy/paste example for PowerShell on a 24GB GPU.
# Target: LTX-2.3 video LoRA training with:
# - TREAD
# - DOP for video-only data (no audio training clips)
# - 24GB-friendly memory settings
#
# Replace the placeholder paths before running.
# Change the DOP class prompt to match your dataset's base class.

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py `
  --dataset_config "<path\to\dataset.toml>" `
  --ltx2_checkpoint "<path\to\ltx-2.3-22b-dev.safetensors>" `
  --gemma_root "<path\to\gemma-3-12b-it-qat-q4_0-unquantized>" `
  --gemma_load_in_8bit `
  --ltx2_mode video `
  --ltx_version 2.3 `
  --ltx_version_check_mode error `
  --network_module networks.lora_ltx2 `
  --lora_target_preset video_sa_ca_ff `
  --network_dim 32 `
  --network_alpha 32 `
  --optimizer_type AdamW8bit `
  --learning_rate 1e-4 `
  --lr_scheduler constant_with_warmup `
  --lr_warmup_steps 100 `
  --max_train_steps 5000 `
  --train_batch_size 1 `
  --sdpa `
  --fp8_base `
  --fp8_scaled `
  --gradient_checkpointing `
  --blocks_to_swap 10 `
  --use_pinned_memory_for_block_swap `
  --caption_dropout_rate 0.05 `
  --dop `
  --dop_args class=person multiplier=0.5 `
  --tread selection_ratio=0.5 start_layer_idx=3 end_layer_idx=-4 `
  --timestep_sampling shifted_logit_normal `
  --sample_at_first `
  --sample_every_n_epochs 1 `
  --sample_prompts "<path\to\sample_prompts.txt>" `
  --sample_with_offloading `
  --sample_tiled_vae `
  --sample_vae_tile_size 512 `
  --sample_vae_tile_overlap 64 `
  --sample_vae_temporal_tile_size 48 `
  --sample_vae_temporal_tile_overlap 8 `
  --save_every_n_steps 500 `
  --save_state `
  --output_dir "<path\to\output>" `
  --output_name ltx23_24gb_tread_dop `
  --logging_dir "<path\to\logs>"
