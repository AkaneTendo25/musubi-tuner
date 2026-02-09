<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import CheckpointInput from '$lib/components/CheckpointInput.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, updateSection } from '$lib/stores/project.js';
	import { processStatuses, startProcess, stopProcess } from '$lib/stores/processes.js';
	import { goto } from '$app/navigation';

	function update(key, value) { updateSection('training', key, value); }

	let t = $derived($projectConfig?.training || {});
	let trainingStatus = $derived($processStatuses.training || { state: 'idle', exit_code: null });
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-4">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Training</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">Configure and run LoRA training.</p>
		</div>

		<!-- Two-column layout -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
			<!-- Left column -->
			<div class="space-y-3">
				<FormGroup title="Model">
					<div class="space-y-2 pt-2">
						<CheckpointInput label="LTX-2 Checkpoint" value={t.ltx2_checkpoint || ''} onchange={(v) => update('ltx2_checkpoint', v)} showFiles tooltip="Path to LTX-2 checkpoint" />
						<CheckpointInput label="Gemma Root" value={t.gemma_root || ''} onchange={(v) => update('gemma_root', v)} tooltip="Gemma text encoder directory" />
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="Mode" value={t.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} tooltip="Video/AV/Audio" />
							<FormSelect label="Precision" value={t.mixed_precision || 'bf16'} options={['no', 'fp16', 'bf16']} onchange={(e) => update('mixed_precision', e.target.value)} tooltip="Mixed precision mode" />
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="FP8 Base" checked={t.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision (VRAM savings)" />
							<FormToggle label="FP8 Scaled" checked={t.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} tooltip="Scaled FP8 for stability" />
							<FormToggle label="Flash Attn" checked={t.flash_attn ?? true} onchange={(e) => update('flash_attn', e.target.checked)} tooltip="Flash Attention 2" />
							<FormToggle label="Gemma 8b" checked={t.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
							<FormToggle label="Gemma 4b" checked={t.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="LoRA">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Dim" type="number" value={t.network_dim ?? 16} oninput={(e) => update('network_dim', Number(e.target.value))} min={1} tooltip="LoRA rank" />
							<FormField label="Alpha" type="number" value={t.network_alpha ?? 16} oninput={(e) => update('network_alpha', Number(e.target.value))} min={1} tooltip="LoRA alpha" />
							<FormSelect label="Target" value={t.lora_target_preset || 't2v'} options={[
								{ value: 't2v', label: 't2v (attn)' },
								{ value: 'v2v', label: 'v2v (attn+FFN)' },
								{ value: 'audio', label: 'audio' },
								{ value: 'full', label: 'full (all)' }
							]} onchange={(e) => update('lora_target_preset', e.target.value)} tooltip="Target layers" />
						</div>
						<FormField label="Network Args" value={t.network_args || ''} oninput={(e) => update('network_args', e.target.value)} placeholder="key=value ..." tooltip="Extra network args (space-separated key=value)" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Dropout" type="number" value={t.network_dropout ?? ''} oninput={(e) => update('network_dropout', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="LoRA dropout rate" />
							<FormField label="Scale W Norms" type="number" value={t.scale_weight_norms ?? ''} oninput={(e) => update('scale_weight_norms', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Max norm for weight scaling" />
						</div>
						<PathInput label="Network Weights" value={t.network_weights || ''} oninput={(e) => update('network_weights', e.target.value)} showFiles tooltip="Resume from existing LoRA weights" />
						<FormToggle label="Dim from Weights" checked={t.dim_from_weights ?? false} onchange={(e) => update('dim_from_weights', e.target.checked)} tooltip="Auto-detect dim/alpha from weights" />
					</div>
				</FormGroup>

				<FormGroup title="Optimizer">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="LR" value={t.learning_rate ?? 1e-4} oninput={(e) => update('learning_rate', Number(e.target.value))} step="any" tooltip="Learning rate" />
							<FormField label="Optimizer" value={t.optimizer_type || 'adamw8bit'} oninput={(e) => update('optimizer_type', e.target.value)} tooltip="Optimizer type" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormSelect label="Scheduler" value={t.lr_scheduler || 'constant_with_warmup'} options={['constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'linear', 'polynomial', 'rex']} onchange={(e) => update('lr_scheduler', e.target.value)} tooltip="LR schedule" />
							<FormField label="Warmup" type="number" value={t.lr_warmup_steps ?? 100} oninput={(e) => update('lr_warmup_steps', Number(e.target.value))} min={0} tooltip="Warmup steps" />
							<FormField label="Grad Accum" type="number" value={t.gradient_accumulation_steps ?? 1} oninput={(e) => update('gradient_accumulation_steps', Number(e.target.value))} min={1} tooltip="Gradient accumulation" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Max Grad Norm" type="number" value={t.max_grad_norm ?? 1.0} oninput={(e) => update('max_grad_norm', Number(e.target.value))} step="0.1" tooltip="Gradient clipping" />
							<FormField label="Optimizer Args" value={t.optimizer_args || ''} oninput={(e) => update('optimizer_args', e.target.value)} placeholder="key=value ..." tooltip="Extra optimizer args" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Decay Steps" type="number" value={t.lr_decay_steps ?? ''} oninput={(e) => update('lr_decay_steps', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="LR decay steps" />
							<FormField label="Cycles" type="number" value={t.lr_scheduler_num_cycles ?? ''} oninput={(e) => update('lr_scheduler_num_cycles', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="Cosine restarts cycles" />
							<FormField label="Min LR Ratio" type="number" value={t.lr_scheduler_min_lr_ratio ?? ''} oninput={(e) => update('lr_scheduler_min_lr_ratio', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.01" tooltip="Minimum LR ratio" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Schedule">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Max Steps" type="number" value={t.max_train_steps ?? 1600} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Total training steps" />
							<FormField label="Max Epochs" type="number" value={t.max_train_epochs ?? ''} oninput={(e) => update('max_train_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Epochs (overrides steps)" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormSelect label="Timestep" value={t.timestep_sampling || 'shifted_logit_normal'} options={['sigma', 'uniform', 'sigmoid', 'shift', 'shifted_logit_normal', 'logsnr']} onchange={(e) => update('timestep_sampling', e.target.value)} tooltip="Timestep sampling" />
							<FormField label="Flow Shift" type="number" value={t.discrete_flow_shift ?? 1.0} oninput={(e) => update('discrete_flow_shift', Number(e.target.value))} step="0.1" tooltip="Flow matching shift" />
							<FormSelect label="Weighting" value={t.weighting_scheme || 'none'} options={['none', 'logit_normal', 'mode', 'cosmap', 'sigma_sqrt']} onchange={(e) => update('weighting_scheme', e.target.value)} tooltip="Loss weighting" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Seed" type="number" value={t.seed ?? ''} oninput={(e) => update('seed', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Random seed" />
							<FormField label="Guidance" type="number" value={t.guidance_scale ?? ''} oninput={(e) => update('guidance_scale', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Training guidance scale" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Sigmoid Scale" type="number" value={t.sigmoid_scale ?? ''} oninput={(e) => update('sigmoid_scale', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Sigmoid scale for timestep sampling" />
							<FormField label="Logit Mean" type="number" value={t.logit_mean ?? ''} oninput={(e) => update('logit_mean', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Logit normal mean" />
							<FormField label="Logit Std" type="number" value={t.logit_std ?? ''} oninput={(e) => update('logit_std', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Logit normal std" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Min Timestep" type="number" value={t.min_timestep ?? ''} oninput={(e) => update('min_timestep', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.01" min={0} max={1} tooltip="Minimum timestep value" />
							<FormField label="Max Timestep" type="number" value={t.max_timestep ?? ''} oninput={(e) => update('max_timestep', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.01" min={0} max={1} tooltip="Maximum timestep value" />
						</div>
					</div>
				</FormGroup>
			</div>

			<!-- Right column -->
			<div class="space-y-3">
				<FormGroup title="Output">
					<div class="space-y-2 pt-2">
						<PathInput label="Output Dir" value={t.output_dir || ''} oninput={(e) => update('output_dir', e.target.value)} tooltip="Checkpoint save directory" />
						<FormField label="Name" value={t.output_name || 'ltx2_lora'} oninput={(e) => update('output_name', e.target.value)} tooltip="Checkpoint filename" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Save / Epochs" type="number" value={t.save_every_n_epochs ?? ''} oninput={(e) => update('save_every_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Save every N epochs" />
							<FormField label="Save / Steps" type="number" value={t.save_every_n_steps ?? ''} oninput={(e) => update('save_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Save every N steps" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Keep Last N Epochs" type="number" value={t.save_last_n_epochs ?? ''} oninput={(e) => update('save_last_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N epoch checkpoints" />
							<FormField label="Keep Last N Steps" type="number" value={t.save_last_n_steps ?? ''} oninput={(e) => update('save_last_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N step checkpoints" />
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Save State" checked={t.save_state ?? false} onchange={(e) => update('save_state', e.target.checked)} tooltip="Save optimizer state" />
							<FormToggle label="State on End" checked={t.save_state_on_train_end ?? false} onchange={(e) => update('save_state_on_train_end', e.target.checked)} tooltip="Save state at training end" />
							<FormToggle label="No Comfy Convert" checked={t.no_convert_to_comfy ?? false} onchange={(e) => update('no_convert_to_comfy', e.target.checked)} tooltip="Skip ComfyUI format conversion" />
						</div>
						<PathInput label="Resume From" value={t.resume || ''} oninput={(e) => update('resume', e.target.value)} showFiles tooltip="Resume training from saved state" />
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="Logger" value={t.log_with || ''} options={[{value:'',label:'None'},{value:'tensorboard',label:'TensorBoard'},{value:'wandb',label:'W&B'}]} onchange={(e) => update('log_with', e.target.value || null)} tooltip="Logging integration" />
							{#if t.log_with}
								<PathInput label="Log Dir" value={t.logging_dir || ''} oninput={(e) => update('logging_dir', e.target.value)} tooltip="Log directory" />
							{/if}
						</div>
						{#if t.log_with === 'wandb'}
							<FormField label="W&B Run Name" value={t.wandb_run_name || ''} oninput={(e) => update('wandb_run_name', e.target.value)} placeholder="Auto" tooltip="Weights & Biases run name" />
						{/if}
						<FormField label="Comment" value={t.training_comment || ''} oninput={(e) => update('training_comment', e.target.value)} placeholder="Optional training comment" tooltip="Saved in checkpoint metadata" />
					</div>
				</FormGroup>

				<FormGroup title="Memory">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Blocks to Swap" type="number" value={t.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0-40" min={0} max={40} tooltip="CPU offload blocks" />
							<FormSelect label="Split Attn" value={t.split_attn_target || ''} options={[{value:'',label:'None'},{value:'all',label:'All'},{value:'self',label:'Self'},{value:'cross',label:'Cross'}]} onchange={(e) => update('split_attn_target', e.target.value || null)} tooltip="Split attention target" />
						</div>
						{#if t.split_attn_target}
							<FormSelect label="Split Mode" value={t.split_attn_mode || ''} options={[{value:'',label:'None'},{value:'batch',label:'Batch'},{value:'query',label:'Query'}]} onchange={(e) => update('split_attn_mode', e.target.value || null)} tooltip="Split mode" />
						{/if}
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="FFN Chunk" value={t.ffn_chunk_target || ''} options={[{value:'',label:'None'},{value:'all',label:'All'},{value:'video',label:'Video'},{value:'audio',label:'Audio'}]} onchange={(e) => update('ffn_chunk_target', e.target.value || null)} tooltip="FFN chunking" />
							{#if t.ffn_chunk_target}
								<FormField label="Chunk Size" type="number" value={t.ffn_chunk_size ?? 0} oninput={(e) => update('ffn_chunk_size', Number(e.target.value))} tooltip="Tokens per chunk" />
							{/if}
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Grad Checkpoint" checked={t.gradient_checkpointing ?? true} onchange={(e) => update('gradient_checkpointing', e.target.checked)} tooltip="Gradient checkpointing" />
							<FormToggle label="Blockwise" checked={t.blockwise_checkpointing ?? false} onchange={(e) => update('blockwise_checkpointing', e.target.checked)} tooltip="Per-block checkpointing" />
							<FormToggle label="Pinned Memory" checked={t.use_pinned_memory_for_block_swap ?? false} onchange={(e) => update('use_pinned_memory_for_block_swap', e.target.checked)} tooltip="Pinned memory for block swap" />
							<FormToggle label="Offload img/txt" checked={t.img_in_txt_in_offloading ?? false} onchange={(e) => update('img_in_txt_in_offloading', e.target.checked)} tooltip="Offload img_in/txt_in to CPU" />
						</div>
						{#if t.blockwise_checkpointing}
							<FormField label="Blocks to Checkpoint" type="number" value={t.blocks_to_checkpoint ?? ''} oninput={(e) => update('blocks_to_checkpoint', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Number of blocks to checkpoint (default: all)" />
						{/if}
						{#if t.split_attn_target}
							<FormField label="Attn Chunk Size" type="number" value={t.split_attn_chunk_size ?? ''} oninput={(e) => update('split_attn_chunk_size', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" tooltip="Split attention chunk size" />
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Compile & CUDA" collapsed>
					<div class="space-y-2 pt-2">
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="torch.compile" checked={t.compile ?? false} onchange={(e) => update('compile', e.target.checked)} tooltip="Enable torch.compile" />
							<FormToggle label="TF32" checked={t.cuda_allow_tf32 ?? false} onchange={(e) => update('cuda_allow_tf32', e.target.checked)} tooltip="Allow TF32 on Ampere+" />
							<FormToggle label="cuDNN Bench" checked={t.cuda_cudnn_benchmark ?? false} onchange={(e) => update('cuda_cudnn_benchmark', e.target.checked)} tooltip="cuDNN benchmark mode" />
						</div>
						{#if t.compile}
							<div class="grid grid-cols-2 gap-2">
								<FormField label="Backend" value={t.compile_backend || 'inductor'} oninput={(e) => update('compile_backend', e.target.value)} tooltip="Compile backend" />
								<FormField label="Mode" value={t.compile_mode || ''} oninput={(e) => update('compile_mode', e.target.value)} placeholder="default" tooltip="Compile mode (default, reduce-overhead, max-autotune)" />
							</div>
						{/if}
						<FormField label="CUDA Mem Fraction" type="number" value={t.cuda_memory_fraction ?? ''} oninput={(e) => update('cuda_memory_fraction', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="Limit CUDA memory fraction" />
					</div>
				</FormGroup>

				<FormGroup title="Sampling">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Every N Steps" type="number" value={t.sample_every_n_steps ?? ''} oninput={(e) => update('sample_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Sample every N steps" />
							<FormField label="Every N Epochs" type="number" value={t.sample_every_n_epochs ?? ''} oninput={(e) => update('sample_every_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Sample every N epochs" />
						</div>
						<PathInput label="Sample Prompts" value={t.sample_prompts || ''} oninput={(e) => update('sample_prompts', e.target.value)} showFiles tooltip="Prompts file" />
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Precached" checked={t.use_precached_sample_prompts ?? false} onchange={(e) => update('use_precached_sample_prompts', e.target.checked)} tooltip="Use cached embeddings" />
							<FormToggle label="Offload" checked={t.sample_with_offloading ?? false} onchange={(e) => update('sample_with_offloading', e.target.checked)} tooltip="Offload during sampling" />
							<FormToggle label="Merge Audio" checked={t.sample_merge_audio ?? false} onchange={(e) => update('sample_merge_audio', e.target.checked)} tooltip="Merge audio in samples" />
							<FormToggle label="No Audio" checked={t.sample_disable_audio ?? false} onchange={(e) => update('sample_disable_audio', e.target.checked)} tooltip="Disable audio sampling" />
							<FormToggle label="At First" checked={t.sample_at_first ?? false} onchange={(e) => update('sample_at_first', e.target.checked)} tooltip="Generate samples before training" />
							<FormToggle label="Tiled VAE" checked={t.sample_tiled_vae ?? false} onchange={(e) => update('sample_tiled_vae', e.target.checked)} tooltip="Tiled VAE for sampling (saves VRAM)" />
							<FormToggle label="Audio Only" checked={t.sample_audio_only ?? false} onchange={(e) => update('sample_audio_only', e.target.checked)} tooltip="Audio-only samples" />
							<FormToggle label="No Flash Attn" checked={t.sample_disable_flash_attn ?? false} onchange={(e) => update('sample_disable_flash_attn', e.target.checked)} tooltip="Disable flash attention during sampling" />
						</div>
						{#if t.use_precached_sample_prompts}
							<PathInput label="Cache Dir" value={t.sample_prompts_cache || ''} oninput={(e) => update('sample_prompts_cache', e.target.value)} showFiles tooltip="Cached embeddings dir" />
						{/if}
						{#if t.sample_tiled_vae}
							<FormField label="VAE Tile Size" type="number" value={t.sample_vae_tile_size ?? ''} oninput={(e) => update('sample_vae_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" tooltip="Tiled VAE spatial tile size" />
						{/if}
						<div class="grid grid-cols-3 gap-2">
							<FormField label="W" type="number" value={t.width ?? 768} oninput={(e) => update('width', Number(e.target.value))} min={64} step={64} tooltip="Sample width" />
							<FormField label="H" type="number" value={t.height ?? 512} oninput={(e) => update('height', Number(e.target.value))} min={64} step={64} tooltip="Sample height" />
							<FormField label="Frames" type="number" value={t.sample_num_frames ?? 45} oninput={(e) => update('sample_num_frames', Number(e.target.value))} min={1} tooltip="Sample frames" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Loss & Misc">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Video Weight" type="number" value={t.video_loss_weight ?? 1.0} oninput={(e) => update('video_loss_weight', Number(e.target.value))} step="0.1" tooltip="Video loss multiplier" />
							<FormField label="Audio Weight" type="number" value={t.audio_loss_weight ?? 1.0} oninput={(e) => update('audio_loss_weight', Number(e.target.value))} step="0.1" tooltip="Audio loss multiplier" />
						</div>
						<FormToggle label="Separate Audio Buckets" checked={t.separate_audio_buckets ?? true} onchange={(e) => update('separate_audio_buckets', e.target.checked)} tooltip="Separate audio/video buckets" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Workers" type="number" value={t.max_data_loader_n_workers ?? 2} oninput={(e) => update('max_data_loader_n_workers', Number(e.target.value))} min={0} tooltip="Dataloader workers" />
							<FormField label="1st Frame P" type="number" value={t.ltx2_first_frame_conditioning_p ?? 0.1} oninput={(e) => update('ltx2_first_frame_conditioning_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="First frame conditioning prob" />
						</div>
						<FormToggle label="Persistent Workers" checked={t.persistent_data_loader_workers ?? true} onchange={(e) => update('persistent_data_loader_workers', e.target.checked)} tooltip="Keep workers between epochs" />
					</div>
				</FormGroup>
			</div>
		</div>

		<!-- Controls -->
		<div class="py-4 flex items-center gap-4">
			<ProcessControls processType="training" status={trainingStatus} onStart={() => { startProcess('training'); goto('/training/dashboard'); }} onStop={() => stopProcess('training')} />
			<div class="flex-1"></div>
			{#if trainingStatus.state === 'running' || trainingStatus.state === 'stopping' || trainingStatus.state === 'finished'}
				<a href="/training/dashboard" class="px-4 py-2 text-[13px] font-medium" style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);">Dashboard</a>
			{/if}
		</div>

		<CommandPanel processType="training" defaultFilename="train.bat" />
	</div>
{/if}
