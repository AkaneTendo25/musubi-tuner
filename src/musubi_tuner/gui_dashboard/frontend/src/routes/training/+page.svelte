<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormCombobox from '$lib/components/FormCombobox.svelte';
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

	// Common optimizer presets
	const optimizerOptions = [
		'adamw8bit',
		'adamw',
		'adafactor',
		'adagrad',
		'lion8bit',
		'lion',
		'sgd',
		'sgdnesterov',
		'rmsprop',
		'adadelta',
		'adamax',
		'prodigy',
		'came',
	];

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
						<PathInput label="Gemma Safetensors" value={t.gemma_safetensors || ''} oninput={(e) => update('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
						<div class="grid grid-cols-3 gap-2">
							<FormSelect label="Mode" value={t.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} tooltip="Video/AV/Audio" />
							<FormSelect label="Precision" value={t.mixed_precision || 'bf16'} options={['no', 'fp16', 'bf16']} onchange={(e) => update('mixed_precision', e.target.value)} tooltip="Mixed precision mode" />
							<FormSelect label="LTX Version" value={t.ltx_version || '2.0'} options={['2.0', '2.3']} onchange={(e) => update('ltx_version', e.target.value)} tooltip="Target LTX version behavior" />
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="FP8 Base" checked={t.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision (VRAM savings)" />
							<FormToggle label="FP8 Scaled" checked={t.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} tooltip="Scaled FP8 for stability" />
							<FormToggle label="Flash Attn" checked={t.flash_attn ?? true} onchange={(e) => update('flash_attn', e.target.checked)} tooltip="Flash Attention 2" />
							<FormToggle label="SDPA" checked={t.sdpa ?? false} onchange={(e) => update('sdpa', e.target.checked)} tooltip="PyTorch SDPA attention" />
							<FormToggle label="Sage Attn" checked={t.sage_attn ?? false} onchange={(e) => update('sage_attn', e.target.checked)} tooltip="Sage Attention backend" />
							<FormToggle label="xFormers" checked={t.xformers ?? false} onchange={(e) => update('xformers', e.target.checked)} tooltip="xFormers attention" />
							<FormToggle label="Gemma 8b" checked={t.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
							<FormToggle label="Gemma 4b" checked={t.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
							<FormToggle label="No Dbl Quant" checked={t.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => update('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable double quantization (4-bit)" />
							<FormToggle label="Audio Only Model" checked={t.ltx2_audio_only_model ?? false} onchange={(e) => update('ltx2_audio_only_model', e.target.checked)} tooltip="Audio-only model architecture" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="LoRA">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Dim" type="number" value={t.network_dim ?? 16} oninput={(e) => update('network_dim', Number(e.target.value))} min={1} tooltip="LoRA rank" />
							<FormField label="Alpha" type="number" value={t.network_alpha ?? 16} oninput={(e) => update('network_alpha', Number(e.target.value))} min={1} tooltip="LoRA alpha" />
							<FormSelect label="Target" value={t.lora_target_preset || 't2v'} options={[
								{ value: 't2v', label: 't2v (all attn)' },
								{ value: 'v2v', label: 'v2v (all attn+FFN)' },
								{ value: 'video_sa', label: 'V:SA' },
								{ value: 'video_sa_ff', label: 'V:SA+FF' },
								{ value: 'video_sa_ca_ff', label: 'V:SA+CA+FF' },
								{ value: 'audio', label: 'audio' },
								{ value: 'audio_ref_only_ic', label: 'audio ref IC' },
								{ value: 'full', label: 'full (all)' }
							]} onchange={(e) => update('lora_target_preset', e.target.value)} tooltip="Target layers" />
						</div>
						<div class="grid grid-cols-2 gap-2">
						<FormSelect label="IC-LoRA Strategy" value={t.ic_lora_strategy || 'auto'} options={[
							{ value: 'auto', label: 'auto' },
							{ value: 'none', label: 'none' },
							{ value: 'v2v', label: 'v2v' },
							{ value: 'audio_ref_only_ic', label: 'audio_ref_only_ic' },
						]} onchange={(e) => update('ic_lora_strategy', e.target.value)} tooltip="IC-LoRA conditioning strategy. 'auto' follows lora_target_preset; 'audio_ref_only_ic' = audio-reference ID-LoRA style (requires av or audio mode)" />
					</div>
					{#if t.ic_lora_strategy === 'audio_ref_only_ic'}
					<div class="p-2 space-y-2" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
						<span class="text-[11px] font-medium" style="color: var(--text-muted);">Audio-Reference IC-LoRA</span>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Negative Positions" checked={t.audio_ref_use_negative_positions ?? false} onchange={(e) => update('audio_ref_use_negative_positions', e.target.checked)} tooltip="Place reference-audio token positions in negative time" />
							<FormToggle label="Mask Cross-Attn to Ref" checked={t.audio_ref_mask_cross_attention_to_reference ?? false} onchange={(e) => update('audio_ref_mask_cross_attention_to_reference', e.target.checked)} tooltip="Video attends only to target audio, not reference-audio tokens" />
							<FormToggle label="Mask Ref from Text" checked={t.audio_ref_mask_reference_from_text_attention ?? false} onchange={(e) => update('audio_ref_mask_reference_from_text_attention', e.target.checked)} tooltip="Block reference-audio tokens from attending to text tokens" />
						</div>
						<FormField label="Identity Guidance Scale" type="number" value={t.audio_ref_identity_guidance_scale ?? 0.0} oninput={(e) => update('audio_ref_identity_guidance_scale', Number(e.target.value))} step="0.1" min={0} tooltip="Extra forward pass without reference to isolate and amplify speaker identity (0 = disabled, recommended: 3.0)" />
						<div class="flex flex-wrap gap-x-4 gap-y-1 items-end">
							<FormToggle label="AV Bimodal CFG" checked={t.av_bimodal_cfg ?? false} onchange={(e) => update('av_bimodal_cfg', e.target.checked)} tooltip="Extra forward pass with cross-modal attention disabled to strengthen independent audio/video generation" />
							{#if t.av_bimodal_cfg}
							<FormField label="Bimodal Scale" type="number" value={t.av_bimodal_scale ?? 3.0} oninput={(e) => update('av_bimodal_scale', Number(e.target.value))} step="0.1" min={1} tooltip="Bimodal guidance strength. Applied as (scale-1) × delta. Default: 3.0" />
							{/if}
						</div>
					</div>
					{/if}
					<FormField label="Network Args" value={t.network_args || ''} oninput={(e) => update('network_args', e.target.value)} placeholder="key=value ..." tooltip="Extra network args (space-separated key=value)" />
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Dropout" type="number" value={t.network_dropout ?? ''} oninput={(e) => update('network_dropout', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="LoRA dropout rate" />
							<FormField label="Scale W Norms" type="number" value={t.scale_weight_norms ?? ''} oninput={(e) => update('scale_weight_norms', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Max norm for weight scaling" />
							<FormField label="Caption Drop" type="number" value={t.caption_dropout_rate ?? 0} oninput={(e) => update('caption_dropout_rate', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Caption dropout rate for CFG training" />
						</div>
						<PathInput label="Network Weights" value={t.network_weights || ''} oninput={(e) => update('network_weights', e.target.value)} showFiles tooltip="Resume from existing LoRA weights" />
						<PathInput label="LyCORIS Config" value={t.lycoris_config || ''} oninput={(e) => update('lycoris_config', e.target.value)} showFiles tooltip="Path to LyCORIS TOML config (enables LyCORIS mode)" />
						<FormSelect label="LyCORIS Quant Check" value={t.lycoris_quantized_base_check_mode || 'warn'} options={['off', 'warn', 'error']} onchange={(e) => update('lycoris_quantized_base_check_mode', e.target.value)} tooltip="Check for quantized base with LyCORIS" />
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Dim from Weights" checked={t.dim_from_weights ?? false} onchange={(e) => update('dim_from_weights', e.target.checked)} tooltip="Auto-detect dim/alpha from weights" />
							<FormToggle label="Save Orig LoRA" checked={t.save_original_lora ?? true} onchange={(e) => update('save_original_lora', e.target.checked)} tooltip="Save original LoRA format" />
							<FormToggle label="Train Connectors" checked={t.train_connectors ?? false} onchange={(e) => update('train_connectors', e.target.checked)} tooltip="Also apply LoRA to text connector modules. Requires caching with 'Cache Pre-Connector Features' enabled. Not compatible with LyCORIS." />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Quantization">
					<div class="space-y-2 pt-2">
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="NF4 Base" checked={t.nf4_base ?? false} onchange={(e) => update('nf4_base', e.target.checked)} tooltip="NF4 4-bit quantization (~75% VRAM savings)" />
							<FormToggle label="LoftQ Init" checked={t.loftq_init ?? false} onchange={(e) => update('loftq_init', e.target.checked)} tooltip="LoftQ initialization (compensates NF4 error)" />
							<FormToggle label="W8A8" checked={t.fp8_w8a8 ?? false} onchange={(e) => update('fp8_w8a8', e.target.checked)} tooltip="W8A8 activation quantization (requires FP8 Scaled)" />
							<FormToggle label="AWQ Calibration" checked={t.awq_calibration ?? false} onchange={(e) => update('awq_calibration', e.target.checked)} tooltip="Activation-aware calibration for NF4" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="NF4 Block Size" type="number" value={t.nf4_block_size ?? 32} oninput={(e) => update('nf4_block_size', Number(e.target.value))} disabled={!t.nf4_base} tooltip="Block size for NF4 quantization" />
							<FormField label="LoftQ Iters" type="number" value={t.loftq_iters ?? 2} oninput={(e) => update('loftq_iters', Number(e.target.value))} min={1} disabled={!t.loftq_init} tooltip="LoftQ alternating iterations" />
							<FormSelect label="W8A8 Mode" value={t.w8a8_mode || 'int8'} options={['int8', 'fp8']} onchange={(e) => update('w8a8_mode', e.target.value)} disabled={!t.fp8_w8a8} tooltip="int8 (Turing+) or fp8 (Ada+)" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="AWQ Alpha" type="number" value={t.awq_alpha ?? 0.25} oninput={(e) => update('awq_alpha', Number(e.target.value))} step="0.05" min={0} max={1} disabled={!t.awq_calibration} tooltip="AWQ scaling strength" />
							<FormField label="AWQ Batches" type="number" value={t.awq_num_batches ?? 8} oninput={(e) => update('awq_num_batches', Number(e.target.value))} min={1} disabled={!t.awq_calibration} tooltip="Calibration batches" />
							<FormSelect label="Quant Device" value={t.quantize_device || ''} options={[{value:'',label:'Auto'},{value:'cuda',label:'CUDA'},{value:'cpu',label:'CPU'}]} onchange={(e) => update('quantize_device', e.target.value || null)} tooltip="Device for quantization math" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Optimizer">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="LR" value={t.learning_rate ?? 1e-4} oninput={(e) => update('learning_rate', Number(e.target.value))} step="any" tooltip="Learning rate" />
							<FormCombobox label="Optimizer" value={t.optimizer_type || 'adamw8bit'} oninput={(e) => update('optimizer_type', e.target.value)} options={optimizerOptions} tooltip="Optimizer type (select preset or type custom)" />
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
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Decay Steps" type="number" value={t.lr_decay_steps ?? ''} oninput={(e) => update('lr_decay_steps', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="LR decay steps" />
							<FormField label="Timescale" type="number" value={t.lr_scheduler_timescale ?? ''} oninput={(e) => update('lr_scheduler_timescale', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="LR scheduler timescale" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Cycles" type="number" value={t.lr_scheduler_num_cycles ?? ''} oninput={(e) => update('lr_scheduler_num_cycles', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="Cosine restarts cycles" />
							<FormField label="Min LR Ratio" type="number" value={t.lr_scheduler_min_lr_ratio ?? ''} oninput={(e) => update('lr_scheduler_min_lr_ratio', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.01" tooltip="Minimum LR ratio" />
						</div>
						<FormField label="Audio LR" type="number" value={t.audio_lr ?? ''} oninput={(e) => update('audio_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Same as LR" step="any" tooltip="Separate LR for audio LoRA modules" />
						<FormField label="LR Args" value={t.lr_args || ''} oninput={(e) => update('lr_args', e.target.value)} placeholder="pattern=lr ..." tooltip="Per-module LR overrides (e.g. audio_attn=1e-6)" />
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

				<FormGroup title="Memory">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Blocks to Swap" type="number" value={t.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0-40" min={0} max={40} tooltip="CPU offload blocks" />
							<FormSelect label="Split Attn" value={t.split_attn_target || ''} options={[{value:'',label:'None'},{value:'all',label:'All'},{value:'self',label:'Self'},{value:'cross',label:'Cross'}]} onchange={(e) => update('split_attn_target', e.target.value || null)} tooltip="Split attention target" />
						</div>
						<FormSelect label="Split Mode" value={t.split_attn_mode || ''} options={[{value:'',label:'None'},{value:'batch',label:'Batch'},{value:'query',label:'Query'}]} onchange={(e) => update('split_attn_mode', e.target.value || null)} disabled={!t.split_attn_target} tooltip="Split mode" />
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="FFN Chunk" value={t.ffn_chunk_target || ''} options={[{value:'',label:'None'},{value:'all',label:'All'},{value:'video',label:'Video'},{value:'audio',label:'Audio'}]} onchange={(e) => update('ffn_chunk_target', e.target.value || null)} tooltip="FFN chunking" />
							<FormField label="Chunk Size" type="number" value={t.ffn_chunk_size ?? 0} oninput={(e) => update('ffn_chunk_size', Number(e.target.value))} disabled={!t.ffn_chunk_target} tooltip="Tokens per chunk" />
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Grad Checkpoint" checked={t.gradient_checkpointing ?? true} onchange={(e) => update('gradient_checkpointing', e.target.checked)} tooltip="Gradient checkpointing" />
							<FormToggle label="GC CPU Offload" checked={t.gradient_checkpointing_cpu_offload ?? false} onchange={(e) => update('gradient_checkpointing_cpu_offload', e.target.checked)} tooltip="Offload checkpointed activations to CPU" />
							<FormToggle label="Blockwise" checked={t.blockwise_checkpointing ?? false} onchange={(e) => update('blockwise_checkpointing', e.target.checked)} tooltip="Per-block checkpointing" />
							<FormToggle label="Pinned Memory" checked={t.use_pinned_memory_for_block_swap ?? false} onchange={(e) => update('use_pinned_memory_for_block_swap', e.target.checked)} tooltip="Pinned memory for block swap" />
							<FormToggle label="Offload img/txt" checked={t.img_in_txt_in_offloading ?? false} onchange={(e) => update('img_in_txt_in_offloading', e.target.checked)} tooltip="Offload img_in/txt_in to CPU" />
						</div>
						<FormField label="Blocks to Checkpoint" type="number" value={t.blocks_to_checkpoint ?? ''} oninput={(e) => update('blocks_to_checkpoint', e.target.value ? Number(e.target.value) : null)} placeholder="All" disabled={!t.blockwise_checkpointing} tooltip="Number of blocks to checkpoint (default: all)" />
						<FormField label="Attn Chunk Size" type="number" value={t.split_attn_chunk_size ?? ''} oninput={(e) => update('split_attn_chunk_size', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!t.split_attn_target} tooltip="Split attention chunk size" />
					</div>
				</FormGroup>

				<FormGroup title="Compile & CUDA">
					<div class="space-y-2 pt-2">
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="torch.compile" checked={t.compile ?? false} onchange={(e) => update('compile', e.target.checked)} tooltip="Enable torch.compile" />
							<FormToggle label="TF32" checked={t.cuda_allow_tf32 ?? false} onchange={(e) => update('cuda_allow_tf32', e.target.checked)} tooltip="Allow TF32 on Ampere+" />
							<FormToggle label="cuDNN Bench" checked={t.cuda_cudnn_benchmark ?? false} onchange={(e) => update('cuda_cudnn_benchmark', e.target.checked)} tooltip="cuDNN benchmark mode" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Backend" value={t.compile_backend || 'inductor'} oninput={(e) => update('compile_backend', e.target.value)} disabled={!t.compile} tooltip="Compile backend" />
							<FormField label="Mode" value={t.compile_mode || ''} oninput={(e) => update('compile_mode', e.target.value)} placeholder="default" disabled={!t.compile} tooltip="Compile mode (default, reduce-overhead, max-autotune)" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="CUDA Mem Fraction" type="number" value={t.cuda_memory_fraction ?? ''} oninput={(e) => update('cuda_memory_fraction', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="Limit CUDA memory fraction" />
							<FormField label="Cache Size Limit" type="number" value={t.compile_cache_size_limit ?? ''} oninput={(e) => update('compile_cache_size_limit', e.target.value ? Number(e.target.value) : null)} placeholder="Default" disabled={!t.compile} tooltip="torch.compile cache size limit" />
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
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Keep Last N Epochs (State)" type="number" value={t.save_last_n_epochs_state ?? ''} oninput={(e) => update('save_last_n_epochs_state', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N epoch optimizer states" />
							<FormField label="Keep Last N Steps (State)" type="number" value={t.save_last_n_steps_state ?? ''} oninput={(e) => update('save_last_n_steps_state', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N step optimizer states" />
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Save State" checked={t.save_state ?? false} onchange={(e) => update('save_state', e.target.checked)} tooltip="Save optimizer state" />
							<FormToggle label="State on End" checked={t.save_state_on_train_end ?? false} onchange={(e) => update('save_state_on_train_end', e.target.checked)} tooltip="Save state at training end" />
							<FormToggle label="Ckpt Metadata" checked={t.save_checkpoint_metadata ?? false} onchange={(e) => update('save_checkpoint_metadata', e.target.checked)} tooltip="Save JSON metadata alongside each checkpoint" />
							<FormToggle label="No Metadata" checked={t.no_metadata ?? false} onchange={(e) => update('no_metadata', e.target.checked)} tooltip="Skip metadata in checkpoint" />
							<FormToggle label="No Comfy Convert" checked={t.no_convert_to_comfy ?? false} onchange={(e) => update('no_convert_to_comfy', e.target.checked)} tooltip="Skip ComfyUI format conversion" />
							<FormToggle label="Full FP16" checked={t.full_fp16 ?? false} onchange={(e) => update('full_fp16', e.target.checked)} tooltip="FP16 gradients (stochastic rounding)" />
							<FormToggle label="Full BF16" checked={t.full_bf16 ?? false} onchange={(e) => update('full_bf16', e.target.checked)} tooltip="BF16 gradients (stochastic rounding)" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="Loss Type" value={t.loss_type || 'mse'} options={['mse', 'mae', 'l1', 'huber', 'smooth_l1']} onchange={(e) => update('loss_type', e.target.value)} tooltip="Loss function type" />
							<FormField label="Huber Delta" type="number" value={t.huber_delta ?? 1.0} oninput={(e) => update('huber_delta', Number(e.target.value))} step="0.1" disabled={!['huber','smooth_l1'].includes(t.loss_type)} tooltip="Delta for Huber/smooth_l1 loss" />
						</div>
						<PathInput label="Resume From" value={t.resume || ''} oninput={(e) => update('resume', e.target.value)} showFiles tooltip="Resume training from saved state" />
						<div class="grid grid-cols-2 gap-2">
							<FormToggle label="Auto-resume" checked={t.autoresume ?? false} onchange={(e) => update('autoresume', e.target.checked)} tooltip="Auto-resume from latest state in output_dir" />
							<FormToggle label="Reset Dataloader" checked={t.reset_dataloader ?? false} onchange={(e) => update('reset_dataloader', e.target.checked)} tooltip="Skip mid-epoch resume, restart epoch from beginning" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormToggle label="Reset Optimizer" checked={t.reset_optimizer ?? false} onchange={(e) => update('reset_optimizer', e.target.checked)} tooltip="Clear optimizer momentum/variance on resume" />
							<FormToggle label="Reset Optim Params" checked={t.reset_optimizer_params ?? false} onchange={(e) => update('reset_optimizer_params', e.target.checked)} tooltip="Reset lr/weight_decay to CLI values on resume" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="Logger" value={t.log_with || ''} options={[{value:'',label:'None'},{value:'tensorboard',label:'TensorBoard'},{value:'wandb',label:'W&B'}]} onchange={(e) => update('log_with', e.target.value || null)} tooltip="Logging integration" />
							<PathInput label="Log Dir" value={t.logging_dir || ''} oninput={(e) => update('logging_dir', e.target.value)} disabled={!t.log_with} tooltip="Log directory" />
						</div>
						<FormField label="W&B Run Name" value={t.wandb_run_name || ''} oninput={(e) => update('wandb_run_name', e.target.value)} disabled={t.log_with !== 'wandb'} placeholder="Auto" tooltip="Weights & Biases run name" />
						<FormField label="W&B API Key" value={t.wandb_api_key || ''} oninput={(e) => update('wandb_api_key', e.target.value)} disabled={t.log_with !== 'wandb'} placeholder="Optional" tooltip="W&B API key (or set WANDB_API_KEY env)" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Log Prefix" value={t.log_prefix || ''} oninput={(e) => update('log_prefix', e.target.value)} placeholder="None" tooltip="Prefix for log metrics" />
							<FormField label="Tracker Name" value={t.log_tracker_name || ''} oninput={(e) => update('log_tracker_name', e.target.value)} placeholder="Auto" tooltip="Custom tracker/project name" />
						</div>
						<FormField label="CUDA Memory Log" type="number" value={t.log_cuda_memory_every_n_steps ?? ''} oninput={(e) => update('log_cuda_memory_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Log CUDA memory every N steps" />
						<FormField label="Comment" value={t.training_comment || ''} oninput={(e) => update('training_comment', e.target.value)} placeholder="Optional training comment" tooltip="Saved in checkpoint metadata" />
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
							<FormToggle label="Precached" checked={t.use_precached_sample_prompts ?? false} onchange={(e) => update('use_precached_sample_prompts', e.target.checked)} tooltip="Use cached text embeddings" />
							<FormToggle label="Precached I2V" checked={t.use_precached_sample_latents ?? false} onchange={(e) => update('use_precached_sample_latents', e.target.checked)} tooltip="Use cached I2V latents" />
							<FormToggle label="Offload" checked={t.sample_with_offloading ?? false} onchange={(e) => update('sample_with_offloading', e.target.checked)} tooltip="Offload during sampling" />
							<FormToggle label="Merge Audio" checked={t.sample_merge_audio ?? false} onchange={(e) => update('sample_merge_audio', e.target.checked)} tooltip="Merge audio in samples" />
							<FormToggle label="No Audio" checked={t.sample_disable_audio ?? false} onchange={(e) => update('sample_disable_audio', e.target.checked)} tooltip="Disable audio sampling" />
							<FormToggle label="At First" checked={t.sample_at_first ?? false} onchange={(e) => update('sample_at_first', e.target.checked)} tooltip="Generate samples before training" />
							<FormToggle label="Tiled VAE" checked={t.sample_tiled_vae ?? false} onchange={(e) => update('sample_tiled_vae', e.target.checked)} tooltip="Tiled VAE for sampling (saves VRAM)" />
							<FormToggle label="Two-Stage" checked={t.sample_two_stage ?? false} onchange={(e) => update('sample_two_stage', e.target.checked)} tooltip="Two-stage spatial upsampling" />
							<FormToggle label="Audio Only" checked={t.sample_audio_only ?? false} onchange={(e) => update('sample_audio_only', e.target.checked)} tooltip="Audio-only samples" />
							<FormToggle label="No Flash Attn" checked={t.sample_disable_flash_attn ?? false} onchange={(e) => update('sample_disable_flash_attn', e.target.checked)} tooltip="Disable flash attention during sampling" />
							<FormToggle label="I2V Mask" checked={t.sample_i2v_token_timestep_mask ?? true} onchange={(e) => update('sample_i2v_token_timestep_mask', e.target.checked)} tooltip="Token timestep mask for I2V sampling" />
							<FormToggle label="Audio Subprocess" checked={t.sample_audio_subprocess ?? true} onchange={(e) => update('sample_audio_subprocess', e.target.checked)} tooltip="Run audio sampling in subprocess" />
							<FormToggle label="V2V Reference" checked={t.sample_include_reference ?? false} onchange={(e) => update('sample_include_reference', e.target.checked)} tooltip="Show V2V reference side-by-side" />
						</div>
						<PathInput label="Cache Dir" value={t.sample_prompts_cache || ''} oninput={(e) => update('sample_prompts_cache', e.target.value)} showFiles disabled={!t.use_precached_sample_prompts} tooltip="Cached text embeddings dir" />
						<PathInput label="I2V Latents Cache" value={t.sample_latents_cache || ''} oninput={(e) => update('sample_latents_cache', e.target.value)} showFiles disabled={!t.use_precached_sample_latents} tooltip="Cached I2V conditioning latents (.pt)" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="VAE Tile Size" type="number" value={t.sample_vae_tile_size ?? ''} oninput={(e) => update('sample_vae_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!t.sample_tiled_vae} tooltip="Tiled VAE spatial tile size" />
							<FormField label="VAE Tile Overlap" type="number" value={t.sample_vae_tile_overlap ?? ''} oninput={(e) => update('sample_vae_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="64" disabled={!t.sample_tiled_vae} tooltip="Spatial tile overlap" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Temporal Tile" type="number" value={t.sample_vae_temporal_tile_size ?? ''} oninput={(e) => update('sample_vae_temporal_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Off" disabled={!t.sample_tiled_vae} tooltip="Temporal tile size (0=disabled)" />
							<FormField label="Temporal Overlap" type="number" value={t.sample_vae_temporal_tile_overlap ?? ''} oninput={(e) => update('sample_vae_temporal_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="8" disabled={!t.sample_tiled_vae} tooltip="Temporal tile overlap" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Ref Downscale" type="number" value={t.reference_downscale ?? 1} oninput={(e) => update('reference_downscale', Number(e.target.value))} min={1} disabled={!t.sample_include_reference} tooltip="V2V reference downscale factor" />
							<FormField label="Ref Frames" type="number" value={t.reference_frames ?? 1} oninput={(e) => update('reference_frames', Number(e.target.value))} min={1} disabled={!t.sample_include_reference} tooltip="Number of reference frames" />
						</div>
						{#if t.sample_two_stage}
						<PathInput label="Upsampler Path" value={t.spatial_upsampler_path || ''} oninput={(e) => update('spatial_upsampler_path', e.target.value)} showFiles tooltip="Spatial upsampler model path" />
						<PathInput label="Distilled LoRA" value={t.distilled_lora_path || ''} oninput={(e) => update('distilled_lora_path', e.target.value)} showFiles tooltip="Distilled LoRA for stage 2" />
						<FormField label="Stage 2 Steps" type="number" value={t.sample_stage2_steps ?? 3} oninput={(e) => update('sample_stage2_steps', Number(e.target.value))} min={1} tooltip="Number of denoising steps for stage 2" />
						{/if}
						<div class="grid grid-cols-3 gap-2">
							<FormField label="W" type="number" value={t.width ?? 768} oninput={(e) => update('width', Number(e.target.value))} min={64} step={64} tooltip="Sample width" />
							<FormField label="H" type="number" value={t.height ?? 512} oninput={(e) => update('height', Number(e.target.value))} min={64} step={64} tooltip="Sample height" />
							<FormField label="Frames" type="number" value={t.sample_num_frames ?? 45} oninput={(e) => update('sample_num_frames', Number(e.target.value))} min={1} tooltip="Sample frames" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Validate / Steps" type="number" value={t.validate_every_n_steps ?? ''} oninput={(e) => update('validate_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Run validation every N steps" />
							<FormField label="Validate / Epochs" type="number" value={t.validate_every_n_epochs ?? ''} oninput={(e) => update('validate_every_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Run validation every N epochs" />
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
					{#if t.ltx2_mode === 'audio'}
					<FormField label="Audio Seq Resolution" type="number" value={t.audio_only_sequence_resolution ?? 64} oninput={(e) => update('audio_only_sequence_resolution', Number(e.target.value))} min={0} tooltip="Virtual pixel resolution for shifted_logit_normal in audio-only mode (0 = use cached geometry)" />
					{/if}
						</div>
						<FormToggle label="Persistent Workers" checked={t.persistent_data_loader_workers ?? true} onchange={(e) => update('persistent_data_loader_workers', e.target.checked)} tooltip="Keep workers between epochs" />
					</div>
				</FormGroup>

				<FormGroup title="Metadata">
					<div class="space-y-2 pt-2">
						<FormField label="Title" value={t.metadata_title || ''} oninput={(e) => update('metadata_title', e.target.value)} placeholder="LoRA name" tooltip="Model card title" />
						<FormField label="Author" value={t.metadata_author || ''} oninput={(e) => update('metadata_author', e.target.value)} tooltip="Model card author" />
						<FormField label="Description" value={t.metadata_description || ''} oninput={(e) => update('metadata_description', e.target.value)} tooltip="Model card description" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="License" value={t.metadata_license || ''} oninput={(e) => update('metadata_license', e.target.value)} placeholder="e.g. mit" tooltip="SPDX license identifier" />
							<FormField label="Tags" value={t.metadata_tags || ''} oninput={(e) => update('metadata_tags', e.target.value)} placeholder="tag1,tag2" tooltip="Comma-separated tags" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="HuggingFace Upload">
					<div class="space-y-2 pt-2">
						<FormField label="Repo ID" value={t.huggingface_repo_id || ''} oninput={(e) => update('huggingface_repo_id', e.target.value)} placeholder="user/model" tooltip="HuggingFace repo id (user/model)" />
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Repo Type" value={t.huggingface_repo_type || ''} oninput={(e) => update('huggingface_repo_type', e.target.value)} placeholder="model" tooltip="HuggingFace repo type" />
							<FormField label="Visibility" value={t.huggingface_repo_visibility || ''} oninput={(e) => update('huggingface_repo_visibility', e.target.value)} placeholder="private" tooltip="Repo visibility (public/private)" />
						</div>
						<FormField label="Path in Repo" value={t.huggingface_path_in_repo || ''} oninput={(e) => update('huggingface_path_in_repo', e.target.value)} placeholder="/" tooltip="Upload path within repo" />
						<FormField label="HF Token" value={t.huggingface_token || ''} oninput={(e) => update('huggingface_token', e.target.value)} placeholder="hf_..." tooltip="HuggingFace API token (or set HF_TOKEN env)" />
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="Upload State" checked={t.save_state_to_huggingface ?? false} onchange={(e) => update('save_state_to_huggingface', e.target.checked)} tooltip="Also upload optimizer state" />
							<FormToggle label="Resume from HF" checked={t.resume_from_huggingface ?? false} onchange={(e) => update('resume_from_huggingface', e.target.checked)} tooltip="Resume training from HuggingFace" />
							<FormToggle label="Async Upload" checked={t.async_upload ?? false} onchange={(e) => update('async_upload', e.target.checked)} tooltip="Upload checkpoints asynchronously" />
						</div>
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
