<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import CheckpointInput from '$lib/components/CheckpointInput.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, updateSection } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, fetchLogs } from '$lib/stores/processes.js';
	import { onMount } from 'svelte';

	onMount(() => {
		fetchLogs('cache_latents');
		fetchLogs('cache_text');
		fetchLogs('cache_dino');
	});

	function updateCaching(key, value) { updateSection('caching', key, value); }

	let caching = $derived($projectConfig?.caching || {});
	let latentStatus = $derived($processStatuses.cache_latents || { state: 'idle', exit_code: null });
	let textStatus = $derived($processStatuses.cache_text || { state: 'idle', exit_code: null });
	let dinoStatus = $derived($processStatuses.cache_dino || { state: 'idle', exit_code: null });
	let latentLogs = $derived($processLogs.cache_latents || []);
	let textLogs = $derived($processLogs.cache_text || []);
	let dinoLogs = $derived($processLogs.cache_dino || []);
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-5">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Caching</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">Cache latents and text encoder outputs before training.</p>
		</div>

		<!-- Shared Settings -->
		<div class="p-4 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
				<CheckpointInput label="LTX-2 Checkpoint" value={caching.ltx2_checkpoint || ''} onchange={(v) => updateCaching('ltx2_checkpoint', v)} showFiles tooltip="Path to the LTX-2 model checkpoint file" />
				<CheckpointInput label="Gemma Root" value={caching.gemma_root || ''} onchange={(v) => updateCaching('gemma_root', v)} tooltip="Root directory containing Gemma text encoder weights" />
				<FormSelect label="LTX-2 Mode" value={caching.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => updateCaching('ltx2_mode', e.target.value)} tooltip="Video: visual only, AV: audio+video, Audio: audio only" />
				<FormSelect label="VAE Dtype" value={caching.vae_dtype || 'bfloat16'} options={['float16', 'bfloat16', 'float32']} onchange={(e) => updateCaching('vae_dtype', e.target.value)} tooltip="Data type for VAE (bfloat16 recommended)" />
			</div>
			<div class="grid grid-cols-2 gap-3">
				<PathInput label="Gemma Safetensors" value={caching.gemma_safetensors || ''} oninput={(e) => updateCaching('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
				<PathInput label="Text Encoder Ckpt" value={caching.ltx2_text_encoder_checkpoint || ''} oninput={(e) => updateCaching('ltx2_text_encoder_checkpoint', e.target.value)} showFiles tooltip="Separate text encoder checkpoint (if different from main)" />
			</div>
			<div class="flex items-center gap-5">
				<FormField label="Device" value={caching.device || 'cuda'} oninput={(e) => updateCaching('device', e.target.value)} tooltip="Torch device (cuda, cpu)" />
				<FormField label="Workers" type="number" value={caching.num_workers ?? ''} oninput={(e) => updateCaching('num_workers', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" tooltip="Number of data loader workers" />
				<FormToggle label="Skip Existing" checked={caching.skip_existing ?? true} onchange={(e) => updateCaching('skip_existing', e.target.checked)} tooltip="Skip files that already have cached outputs" />
				<FormToggle label="Keep Cache" checked={caching.keep_cache ?? false} onchange={(e) => updateCaching('keep_cache', e.target.checked)} tooltip="Keep old cache files when re-caching" />
			</div>
		</div>

		<!-- Two columns: Latents | Text -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-5">
			<!-- Cache Latents -->
			<div class="space-y-3">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Cache Latents</span>

				<FormGroup title="VAE Tiling">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-3">
							<FormField label="Chunk Size" type="number" value={caching.vae_chunk_size ?? ''} oninput={(e) => updateCaching('vae_chunk_size', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Frames per VAE chunk" />
							<FormField label="Spatial Tile" type="number" value={caching.vae_spatial_tile_size ?? ''} oninput={(e) => updateCaching('vae_spatial_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="e.g. 512" tooltip="Spatial tile size (reduces VRAM)" />
						</div>
						<div class="grid grid-cols-3 gap-3">
							<FormField label="Spatial Overlap" type="number" value={caching.vae_spatial_tile_overlap ?? ''} oninput={(e) => updateCaching('vae_spatial_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="64" tooltip="Spatial tile overlap" />
							<FormField label="Temporal Tile" type="number" value={caching.vae_temporal_tile_size ?? ''} oninput={(e) => updateCaching('vae_temporal_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Temporal tile size" />
							<FormField label="Temporal Overlap" type="number" value={caching.vae_temporal_tile_overlap ?? ''} oninput={(e) => updateCaching('vae_temporal_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="24" tooltip="Temporal tile overlap" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Reference (V2V)">
					<div class="grid grid-cols-2 gap-3 pt-2">
						<FormField label="Ref Frames" type="number" value={caching.reference_frames ?? 1} oninput={(e) => updateCaching('reference_frames', Number(e.target.value))} min={1} tooltip="Reference frames for V2V" />
						<FormField label="Ref Downscale" type="number" value={caching.reference_downscale ?? 1} oninput={(e) => updateCaching('reference_downscale', Number(e.target.value))} min={1} tooltip="Reference downscale factor" />
					</div>
				</FormGroup>

				{#if caching.ltx2_mode === 'av' || caching.ltx2_mode === 'audio'}
					<FormGroup title="Audio Source" collapsed={false}>
						<div class="space-y-2 pt-2">
							<FormSelect label="Source" value={caching.ltx2_audio_source || 'video'} options={['video', 'audio_files']} onchange={(e) => updateCaching('ltx2_audio_source', e.target.value)} tooltip="Extract from video or load separate files" />
							{#if caching.ltx2_audio_source === 'audio_files'}
								<PathInput label="Audio Dir" value={caching.ltx2_audio_dir || ''} oninput={(e) => updateCaching('ltx2_audio_dir', e.target.value)} tooltip="Directory with audio files" />
								<FormField label="Extension" value={caching.ltx2_audio_ext || '.wav'} oninput={(e) => updateCaching('ltx2_audio_ext', e.target.value)} tooltip="Audio file extension" />
							{/if}
							<div class="grid grid-cols-2 gap-2">
								<FormField label="Audio Dtype" value={caching.ltx2_audio_dtype || ''} oninput={(e) => updateCaching('ltx2_audio_dtype', e.target.value)} placeholder="Auto" tooltip="Audio latent dtype" />
								<FormField label="Audio Seq Res" type="number" value={caching.audio_only_sequence_resolution ?? 64} oninput={(e) => updateCaching('audio_only_sequence_resolution', Number(e.target.value))} min={1} tooltip="Audio-only sequence resolution" />
							</div>
						</div>
					</FormGroup>
				{/if}

				<ProcessControls processType="cache_latents" status={latentStatus} onStart={() => startProcess('cache_latents')} onStop={() => stopProcess('cache_latents')} />
				<ProcessConsole lines={latentLogs} />
				<CommandPanel processType="cache_latents" defaultFilename="cache_latents.bat" />
			</div>

			<!-- Cache Text -->
			<div class="space-y-3">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Cache Text Encoder</span>

				<FormSelect label="Mixed Precision" value={caching.mixed_precision || 'bf16'} options={['no', 'fp16', 'bf16']} onchange={(e) => updateCaching('mixed_precision', e.target.value)} tooltip="Mixed precision mode (bf16 recommended)" />

				<FormGroup title="Gemma Quantization">
					<div class="space-y-2 pt-2">
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="8-bit" checked={caching.gemma_load_in_8bit ?? false} onchange={(e) => updateCaching('gemma_load_in_8bit', e.target.checked)} tooltip="Load Gemma with 8-bit quantization" />
							<FormToggle label="4-bit" checked={caching.gemma_load_in_4bit ?? false} onchange={(e) => updateCaching('gemma_load_in_4bit', e.target.checked)} tooltip="Load Gemma with 4-bit quantization" />
							<FormToggle label="No Dbl Quant" checked={caching.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => updateCaching('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable double quantization" />
						</div>
						{#if caching.gemma_load_in_4bit}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect label="Quant Type" value={caching.gemma_bnb_4bit_quant_type || 'nf4'} options={['nf4', 'fp4']} onchange={(e) => updateCaching('gemma_bnb_4bit_quant_type', e.target.value)} tooltip="NF4 recommended" />
								<FormSelect label="Compute Dtype" value={caching.gemma_bnb_4bit_compute_dtype || 'auto'} options={['auto', 'fp16', 'bf16', 'fp32']} onchange={(e) => updateCaching('gemma_bnb_4bit_compute_dtype', e.target.value)} tooltip="Compute dtype for 4-bit" />
							</div>
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Precache Samples">
					<div class="space-y-2 pt-2">
						<FormToggle label="Precache Sample Prompts" checked={caching.precache_sample_prompts ?? false} onchange={(e) => updateCaching('precache_sample_prompts', e.target.checked)} tooltip="Cache text embeddings for sample prompts" />
						{#if caching.precache_sample_prompts}
							<PathInput label="Prompts File" value={caching.sample_prompts || ''} oninput={(e) => updateCaching('sample_prompts', e.target.value)} showFiles tooltip="Text file with prompts (one per line)" />
							<PathInput label="Cache Dir" value={caching.sample_prompts_cache || ''} oninput={(e) => updateCaching('sample_prompts_cache', e.target.value)} tooltip="Output directory for cached embeddings" />
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Precache Preservation">
					<div class="space-y-2 pt-2">
						<FormToggle label="Precache Preservation" checked={caching.precache_preservation_prompts ?? false} onchange={(e) => updateCaching('precache_preservation_prompts', e.target.checked)} tooltip="Cache preservation/regularization prompts" />
						{#if caching.precache_preservation_prompts}
							<PathInput label="Cache Dir" value={caching.preservation_prompts_cache || ''} oninput={(e) => updateCaching('preservation_prompts_cache', e.target.value)} tooltip="Output directory for cached preservation embeddings" />
							<FormToggle label="Blank" checked={caching.blank_preservation ?? false} onchange={(e) => updateCaching('blank_preservation', e.target.checked)} tooltip="Use blank prompts" />
							<FormToggle label="DOP" checked={caching.dop ?? false} onchange={(e) => updateCaching('dop', e.target.checked)} tooltip="Differential Output Preservation" />
							{#if caching.dop}
								<FormField label="Class Prompt" value={caching.dop_class_prompt || ''} oninput={(e) => updateCaching('dop_class_prompt', e.target.value)} placeholder="e.g. woman" tooltip="Class word for DOP" />
							{/if}
						{/if}
					</div>
				</FormGroup>

				<ProcessControls processType="cache_text" status={textStatus} onStart={() => startProcess('cache_text')} onStop={() => stopProcess('cache_text')} />
				<ProcessConsole lines={textLogs} />
				<CommandPanel processType="cache_text" defaultFilename="cache_text.bat" />
			</div>
		</div>
	</div>
{/if}
