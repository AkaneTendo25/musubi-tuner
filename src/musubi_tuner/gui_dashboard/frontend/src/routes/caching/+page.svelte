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
	import { defaultModelDir, effectiveGemmaRoot, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { startModelDownload, getModelDownloadStatus, cancelModelDownload, formatModelDownloadStatus, getModelDownloadTone, isActiveModelDownload } from '$lib/utils/modelDownloads.js';
	import { projectConfig, projectLoaded, updateSection, saveProjectNow } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, preloadLogsIfActive, startLogPolling } from '$lib/stores/processes.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import { onMount } from 'svelte';

	let cwd = $state('');
	let downloading = $state('');
	let downloadJobId = $state('');
	let downloadState = $state('');
	let modelStatus = $state('');
	let modelStatusTone = $state('muted');
	let downloadPollTimer = null;

	onMount(() => {
		fetch('/api/fs/cwd').then((res) => res.ok ? res.json() : null).then((data) => { cwd = data?.cwd || ''; }).catch(() => {});
		preloadLogsIfActive(['cache_latents', 'cache_text', 'cache_dino']);
		const logInterval = startLogPolling(['cache_latents', 'cache_text', 'cache_dino'], 1000);
		return () => {
			clearInterval(logInterval);
			if (downloadPollTimer) clearTimeout(downloadPollTimer);
		};
	});

	function updateCaching(key, value) { updateSection('caching', key, value); }

	let caching = $derived($projectConfig?.caching || {});
	let latentStatus = $derived($processStatuses.cache_latents || { state: 'idle', exit_code: null });
	let textStatus = $derived($processStatuses.cache_text || { state: 'idle', exit_code: null });
	let dinoStatus = $derived($processStatuses.cache_dino || { state: 'idle', exit_code: null });
	let latentLogs = $derived($processLogs.cache_latents || []);
	let textLogs = $derived($processLogs.cache_text || []);
	let dinoLogs = $derived($processLogs.cache_dino || []);
	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let resolvedLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, caching.ltx2_checkpoint || ''));
	let resolvedGemma = $derived(effectiveGemmaRoot(cwd, $projectConfig, caching.gemma_root || '', caching.gemma_safetensors || ''));
	let hasActiveDownload = $derived(Boolean(downloadJobId) && ['queued', 'running', 'cancelling'].includes(downloadState));

	function setModelStatus(status) {
		modelStatus = formatModelDownloadStatus(status);
		modelStatusTone = getModelDownloadTone(status);
		downloadState = status?.state || '';
	}

	async function finalizeDownload(status) {
		if (status.state === 'completed' && status.path) {
			updateCaching('ltx2_checkpoint', downloading === 'ltxav' ? status.path : caching.ltx2_checkpoint || '');
			if (downloading === 'gemma-unsloth') {
				updateCaching('gemma_root', status.path);
				updateCaching('gemma_safetensors', '');
			}
			projectConfig.update((config) => config ? { ...config, model_dir: modelDir } : config);
			await saveProjectNow();
		}
		downloadJobId = '';
		downloading = '';
	}

	async function pollDownload(jobId) {
		if (downloadPollTimer) clearTimeout(downloadPollTimer);
		try {
			const status = await getModelDownloadStatus(jobId);
			setModelStatus(status);
			if (!isActiveModelDownload(status)) {
				await finalizeDownload(status);
				return;
			}
			downloadPollTimer = setTimeout(() => pollDownload(jobId), 1000);
		} catch (e) {
			setModelStatus({ state: 'failed', error: e.message || 'Download status failed' });
			downloadJobId = '';
			downloading = '';
		}
	}

	async function downloadModel(preset) {
		if (downloadJobId) return;
		const targetPath = preset === 'ltxav' ? resolvedLtx : resolvedGemma;
		if (!targetPath) return;
		try {
			const job = await startModelDownload(preset, targetPath);
			downloading = preset;
			downloadJobId = job.job_id || '';
			setModelStatus(job);
			if (downloadJobId) {
				await pollDownload(downloadJobId);
			}
		} catch (e) {
			setModelStatus({ state: 'failed', error: e.message || 'Download failed' });
		}
	}

	async function stopDownload() {
		if (!downloadJobId) return;
		try {
			const status = await cancelModelDownload(downloadJobId);
			setModelStatus(status);
		} catch (e) {
			setModelStatus({ state: 'failed', error: e.message || 'Cancel failed' });
		}
	}
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
			<div class="grid grid-cols-2 xl:grid-cols-3 gap-3">
				<CheckpointInput label="LTX-2 Checkpoint" value={caching.ltx2_checkpoint || ''} onchange={(v) => updateCaching('ltx2_checkpoint', v)} showFiles tooltip="Path to the LTX-2 model checkpoint file" actionLabel="D" actionBusyLabel="..." actionDisabled={downloading === 'ltxav' && hasActiveDownload} actionTooltip={`Download to ${resolvedLtx}`} onaction={() => downloadModel('ltxav')} />
				<CheckpointInput label="Gemma Root" value={caching.gemma_root || ''} onchange={(v) => updateCaching('gemma_root', v)} tooltip="Root directory containing Gemma text encoder weights" actionLabel="D" actionBusyLabel="..." actionDisabled={downloading === 'gemma-unsloth' && hasActiveDownload} actionTooltip={`Download to ${resolvedGemma}`} onaction={() => downloadModel('gemma-unsloth')} />
				<FormSelect label="LTX-2 Mode" value={caching.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => updateCaching('ltx2_mode', e.target.value)} tooltip="Video: visual only, AV: audio+video, Audio: audio only" />
			</div>
			{#if modelStatus}
				<div class="flex items-center justify-between gap-3 text-[11px] px-3 py-2" style="color: {modelStatusTone === 'success' ? 'var(--success)' : modelStatusTone === 'accent' ? 'var(--accent)' : modelStatusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {modelStatusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : modelStatusTone === 'accent' ? 'var(--accent-muted)' : modelStatusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);">
					<span>{modelStatus}</span>
					{#if hasActiveDownload}
						<button
							type="button"
							onclick={stopDownload}
							disabled={downloadState === 'cancelling'}
							class="px-2 py-1 text-[11px] font-medium disabled:opacity-40"
							style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
						>Stop</button>
					{/if}
				</div>
			{/if}
			<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
				<PathInput label="Gemma Safetensors" value={caching.gemma_safetensors || ''} oninput={(e) => updateCaching('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
				<PathInput label="Text Encoder Ckpt" value={caching.ltx2_text_encoder_checkpoint || ''} oninput={(e) => updateCaching('ltx2_text_encoder_checkpoint', e.target.value)} showFiles tooltip="Separate text encoder checkpoint (if different from main)" />
				<FormSelect label="Text Precision" value={caching.mixed_precision || 'no'} options={['no', 'fp16', 'bf16']} onchange={(e) => updateCaching('mixed_precision', e.target.value)} tooltip="Mixed precision mode for text encoder caching." />
				<FormField label="Workers" type="number" value={caching.num_workers ?? ''} oninput={(e) => updateCaching('num_workers', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" tooltip="Number of data loader workers" />
			</div>
			<div class="flex flex-wrap gap-x-4 gap-y-1">
				<FormToggle label="Skip Existing" checked={caching.skip_existing ?? false} onchange={(e) => updateCaching('skip_existing', e.target.checked)} tooltip="Skip files that already have cached outputs" />
			</div>
			{#if $advancedMode}
				<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
					<FormSelect label="VAE Dtype" value={caching.vae_dtype || ''} options={[{ value: '', label: 'bfloat16 (default)' }, 'float16', 'bfloat16', 'float32']} onchange={(e) => updateCaching('vae_dtype', e.target.value || null)} tooltip="VAE dtype for latent caching. Blank uses the default `bfloat16`." />
					<FormField label="Device" value={caching.device || ''} oninput={(e) => updateCaching('device', e.target.value || null)} placeholder="Auto" tooltip="Torch device. Leave blank to auto-select the runtime device." />
					<FormSelect label="Quantize Device" value={caching.quantize_device || ''} options={[{ value: '', label: 'Auto' }, { value: 'cuda', label: 'CUDA' }, { value: 'cpu', label: 'CPU' }]} onchange={(e) => updateCaching('quantize_device', e.target.value || null)} tooltip="Device for quantization-related work. Blank auto-selects." />
					<FormToggle label="Keep Cache" checked={caching.keep_cache ?? false} onchange={(e) => updateCaching('keep_cache', e.target.checked)} tooltip="Keep old cache files when re-caching" />
				</div>
				<PathInput label="Dataset Manifest Output" value={caching.save_dataset_manifest || ''} oninput={(e) => updateCaching('save_dataset_manifest', e.target.value)} showFiles tooltip="Optional path to write a dataset manifest during latent caching." />
			{/if}
		</div>

		<!-- Two columns: Latents | Text -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-5">
			<!-- Cache Latents -->
			<div class="space-y-3">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Cache Latents</span>

				{#if $advancedMode}
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

					<FormGroup title="Precache I2V Latents">
						<div class="space-y-2 pt-2">
							<FormToggle label="Precache Sample Latents" checked={caching.precache_sample_latents ?? false} onchange={(e) => updateCaching('precache_sample_latents', e.target.checked)} tooltip="Pre-encode I2V conditioning latents from prompts defined on the Samples page." />
							{#if caching.precache_sample_latents}
								<PathInput label="External Prompts File" value={caching.sample_prompts || ''} oninput={(e) => updateCaching('sample_prompts', e.target.value)} showFiles tooltip="Optional override. Leave blank to use prompts defined on the Samples page." />
								<PathInput label="Latents Cache Dir" value={caching.sample_latents_cache || ''} oninput={(e) => updateCaching('sample_latents_cache', e.target.value)} tooltip="Directory for cached sample conditioning latents." />
							{/if}
						</div>
					</FormGroup>
				{/if}

				{#if caching.ltx2_mode === 'av' || caching.ltx2_mode === 'audio'}
					<FormGroup title="Audio Source" collapsed={false}>
						<div class="space-y-2 pt-2">
							<FormSelect label="Source" value={caching.ltx2_audio_source || 'video'} options={['video', 'audio_files']} onchange={(e) => updateCaching('ltx2_audio_source', e.target.value)} tooltip="Extract from video or load separate files" />
							{#if caching.ltx2_audio_source === 'audio_files'}
								<PathInput label="Audio Dir" value={caching.ltx2_audio_dir || ''} oninput={(e) => updateCaching('ltx2_audio_dir', e.target.value)} tooltip="Directory with audio files" />
								{#if $advancedMode}
									<FormField label="Extension" value={caching.ltx2_audio_ext || '.wav'} oninput={(e) => updateCaching('ltx2_audio_ext', e.target.value)} tooltip="Audio file extension" />
								{/if}
							{/if}
							{#if $advancedMode}
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Audio Dtype" value={caching.ltx2_audio_dtype || ''} oninput={(e) => updateCaching('ltx2_audio_dtype', e.target.value)} placeholder="Auto" tooltip="Audio latent dtype" />
									<FormField label="Audio Seq Res" type="number" value={caching.audio_only_sequence_resolution ?? 64} oninput={(e) => updateCaching('audio_only_sequence_resolution', Number(e.target.value))} min={1} tooltip="Audio-only sequence resolution" />
								</div>
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Video Latent Channels" type="number" value={caching.audio_video_latent_channels ?? ''} oninput={(e) => updateCaching('audio_video_latent_channels', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" min={1} tooltip="Override video latent channels when caching audio-only latents" />
									<FormField label="Video Latent Dtype" value={caching.audio_video_latent_dtype || ''} oninput={(e) => updateCaching('audio_video_latent_dtype', e.target.value)} placeholder="Auto" tooltip="Override video latent dtype for audio-only caching" />
								</div>
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Target Resolution" type="number" value={caching.audio_only_target_resolution ?? ''} oninput={(e) => updateCaching('audio_only_target_resolution', e.target.value ? Number(e.target.value) : null)} placeholder="Dataset default" min={1} tooltip="Square target resolution used to derive audio-only video latent shapes" />
									<FormField label="Target FPS" type="number" value={caching.audio_only_target_fps ?? ''} oninput={(e) => updateCaching('audio_only_target_fps', e.target.value ? Number(e.target.value) : null)} placeholder="Default" min={0} step="0.1" tooltip="Target FPS used to derive frame count for audio-only caching" />
								</div>
							{/if}
						</div>
					</FormGroup>
				{/if}

				<ProcessControls processType="cache_latents" status={latentStatus} onStart={() => startProcess('cache_latents')} onStop={() => stopProcess('cache_latents')} />
				<ProcessConsole lines={latentLogs} processType="cache_latents" initiallyCollapsed={false} />
				{#if $advancedMode}
					<CommandPanel processType="cache_latents" defaultFilename="cache_latents.bat" />
				{/if}
			</div>

			<!-- Cache Text -->
			<div class="space-y-3">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Cache Text Encoder</span>

				<FormGroup title="Gemma Quantization">
					<div class="space-y-2 pt-2">
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="8-bit" checked={caching.gemma_load_in_8bit ?? false} onchange={(e) => updateCaching('gemma_load_in_8bit', e.target.checked)} tooltip="Load Gemma with 8-bit quantization" />
							<FormToggle label="4-bit" checked={caching.gemma_load_in_4bit ?? false} onchange={(e) => updateCaching('gemma_load_in_4bit', e.target.checked)} tooltip="Load Gemma with 4-bit quantization" />
							<FormToggle label="No Dbl Quant" checked={caching.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => updateCaching('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable double quantization" />
							<FormToggle label="FP8 Weight Offload" checked={caching.gemma_fp8_weight_offload ?? true} onchange={(e) => updateCaching('gemma_fp8_weight_offload', e.target.checked)} tooltip="For FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. Disable this to keep more weights on VRAM and reduce RAM/pagefile pressure." />
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
							<PathInput label="External Prompts File" value={caching.sample_prompts || ''} oninput={(e) => updateCaching('sample_prompts', e.target.value)} showFiles tooltip="Optional override. Leave blank to use prompts defined on the Samples page." />
							<PathInput label="Cache Dir" value={caching.sample_prompts_cache || ''} oninput={(e) => updateCaching('sample_prompts_cache', e.target.value)} tooltip="Output directory for cached embeddings" />
						{/if}
					</div>
				</FormGroup>

				{#if $advancedMode}

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

					<FormGroup title="Connector LoRA">
						<div class="space-y-2 pt-2">
							<FormToggle label="Cache Pre-Connector Features" checked={caching.cache_before_connector ?? false} onchange={(e) => updateCaching('cache_before_connector', e.target.checked)} tooltip="Save pre-connector text features alongside standard embeddings. Required for --train_connectors during training." />
						</div>
					</FormGroup>
				{/if}

				<ProcessControls processType="cache_text" status={textStatus} onStart={() => startProcess('cache_text')} onStop={() => stopProcess('cache_text')} />
				<ProcessConsole lines={textLogs} processType="cache_text" initiallyCollapsed={false} />
				{#if $advancedMode}
					<CommandPanel processType="cache_text" defaultFilename="cache_text.bat" />
				{/if}
			</div>
		</div>
	</div>
{/if}
