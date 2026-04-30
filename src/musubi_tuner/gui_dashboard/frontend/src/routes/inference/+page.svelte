<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import CheckpointInput from '$lib/components/CheckpointInput.svelte';
	import ModelPathStatus from '$lib/components/ModelPathStatus.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { defaultModelDir, effectiveGemmaRoot, effectiveGemmaSafetensors, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { startModelDownload, getModelDownloadStatus, cancelModelDownload, formatModelDownloadStatus, getModelDownloadTone, isActiveModelDownload, getModelDownloadPresets, getModelDownloadPreflight, checkPathExists, scanCheckpoints, modelDownloadTooltip, formatModelPreflightStatus } from '$lib/utils/modelDownloads.js';
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
	let downloadPresets = $state({});
	let ltxDownloadExists = $state(false);
	let gemmaDownloadExists = $state(false);
	let foundLtxPath = $state('');
	let foundGemmaPath = $state('');
	let downloadPollTimer = null;

	onMount(() => {
		fetch('/api/fs/cwd').then((res) => res.ok ? res.json() : null).then((data) => { cwd = data?.cwd || ''; }).catch(() => {});
		getModelDownloadPresets().then((presets) => { downloadPresets = presets; }).catch(() => {});
		preloadLogsIfActive('inference');
		const logInterval = startLogPolling('inference', 1000);
		return () => {
			clearInterval(logInterval);
			if (downloadPollTimer) clearTimeout(downloadPollTimer);
		};
	});

	function update(key, value) { updateSection('inference', key, value); }

	let s = $derived($projectConfig?.inference || {});
	let inferenceStatus = $derived($processStatuses.inference || { state: 'idle', exit_code: null });
	let inferenceLogs = $derived($processLogs.inference || []);
	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let resolvedLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, s.ltx2_checkpoint || ''));
	let activeGemmaSafetensors = $derived(effectiveGemmaSafetensors($projectConfig, s.gemma_safetensors || ''));
	let gemmaRootDisabled = $derived(Boolean(activeGemmaSafetensors));
	let resolvedGemma = $derived(effectiveGemmaRoot(cwd, $projectConfig, s.gemma_root || '', activeGemmaSafetensors));
	let hasActiveDownload = $derived(Boolean(downloadJobId) && ['queued', 'running', 'cancelling'].includes(downloadState));

	$effect(() => {
		const path = resolvedLtx;
		let cancelled = false;
		checkPathExists(path).then((exists) => { if (!cancelled) ltxDownloadExists = exists; }).catch(() => { if (!cancelled) ltxDownloadExists = false; });
		return () => { cancelled = true; };
	});

	$effect(() => {
		const path = resolvedGemma;
		let cancelled = false;
		checkPathExists(path).then((exists) => { if (!cancelled) gemmaDownloadExists = exists; }).catch(() => { if (!cancelled) gemmaDownloadExists = false; });
		return () => { cancelled = true; };
	});

	$effect(() => {
		if (!cwd) return;
		let cancelled = false;
		scanCheckpoints('ltx2', modelDir).then((results) => {
			if (!cancelled) foundLtxPath = results.find((p) => p === resolvedLtx) || results[0] || '';
		}).catch(() => { if (!cancelled) foundLtxPath = ''; });
		return () => { cancelled = true; };
	});

	$effect(() => {
		if (!cwd) return;
		let cancelled = false;
		scanCheckpoints('gemma', modelDir).then((results) => {
			if (!cancelled) foundGemmaPath = results.find((p) => p === resolvedGemma) || results[0] || '';
		}).catch(() => { if (!cancelled) foundGemmaPath = ''; });
		return () => { cancelled = true; };
	});

	function setModelStatus(status) {
		modelStatus = formatModelDownloadStatus(status);
		modelStatusTone = getModelDownloadTone(status);
		downloadState = status?.state || '';
	}

	async function finalizeDownload(status) {
		if (status.state === 'completed' && status.path) {
			update('ltx2_checkpoint', downloading === 'ltxav' ? status.path : s.ltx2_checkpoint || '');
			if (downloading === 'gemma-unsloth') {
				update('gemma_root', status.path);
				update('gemma_safetensors', '');
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
			const preflight = await getModelDownloadPreflight(preset, targetPath);
			setModelStatus({
				state: preflight.ok ? 'queued' : 'failed',
				message: formatModelPreflightStatus(preflight),
				error: preflight.errors?.join('; ')
			});
			if (!preflight.ok) return;
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
	<div class="space-y-4">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Inference</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">Generate videos using your trained LoRA. Advanced mode exposes the full inference CLI surface.</p>
		</div>

		<!-- Two-column layout -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
			<!-- Left column -->
			<div class="space-y-3">
				<FormGroup title="Model">
					<div class="space-y-2 pt-2">
						<CheckpointInput label="LTX-2 Checkpoint" value={s.ltx2_checkpoint || ''} onchange={(v) => update('ltx2_checkpoint', v)} showFiles tooltip="Path to LTX-2 checkpoint" actionLabel="D" actionBusyLabel="..." actionDisabled={hasActiveDownload || ltxDownloadExists} actionTooltip={modelDownloadTooltip(downloadPresets, 'ltxav', resolvedLtx, ltxDownloadExists)} onaction={() => downloadModel('ltxav')} />
						<ModelPathStatus exists={ltxDownloadExists} foundPath={foundLtxPath} disabled={hasActiveDownload} onusefound={(path) => update('ltx2_checkpoint', path)} />
						<PathInput label="VAE Checkpoint" value={s.vae || ''} oninput={(e) => update('vae', e.target.value)} showFiles tooltip="Optional separate VAE checkpoint. Leave blank to reuse the LTX-2 checkpoint." />
						<CheckpointInput label="Gemma Root" value={s.gemma_root || ''} onchange={(v) => update('gemma_root', v)} disabled={gemmaRootDisabled} tooltip={gemmaRootDisabled ? 'Ignored while Gemma Safetensors is set' : 'Gemma text encoder directory'} actionLabel="D" actionBusyLabel="..." actionDisabled={gemmaRootDisabled || hasActiveDownload || gemmaDownloadExists} actionTooltip={gemmaRootDisabled ? 'Gemma Safetensors is active' : modelDownloadTooltip(downloadPresets, 'gemma-unsloth', resolvedGemma, gemmaDownloadExists)} onaction={() => downloadModel('gemma-unsloth')} />
						<ModelPathStatus exists={gemmaRootDisabled || gemmaDownloadExists} foundPath={foundGemmaPath} disabled={gemmaRootDisabled || hasActiveDownload} onusefound={(path) => { update('gemma_root', path); update('gemma_safetensors', ''); }} />
						<PathInput label="Gemma Safetensors" value={s.gemma_safetensors || ''} oninput={(e) => update('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
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
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Device" value={s.device || ''} oninput={(e) => update('device', e.target.value)} placeholder="Auto" tooltip="Force `cpu` or `cuda`" />
							<FormSelect label="Mode" value={s.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} tooltip="Video/AV/Audio" />
							<FormSelect label="Precision" value={s.mixed_precision || 'bf16'} options={['no', 'fp16', 'bf16']} onchange={(e) => update('mixed_precision', e.target.value)} tooltip="Mixed precision mode" />
							<FormSelect label="Attn Mode" value={s.attn_mode || 'torch'} options={['torch', 'xformers', 'flash', 'flash3', 'sdpa', 'sage']} onchange={(e) => update('attn_mode', e.target.value)} tooltip="Attention implementation" />
						</div>
						{#if $advancedMode}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect label="VAE Dtype" value={s.vae_dtype || ''} options={[{ value: '', label: 'Default' }, 'bfloat16', 'float16', 'float32']} onchange={(e) => update('vae_dtype', e.target.value || null)} tooltip="Optional dtype override for a separate VAE checkpoint" />
							</div>
						{/if}
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="FP8 Base" checked={s.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision (VRAM savings)" />
							<FormToggle label="FP8 Scaled" checked={s.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} tooltip="Scaled FP8 for stability" />
							<FormToggle label="Gemma 8b" checked={s.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
							<FormToggle label="Gemma 4b" checked={s.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
							<FormToggle label="FP8 Weight Offload" checked={s.gemma_fp8_weight_offload ?? true} onchange={(e) => update('gemma_fp8_weight_offload', e.target.checked)} tooltip="For FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. Disable this to keep more weights on VRAM and reduce RAM/pagefile pressure." />
						</div>
						{#if $advancedMode}
							<FormField label="FP8 Keep Blocks" value={s.fp8_keep_blocks || ''} oninput={(e) => update('fp8_keep_blocks', e.target.value)} placeholder="0,1,2,45" tooltip="Transformer block indices to keep in high precision when FP8 Scaled is enabled. Ranges like 0-2,45 are accepted." />
							<div class="grid grid-cols-2 gap-2">
								<FormSelect label="Gemma 4b Type" value={s.gemma_bnb_4bit_quant_type || 'nf4'} options={['nf4', 'fp4']} onchange={(e) => update('gemma_bnb_4bit_quant_type', e.target.value)} tooltip="bitsandbytes 4-bit quant type" />
								<div class="flex items-end">
									<FormToggle label="Disable Double Quant" checked={s.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => update('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable bitsandbytes nested quantization" />
								</div>
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField label="W8A8 Mode" value={s.w8a8_mode || 'int8'} oninput={(e) => update('w8a8_mode', e.target.value)} placeholder="int8 or fp8" tooltip="W8A8 quantization mode" />
								<FormField label="Network Dim" type="number" value={s.network_dim ?? 0} oninput={(e) => update('network_dim', Number(e.target.value || 0))} min={0} tooltip="LoRA rank for LoftQ initialization" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField label="NF4 Block Size" type="number" value={s.nf4_block_size ?? 64} oninput={(e) => update('nf4_block_size', Number(e.target.value || 64))} min={1} tooltip="NF4 block size" />
								<FormField label="LoftQ Iters" type="number" value={s.loftq_iters ?? 2} oninput={(e) => update('loftq_iters', Number(e.target.value || 2))} min={1} tooltip="LoftQ iteration count" />
								<FormField label="AWQ Batches" type="number" value={s.awq_num_batches ?? 8} oninput={(e) => update('awq_num_batches', Number(e.target.value || 8))} min={1} tooltip="AWQ calibration batches" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField label="AWQ Alpha" type="number" value={s.awq_alpha ?? 0.25} oninput={(e) => update('awq_alpha', Number(e.target.value || 0.25))} step="0.01" tooltip="AWQ alpha" />
								<FormField label="FP8 Upcast Seed" type="number" value={s.fp8_upcast_seed ?? 0} oninput={(e) => update('fp8_upcast_seed', Number(e.target.value || 0))} min={0} tooltip="Seed for stochastic FP8 upcast" />
							</div>
							<div class="flex flex-wrap gap-x-4 gap-y-1">
								<FormToggle label="FlashAttn" checked={s.flash_attn ?? false} onchange={(e) => update('flash_attn', e.target.checked)} tooltip="Emit `--flash_attn`" />
								<FormToggle label="Flash3" checked={s.flash3 ?? false} onchange={(e) => update('flash3', e.target.checked)} tooltip="Emit `--flash3`" />
								<FormToggle label="SDPA" checked={s.sdpa ?? false} onchange={(e) => update('sdpa', e.target.checked)} tooltip="Emit `--sdpa`" />
								<FormToggle label="xFormers" checked={s.xformers ?? false} onchange={(e) => update('xformers', e.target.checked)} tooltip="Emit `--xformers`" />
								<FormToggle label="FP8 W8A8" checked={s.fp8_w8a8 ?? false} onchange={(e) => update('fp8_w8a8', e.target.checked)} tooltip="Use W8A8 quantization for DiT" />
								<FormToggle label="NF4 Base" checked={s.nf4_base ?? false} onchange={(e) => update('nf4_base', e.target.checked)} tooltip="Use NF4 quantization for DiT" />
								<FormToggle label="LoftQ Init" checked={s.loftq_init ?? false} onchange={(e) => update('loftq_init', e.target.checked)} tooltip="Initialize DiT quantization with LoftQ" />
								<FormToggle label="AWQ Calibration" checked={s.awq_calibration ?? false} onchange={(e) => update('awq_calibration', e.target.checked)} tooltip="Run AWQ calibration" />
								<FormToggle label="FP8 Upcast" checked={s.fp8_upcast ?? false} onchange={(e) => update('fp8_upcast', e.target.checked)} tooltip="Upcast FP8 quantization during load" />
								<FormToggle label="Stochastic Upcast" checked={s.fp8_upcast_stochastic ?? false} onchange={(e) => update('fp8_upcast_stochastic', e.target.checked)} tooltip="Use stochastic FP8 upcast" />
							</div>
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="LoRA">
					<div class="space-y-2 pt-2">
						<PathInput label="LoRA Weight" value={s.lora_weight || ''} oninput={(e) => update('lora_weight', e.target.value)} showFiles tooltip="Path to LoRA safetensors file" />
						<FormField label="Multiplier" type="number" value={s.lora_multiplier ?? 1.0} oninput={(e) => update('lora_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="LoRA weight multiplier" />
						{#if $advancedMode}
							<FormField label="Include Patterns" value={s.include_patterns || ''} oninput={(e) => update('include_patterns', e.target.value)} placeholder="Comma or space separated" tooltip="Optional module include filters for LoRA" />
							<FormField label="Exclude Patterns" value={s.exclude_patterns || ''} oninput={(e) => update('exclude_patterns', e.target.value)} placeholder="Comma or space separated" tooltip="Optional module exclude filters for LoRA" />
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Prompt">
					<div class="space-y-2 pt-2">
						<!-- svelte-ignore a11y_label_has_associated_control -->
						<label class="block">
							<span class="block text-[11px] font-medium mb-1" style="color: var(--text-muted);">Prompt</span>
							<textarea
								class="w-full text-[12px] px-3 py-2 resize-y"
								rows="3"
								value={s.prompt || ''}
								oninput={(e) => update('prompt', e.target.value)}
								placeholder="Describe what you want to generate..."
								style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
								onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
								onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
							></textarea>
						</label>
						<!-- svelte-ignore a11y_label_has_associated_control -->
						<label class="block">
							<span class="block text-[11px] font-medium mb-1" style="color: var(--text-muted);">Negative Prompt</span>
							<textarea
								class="w-full text-[12px] px-3 py-2 resize-y"
								rows="2"
								value={s.negative_prompt || ''}
								oninput={(e) => update('negative_prompt', e.target.value)}
								placeholder="Optional negative prompt..."
								style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
								onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
								onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
							></textarea>
						</label>
						<PathInput label="Prompts File" value={s.from_file || ''} oninput={(e) => update('from_file', e.target.value)} showFiles tooltip="Text file with prompts (one per line, overrides prompt field)" />
						{#if $advancedMode}
							<div class="flex flex-wrap gap-x-4 gap-y-1">
								<FormToggle label="Use Precached Prompts" checked={s.use_precached_sample_prompts ?? false} onchange={(e) => update('use_precached_sample_prompts', e.target.checked)} tooltip="Read prompt embeddings from cache instead of encoding text prompts" />
							</div>
							<PathInput label="Prompts Cache" value={s.sample_prompts_cache || ''} oninput={(e) => update('sample_prompts_cache', e.target.value)} tooltip="Path to precached prompt embeddings" />
						{/if}
					</div>
				</FormGroup>
			</div>

			<!-- Right column -->
			<div class="space-y-3">
				<FormGroup title="Generation">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="Preset" value={s.sampling_preset || 'defaults'} options={[
								{ value: 'legacy', label: 'Legacy' },
								{ value: 'defaults', label: 'Defaults' },
								{ value: 'ltx20', label: 'LTX 2.0' },
								{ value: 'ltx23', label: 'LTX 2.3' },
								{ value: 'ltx23_hq', label: 'LTX 2.3 HQ' },
								{ value: 'distilled_two_stage', label: 'Distilled Two-Stage' }
							]} onchange={(e) => update('sampling_preset', e.target.value)} tooltip="Generation preset. Blank numeric fields below inherit from this preset." />
							<FormSelect label="Default Negative" value={s.use_default_negative_prompt === true ? 'true' : s.use_default_negative_prompt === false ? 'false' : ''} options={[
								{ value: '', label: 'Auto' },
								{ value: 'true', label: 'On' },
								{ value: 'false', label: 'Off' }
							]} onchange={(e) => update('use_default_negative_prompt', e.target.value === '' ? null : e.target.value === 'true')} tooltip="Use the built-in negative prompt when the preset enables CFG and no negative prompt is typed." />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Width" type="number" value={s.width ?? ''} oninput={(e) => update('width', e.target.value ? Number(e.target.value) : null)} min={64} step={64} placeholder="Preset" tooltip="Output width override" />
							<FormField label="Height" type="number" value={s.height ?? ''} oninput={(e) => update('height', e.target.value ? Number(e.target.value) : null)} min={64} step={64} placeholder="Preset" tooltip="Output height override" />
							<FormField label="Frames" type="number" value={s.frame_count ?? ''} oninput={(e) => update('frame_count', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Preset" tooltip="Frame count override" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Frame Rate" type="number" value={s.frame_rate ?? ''} oninput={(e) => update('frame_rate', e.target.value ? Number(e.target.value) : null)} min={1} step={1} placeholder="Preset" tooltip="Frame rate override" />
							<FormField label="Steps" type="number" value={s.sample_steps ?? ''} oninput={(e) => update('sample_steps', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Preset" tooltip="Sampling steps override" />
							<FormField label="Seed" type="number" value={s.seed ?? ''} oninput={(e) => update('seed', e.target.value ? Number(e.target.value) : null)} placeholder="Random" tooltip="Random seed" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Guidance" type="number" value={s.guidance_scale ?? ''} oninput={(e) => update('guidance_scale', e.target.value ? Number(e.target.value) : null)} step="0.1" min={0} placeholder="Preset" tooltip="Guidance scale override" />
							<FormField label="CFG Scale" type="number" value={s.cfg_scale ?? ''} oninput={(e) => update('cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.1" tooltip="Classifier-free guidance (optional)" />
							<FormField label="Flow Shift" type="number" value={s.discrete_flow_shift ?? 5.0} oninput={(e) => update('discrete_flow_shift', Number(e.target.value))} step="0.1" tooltip="Discrete flow shift" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Video CFG" type="number" value={s.video_cfg_scale ?? ''} oninput={(e) => update('video_cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video CFG scale override" />
							<FormField label="Audio CFG" type="number" value={s.audio_cfg_scale ?? ''} oninput={(e) => update('audio_cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio CFG scale override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="STG Scale" type="number" value={s.stg_scale ?? ''} oninput={(e) => update('stg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Spatiotemporal guidance scale override" />
							<FormSelect label="STG Mode" value={s.stg_mode || ''} options={[{ value: '', label: 'Preset' }, 'video', 'audio', 'both']} onchange={(e) => update('stg_mode', e.target.value || null)} tooltip="STG application mode override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="STG Blocks" value={s.stg_blocks || ''} oninput={(e) => update('stg_blocks', e.target.value)} placeholder="Comma or space separated" tooltip="Blocks targeted by STG" />
							<FormField label="Rescale Scale" type="number" value={s.rescale_scale ?? ''} oninput={(e) => update('rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Guidance rescale override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Video Rescale" type="number" value={s.video_rescale_scale ?? ''} oninput={(e) => update('video_rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video CFG rescale override" />
							<FormField label="Audio Rescale" type="number" value={s.audio_rescale_scale ?? ''} oninput={(e) => update('audio_rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio CFG rescale override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField label="Video Modality" type="number" value={s.video_modality_scale ?? ''} oninput={(e) => update('video_modality_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video A2V modality guidance override" />
							<FormField label="Audio Modality" type="number" value={s.audio_modality_scale ?? ''} oninput={(e) => update('audio_modality_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio V2A modality guidance override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect label="AV Bimodal CFG" value={s.av_bimodal_cfg === true ? 'true' : s.av_bimodal_cfg === false ? 'false' : ''} options={[{ value: '', label: 'Preset' }, { value: 'true', label: 'On' }, { value: 'false', label: 'Off' }]} onchange={(e) => update('av_bimodal_cfg', e.target.value === '' ? null : e.target.value === 'true')} tooltip="Cross-modal CFG mode override" />
							<FormField label="AV Bimodal Scale" type="number" value={s.av_bimodal_scale ?? ''} oninput={(e) => update('av_bimodal_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="AV bimodal CFG scale override" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Memory">
					<div class="space-y-2 pt-2">
						<FormToggle label="Offloading" checked={s.offloading ?? false} onchange={(e) => update('offloading', e.target.checked)} tooltip="Emit `--sample_with_offloading` to reduce VRAM pressure during inference" />
						<FormField label="Blocks to Swap" type="number" value={s.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0-40" min={0} max={40} tooltip="Number of DiT blocks swapped to CPU" />
						{#if $advancedMode}
							<div class="flex flex-wrap gap-x-4 gap-y-1">
								<FormToggle label="Pinned Swap Memory" checked={s.use_pinned_memory_for_block_swap ?? false} onchange={(e) => update('use_pinned_memory_for_block_swap', e.target.checked)} tooltip="Use pinned CPU memory for block swapping" />
								<FormToggle label="Disable FlashAttn in VAE" checked={s.sample_disable_flash_attn ?? false} onchange={(e) => update('sample_disable_flash_attn', e.target.checked)} tooltip="Disable FlashAttention in VAE decode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField label="Split Attn Target" value={s.split_attn_target || ''} oninput={(e) => update('split_attn_target', e.target.value)} placeholder="Comma or space separated" tooltip="Attention modules to chunk" />
								<FormField label="Split Attn Mode" value={s.split_attn_mode || ''} oninput={(e) => update('split_attn_mode', e.target.value)} placeholder="row or batch" tooltip="Split attention mode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField label="Split Attn Chunk" type="number" value={s.split_attn_chunk_size ?? 0} oninput={(e) => update('split_attn_chunk_size', Number(e.target.value || 0))} min={0} tooltip="Attention chunk size" />
								<FormField label="FFN Chunk Target" value={s.ffn_chunk_target || ''} oninput={(e) => update('ffn_chunk_target', e.target.value)} placeholder="Comma or space separated" tooltip="FFN modules to chunk" />
							</div>
							<FormField label="FFN Chunk Size" type="number" value={s.ffn_chunk_size ?? 0} oninput={(e) => update('ffn_chunk_size', Number(e.target.value || 0))} min={0} tooltip="FFN chunk size" />
						{/if}
					</div>
				</FormGroup>

				{#if $advancedMode}
					<FormGroup title="Reference Conditioning">
						<div class="space-y-2 pt-2">
							<div class="grid grid-cols-2 gap-2">
								<PathInput label="Reference Image" value={s.reference_image || ''} oninput={(e) => update('reference_image', e.target.value)} showFiles tooltip="Global I2V reference image" />
								<PathInput label="Reference Video" value={s.reference_video || ''} oninput={(e) => update('reference_video', e.target.value)} showFiles tooltip="Global V2V reference video" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField label="Reference Downscale" type="number" value={s.reference_downscale ?? 1} oninput={(e) => update('reference_downscale', Number(e.target.value || 1))} min={1} tooltip="Downscale factor for reference conditioning" />
								<FormField label="Reference Frames" type="number" value={s.reference_frames ?? 1} oninput={(e) => update('reference_frames', Number(e.target.value || 1))} min={1} tooltip="Number of V2V reference frames" />
							</div>
							<div class="flex flex-wrap gap-x-4 gap-y-1">
								<FormToggle label="I2V Timestep Mask" checked={s.sample_i2v_token_timestep_mask ?? true} onchange={(e) => update('sample_i2v_token_timestep_mask', e.target.checked)} tooltip="Enable I2V token timestep masking" />
								<FormToggle label="Include Reference" checked={s.sample_include_reference ?? false} onchange={(e) => update('sample_include_reference', e.target.checked)} tooltip="Append reference media to output preview" />
							</div>
						</div>
					</FormGroup>
				{/if}

				{#if $advancedMode}
					<FormGroup title="Decode">
						<div class="space-y-2 pt-2">
							<div class="flex flex-wrap gap-x-4 gap-y-1">
								<FormToggle label="Disable Audio" checked={s.sample_disable_audio ?? false} onchange={(e) => update('sample_disable_audio', e.target.checked)} tooltip="Skip audio decode in AV mode" />
								<FormToggle label="Audio Only" checked={s.sample_audio_only ?? false} onchange={(e) => update('sample_audio_only', e.target.checked)} tooltip="Output audio only" />
								<FormToggle label="Merge Audio" checked={s.sample_merge_audio ?? false} onchange={(e) => update('sample_merge_audio', e.target.checked)} tooltip="Mux generated audio into video output" />
								<FormToggle label="Two Stage" checked={s.sample_two_stage ?? false} onchange={(e) => update('sample_two_stage', e.target.checked)} tooltip="Enable spatial upsampler second stage" />
								<FormToggle label="Tiled VAE" checked={s.sample_tiled_vae ?? false} onchange={(e) => update('sample_tiled_vae', e.target.checked)} tooltip="Enable tiled VAE decode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<PathInput label="Spatial Upsampler" value={s.spatial_upsampler_path || ''} oninput={(e) => update('spatial_upsampler_path', e.target.value)} showFiles tooltip="Spatial upsampler checkpoint for two-stage decode" />
								<PathInput label="Distilled LoRA" value={s.distilled_lora_path || ''} oninput={(e) => update('distilled_lora_path', e.target.value)} showFiles tooltip="Distilled LoRA for stage two" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField label="Stage2 Steps" type="number" value={s.sample_stage2_steps ?? 3} oninput={(e) => update('sample_stage2_steps', Number(e.target.value || 3))} min={1} tooltip="Second-stage denoising steps" />
								<FormField label="Tile Size" type="number" value={s.sample_vae_tile_size ?? 512} oninput={(e) => update('sample_vae_tile_size', Number(e.target.value || 512))} min={1} tooltip="Spatial tile size" />
								<FormField label="Tile Overlap" type="number" value={s.sample_vae_tile_overlap ?? 64} oninput={(e) => update('sample_vae_tile_overlap', Number(e.target.value || 64))} min={0} tooltip="Spatial tile overlap" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField label="Temporal Tile Size" type="number" value={s.sample_vae_temporal_tile_size ?? 0} oninput={(e) => update('sample_vae_temporal_tile_size', Number(e.target.value || 0))} min={0} tooltip="Temporal tile size, 0 disables temporal tiling" />
								<FormField label="Temporal Overlap" type="number" value={s.sample_vae_temporal_tile_overlap ?? 8} oninput={(e) => update('sample_vae_temporal_tile_overlap', Number(e.target.value || 8))} min={0} tooltip="Temporal tile overlap" />
							</div>
						</div>
					</FormGroup>
				{/if}

				<FormGroup title="Output">
					<div class="space-y-2 pt-2">
						<PathInput label="Output Dir" value={s.output_dir || 'output'} oninput={(e) => update('output_dir', e.target.value)} tooltip="Output directory" />
						<FormField label="Name" value={s.output_name || 'ltx2_sample'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix" />
						{#if $advancedMode}
							<div class="flex flex-wrap gap-x-4 gap-y-1">
								<FormToggle label="Use Precached Latents" checked={s.use_precached_sample_latents ?? false} onchange={(e) => update('use_precached_sample_latents', e.target.checked)} tooltip="Use precached latent tensors for inference" />
							</div>
							<PathInput label="Latents Cache" value={s.sample_latents_cache || ''} oninput={(e) => update('sample_latents_cache', e.target.value)} tooltip="Path to precached latent tensors" />
						{/if}
					</div>
				</FormGroup>
			</div>
		</div>

		<!-- Controls -->
		<div class="py-4">
			<ProcessControls processType="inference" status={inferenceStatus} onStart={() => startProcess('inference')} onStop={() => stopProcess('inference')} />
		</div>

		<ProcessConsole lines={inferenceLogs} processType="inference" />
		<CommandPanel processType="inference" defaultFilename="inference.bat" />
	</div>
{/if}
