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
	import { defaultModelDir, describeExactModelScan, effectiveGemmaRoot, effectiveGemmaSafetensors, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { getModelDownloadPresets, checkPathExists, scanCheckpointsWithProgress, cancelCheckpointScan, formatCheckpointScanStatus, modelDownloadTooltip } from '$lib/utils/modelDownloads.js';
	import { cancelSharedModelDownload, modelDownloadState, resumeModelDownloadPolling, startSharedModelDownload } from '$lib/stores/modelDownloads.js';
	import { projectConfig, projectLoaded, updateSection, saveProjectNow } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, preloadLogsIfActive, startLogPolling } from '$lib/stores/processes.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import { onMount } from 'svelte';

	let cwd = $state('');
	let downloadPresets = $state({});
	let ltxDownloadExists = $state(false);
	let gemmaDownloadExists = $state(false);
	let gemmaSafetensorsExists = $state(false);
	let foundLtxPath = $state('');
	let foundGemmaPath = $state('');
	let foundGemmaSafetensorsPath = $state('');
	let scanningLtx = $state(false);
	let scanningGemma = $state(false);
	let scanningGemmaSafetensors = $state(false);
	let ltxScanMessage = $state('');
	let ltxScanTone = $state('muted');
	let gemmaScanMessage = $state('');
	let gemmaScanTone = $state('muted');
	let gemmaSafetensorsScanMessage = $state('');
	let gemmaSafetensorsScanTone = $state('muted');
	let ltxScanJobId = $state('');
	let gemmaScanJobId = $state('');
	let gemmaSafetensorsScanJobId = $state('');

	onMount(() => {
		fetch('/api/fs/cwd').then((res) => res.ok ? res.json() : null).then((data) => { cwd = data?.cwd || ''; }).catch(() => {});
		getModelDownloadPresets().then((presets) => { downloadPresets = presets; }).catch(() => {});
		resumeModelDownloadPolling();
		preloadLogsIfActive('inference');
		const logInterval = startLogPolling('inference', 1000);
		return () => {
			clearInterval(logInterval);
			if (ltxScanJobId) cancelCheckpointScan(ltxScanJobId).catch(() => {});
			if (gemmaScanJobId) cancelCheckpointScan(gemmaScanJobId).catch(() => {});
			if (gemmaSafetensorsScanJobId) cancelCheckpointScan(gemmaSafetensorsScanJobId).catch(() => {});
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
	let scanTargetGemmaRoot = $derived(effectiveGemmaRoot(cwd, $projectConfig, s.gemma_root || '', ''));
	let downloadState = $derived($modelDownloadState.state || '');
	let modelStatus = $derived($modelDownloadState.message || '');
	let modelStatusTone = $derived($modelDownloadState.tone || 'muted');
	let hasActiveDownload = $derived(Boolean($modelDownloadState.jobId) && ['queued', 'running', 'cancelling'].includes(downloadState));

	function relatedScanTargets() {
		return {
			ltx2: resolvedLtx,
			gemma: scanTargetGemmaRoot,
			gemma_safetensors: activeGemmaSafetensors
		};
	}

	$effect(() => {
		const path = resolvedLtx;
		foundLtxPath = '';
		ltxScanMessage = '';
		let cancelled = false;
		checkPathExists(path).then((exists) => { if (!cancelled) ltxDownloadExists = exists; }).catch(() => { if (!cancelled) ltxDownloadExists = false; });
		return () => { cancelled = true; };
	});

	$effect(() => {
		const path = resolvedGemma;
		foundGemmaPath = '';
		gemmaScanMessage = '';
		let cancelled = false;
		checkPathExists(path).then((exists) => { if (!cancelled) gemmaDownloadExists = exists; }).catch(() => { if (!cancelled) gemmaDownloadExists = false; });
		return () => { cancelled = true; };
	});

	$effect(() => {
		const path = activeGemmaSafetensors;
		foundGemmaSafetensorsPath = '';
		gemmaSafetensorsScanMessage = '';
		if (!path) {
			gemmaSafetensorsExists = false;
			return;
		}
		let cancelled = false;
		checkPathExists(path).then((exists) => { if (!cancelled) gemmaSafetensorsExists = exists; }).catch(() => { if (!cancelled) gemmaSafetensorsExists = false; });
		return () => { cancelled = true; };
	});

	async function scanLtx() {
		if (scanningLtx) return;
		if (!cwd) {
			ltxScanMessage = 'Working directory not loaded yet';
			ltxScanTone = 'danger';
			return;
		}
		scanningLtx = true;
		foundLtxPath = '';
		ltxScanMessage = '';
		try {
			const status = await scanCheckpointsWithProgress('ltx2', modelDir, resolvedLtx, (scanStatus) => {
				ltxScanJobId = scanStatus.job_id || ltxScanJobId;
				ltxScanMessage = formatCheckpointScanStatus(scanStatus);
				ltxScanTone = scanStatus.state === 'failed' ? 'danger' : 'muted';
			}, relatedScanTargets());
			if (status.state === 'completed') {
				const result = describeExactModelScan(status.results || [], resolvedLtx);
				foundLtxPath = result.match;
				ltxScanMessage = result.message;
				ltxScanTone = result.tone;
			}
		} catch (e) {
			foundLtxPath = '';
			ltxScanMessage = e?.message || 'Scan failed';
			ltxScanTone = 'danger';
		} finally {
			scanningLtx = false;
			ltxScanJobId = '';
		}
	}

	async function scanGemma() {
		if (scanningGemma) return;
		if (!cwd) {
			gemmaScanMessage = 'Working directory not loaded yet';
			gemmaScanTone = 'danger';
			return;
		}
		scanningGemma = true;
		foundGemmaPath = '';
		gemmaScanMessage = '';
		try {
			const status = await scanCheckpointsWithProgress('gemma', modelDir, scanTargetGemmaRoot, (scanStatus) => {
				gemmaScanJobId = scanStatus.job_id || gemmaScanJobId;
				gemmaScanMessage = formatCheckpointScanStatus(scanStatus);
				gemmaScanTone = scanStatus.state === 'failed' ? 'danger' : 'muted';
			}, relatedScanTargets());
			if (status.state === 'completed') {
				const result = describeExactModelScan(status.results || [], scanTargetGemmaRoot);
				foundGemmaPath = result.match;
				gemmaScanMessage = result.message;
				gemmaScanTone = result.tone;
			}
		} catch (e) {
			foundGemmaPath = '';
			gemmaScanMessage = e?.message || 'Scan failed';
			gemmaScanTone = 'danger';
		} finally {
			scanningGemma = false;
			gemmaScanJobId = '';
		}
	}

	async function scanGemmaSafetensors() {
		if (scanningGemmaSafetensors) return;
		if (!activeGemmaSafetensors) {
			gemmaSafetensorsScanMessage = 'Set Gemma Safetensors path first';
			gemmaSafetensorsScanTone = 'danger';
			return;
		}
		scanningGemmaSafetensors = true;
		foundGemmaSafetensorsPath = '';
		gemmaSafetensorsScanMessage = '';
		try {
			const status = await scanCheckpointsWithProgress('gemma_safetensors', modelDir, activeGemmaSafetensors, (scanStatus) => {
				gemmaSafetensorsScanJobId = scanStatus.job_id || gemmaSafetensorsScanJobId;
				gemmaSafetensorsScanMessage = formatCheckpointScanStatus(scanStatus);
				gemmaSafetensorsScanTone = scanStatus.state === 'failed' ? 'danger' : 'muted';
			}, relatedScanTargets());
			if (status.state === 'completed') {
				const result = describeExactModelScan(status.results || [], activeGemmaSafetensors);
				foundGemmaSafetensorsPath = result.match;
				gemmaSafetensorsScanMessage = result.message;
				gemmaSafetensorsScanTone = result.tone;
			}
		} catch (e) {
			foundGemmaSafetensorsPath = '';
			gemmaSafetensorsScanMessage = e?.message || 'Scan failed';
			gemmaSafetensorsScanTone = 'danger';
		} finally {
			scanningGemmaSafetensors = false;
			gemmaSafetensorsScanJobId = '';
		}
	}

	async function stopLtxScan() {
		if (!ltxScanJobId) return;
		try {
			const status = await cancelCheckpointScan(ltxScanJobId);
			ltxScanMessage = formatCheckpointScanStatus(status);
		} catch (e) {
			ltxScanMessage = e?.message || 'Cancel failed';
			ltxScanTone = 'danger';
		}
	}

	async function stopGemmaScan() {
		if (!gemmaScanJobId) return;
		try {
			const status = await cancelCheckpointScan(gemmaScanJobId);
			gemmaScanMessage = formatCheckpointScanStatus(status);
		} catch (e) {
			gemmaScanMessage = e?.message || 'Cancel failed';
			gemmaScanTone = 'danger';
		}
	}

	async function stopGemmaSafetensorsScan() {
		if (!gemmaSafetensorsScanJobId) return;
		try {
			const status = await cancelCheckpointScan(gemmaSafetensorsScanJobId);
			gemmaSafetensorsScanMessage = formatCheckpointScanStatus(status);
		} catch (e) {
			gemmaSafetensorsScanMessage = e?.message || 'Cancel failed';
			gemmaSafetensorsScanTone = 'danger';
		}
	}

	async function downloadModel(preset) {
		if (hasActiveDownload) return;
		const targetPath = preset === 'ltxav' ? resolvedLtx : resolvedGemma;
		if (!targetPath) return;
		projectConfig.update((config) => config ? { ...config, model_dir: modelDir } : config);
		await saveProjectNow();
		await startSharedModelDownload({ preset, targetPath, modelDir, section: 'inference' });
	}

	async function stopDownload() {
		await cancelSharedModelDownload();
	}
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-4">
		<!-- Two-column layout -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
			<!-- Left column -->
			<div class="space-y-3">
				<FormGroup title="Model">
					<div class="space-y-2 pt-2">
						<CheckpointInput fieldPath="inference.ltx2_checkpoint" label="LTX-2 Checkpoint" value={s.ltx2_checkpoint || ''} onchange={(v) => update('ltx2_checkpoint', v)} showFiles tooltip="Path to LTX-2 checkpoint" actionLabel="D" actionBusyLabel="..." actionDisabled={hasActiveDownload || ltxDownloadExists} actionTooltip={modelDownloadTooltip(downloadPresets, 'ltxav', resolvedLtx, ltxDownloadExists)} onaction={() => downloadModel('ltxav')} />
						<ModelPathStatus exists={ltxDownloadExists} foundPath={foundLtxPath} disabled={hasActiveDownload} scanning={scanningLtx} scanMessage={ltxScanMessage} scanTone={ltxScanTone} onscan={scanLtx} oncancel={stopLtxScan} onusefound={(path) => update('ltx2_checkpoint', path)} />
						<PathInput fieldPath="inference.vae" value={s.vae || ''} oninput={(e) => update('vae', e.target.value)} showFiles tooltip="Optional separate VAE checkpoint. Leave blank to reuse the LTX-2 checkpoint." />
						<CheckpointInput fieldPath="inference.gemma_root" label="Gemma Root" value={s.gemma_root || ''} onchange={(v) => update('gemma_root', v)} disabled={gemmaRootDisabled} tooltip={gemmaRootDisabled ? 'Ignored while Gemma Safetensors is set' : 'Gemma text encoder directory'} actionLabel="D" actionBusyLabel="..." actionDisabled={gemmaRootDisabled || hasActiveDownload || gemmaDownloadExists} actionTooltip={gemmaRootDisabled ? 'Gemma Safetensors is active' : modelDownloadTooltip(downloadPresets, 'gemma-unsloth', resolvedGemma, gemmaDownloadExists)} onaction={() => downloadModel('gemma-unsloth')} />
						<ModelPathStatus exists={gemmaRootDisabled || gemmaDownloadExists} foundPath={foundGemmaPath} disabled={gemmaRootDisabled || hasActiveDownload} scanning={scanningGemma} scanMessage={gemmaScanMessage} scanTone={gemmaScanTone} onscan={scanGemma} oncancel={stopGemmaScan} onusefound={(path) => { update('gemma_root', path); update('gemma_safetensors', ''); }} />
						<PathInput fieldPath="inference.gemma_safetensors" value={s.gemma_safetensors || ''} oninput={(e) => update('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
						{#if activeGemmaSafetensors}
							<ModelPathStatus exists={gemmaSafetensorsExists} foundPath={foundGemmaSafetensorsPath} disabled={hasActiveDownload} scanning={scanningGemmaSafetensors} scanMessage={gemmaSafetensorsScanMessage} scanTone={gemmaSafetensorsScanTone} onscan={scanGemmaSafetensors} oncancel={stopGemmaSafetensorsScan} onusefound={(path) => update('gemma_safetensors', path)} />
						{/if}
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
							<FormField fieldPath="inference.device" value={s.device || ''} oninput={(e) => update('device', e.target.value)} placeholder="Auto" tooltip="Force `cpu` or `cuda`" />
							<FormSelect fieldPath="inference.ltx2_mode" value={s.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} tooltip="Video/AV/Audio" />
							<FormSelect fieldPath="inference.mixed_precision" value={s.mixed_precision || 'bf16'} options={['no', 'fp16', 'bf16']} onchange={(e) => update('mixed_precision', e.target.value)} tooltip="Mixed precision mode" />
							<FormSelect fieldPath="inference.attn_mode" value={s.attn_mode || 'torch'} options={['torch', 'xformers', 'flash', 'flash3', 'sdpa', 'sage']} onchange={(e) => update('attn_mode', e.target.value)} tooltip="Attention implementation" />
						</div>
						{#if $advancedMode}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="inference.vae_dtype" value={s.vae_dtype || ''} options={[{ value: '', label: 'Default' }, 'bfloat16', 'float16', 'float32']} onchange={(e) => update('vae_dtype', e.target.value || null)} tooltip="Optional dtype override for a separate VAE checkpoint" />
							</div>
						{/if}
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="inference.fp8_base" checked={s.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision (VRAM savings)" />
							<FormToggle fieldPath="inference.fp8_scaled" checked={s.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} tooltip="Scaled FP8 for stability" />
							<FormToggle fieldPath="inference.gemma_load_in_8bit" checked={s.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
							<FormToggle fieldPath="inference.gemma_load_in_4bit" checked={s.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
							<FormToggle fieldPath="inference.gemma_fp8_weight_offload" checked={s.gemma_fp8_weight_offload ?? true} onchange={(e) => update('gemma_fp8_weight_offload', e.target.checked)} tooltip="For FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. Disable this to keep more weights on VRAM and reduce RAM/pagefile pressure." />
						</div>
						{#if $advancedMode}
							<FormField fieldPath="inference.fp8_keep_blocks" value={s.fp8_keep_blocks || ''} oninput={(e) => update('fp8_keep_blocks', e.target.value)} placeholder="0,1,2,45" tooltip="Transformer block indices to keep in high precision when FP8 Scaled is enabled. Ranges like 0-2,45 are accepted." />
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="inference.gemma_bnb_4bit_quant_type" value={s.gemma_bnb_4bit_quant_type || 'nf4'} options={['nf4', 'fp4']} onchange={(e) => update('gemma_bnb_4bit_quant_type', e.target.value)} tooltip="bitsandbytes 4-bit quant type" />
								<div class="flex items-end">
									<FormToggle fieldPath="inference.gemma_bnb_4bit_disable_double_quant" checked={s.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => update('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable bitsandbytes nested quantization" />
								</div>
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField fieldPath="inference.w8a8_mode" value={s.w8a8_mode || 'int8'} oninput={(e) => update('w8a8_mode', e.target.value)} placeholder="int8 or fp8" tooltip="W8A8 quantization mode" />
								<FormField type="number" fieldPath="inference.network_dim" value={s.network_dim ?? 0} oninput={(e) => update('network_dim', Number(e.target.value || 0))} min={0} tooltip="LoRA rank for LoftQ initialization" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="inference.nf4_block_size" value={s.nf4_block_size ?? 64} oninput={(e) => update('nf4_block_size', Number(e.target.value || 64))} min={1} tooltip="NF4 block size" />
								<FormField type="number" fieldPath="inference.loftq_iters" value={s.loftq_iters ?? 2} oninput={(e) => update('loftq_iters', Number(e.target.value || 2))} min={1} tooltip="LoftQ iteration count" />
								<FormField type="number" fieldPath="inference.awq_num_batches" value={s.awq_num_batches ?? 8} oninput={(e) => update('awq_num_batches', Number(e.target.value || 8))} min={1} tooltip="AWQ calibration batches" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="inference.awq_alpha" value={s.awq_alpha ?? 0.25} oninput={(e) => update('awq_alpha', Number(e.target.value || 0.25))} step="0.01" tooltip="AWQ alpha" />
								<FormField type="number" fieldPath="inference.fp8_upcast_seed" value={s.fp8_upcast_seed ?? 0} oninput={(e) => update('fp8_upcast_seed', Number(e.target.value || 0))} min={0} tooltip="Seed for stochastic FP8 upcast" />
							</div>
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="inference.flash_attn" checked={s.flash_attn ?? false} onchange={(e) => update('flash_attn', e.target.checked)} tooltip="Emit `--flash_attn`" />
								<FormToggle fieldPath="inference.flash3" checked={s.flash3 ?? false} onchange={(e) => update('flash3', e.target.checked)} tooltip="Emit `--flash3`" />
								<FormToggle fieldPath="inference.sdpa" checked={s.sdpa ?? false} onchange={(e) => update('sdpa', e.target.checked)} tooltip="Emit `--sdpa`" />
								<FormToggle fieldPath="inference.xformers" checked={s.xformers ?? false} onchange={(e) => update('xformers', e.target.checked)} tooltip="Emit `--xformers`" />
								<FormToggle fieldPath="inference.fp8_w8a8" checked={s.fp8_w8a8 ?? false} onchange={(e) => update('fp8_w8a8', e.target.checked)} tooltip="Use W8A8 quantization for DiT" />
								<FormToggle fieldPath="inference.nf4_base" checked={s.nf4_base ?? false} onchange={(e) => update('nf4_base', e.target.checked)} tooltip="Use NF4 quantization for DiT" />
								<FormToggle fieldPath="inference.loftq_init" checked={s.loftq_init ?? false} onchange={(e) => update('loftq_init', e.target.checked)} tooltip="Initialize DiT quantization with LoftQ" />
								<FormToggle fieldPath="inference.awq_calibration" checked={s.awq_calibration ?? false} onchange={(e) => update('awq_calibration', e.target.checked)} tooltip="Run AWQ calibration" />
								<FormToggle fieldPath="inference.fp8_upcast" checked={s.fp8_upcast ?? false} onchange={(e) => update('fp8_upcast', e.target.checked)} tooltip="Upcast FP8 quantization during load" />
								<FormToggle fieldPath="inference.fp8_upcast_stochastic" checked={s.fp8_upcast_stochastic ?? false} onchange={(e) => update('fp8_upcast_stochastic', e.target.checked)} tooltip="Use stochastic FP8 upcast" />
							</div>
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="LoRA">
					<div class="space-y-2 pt-2">
						<PathInput fieldPath="inference.lora_weight" value={s.lora_weight || ''} oninput={(e) => update('lora_weight', e.target.value)} showFiles tooltip="Path to LoRA safetensors file" />
						<FormField type="number" fieldPath="inference.lora_multiplier" value={s.lora_multiplier ?? 1.0} oninput={(e) => update('lora_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="LoRA weight multiplier" />
						{#if $advancedMode}
							<FormField fieldPath="inference.include_patterns" value={s.include_patterns || ''} oninput={(e) => update('include_patterns', e.target.value)} placeholder="Comma or space separated" tooltip="Optional module include filters for LoRA" />
							<FormField fieldPath="inference.exclude_patterns" value={s.exclude_patterns || ''} oninput={(e) => update('exclude_patterns', e.target.value)} placeholder="Comma or space separated" tooltip="Optional module exclude filters for LoRA" />
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
						<PathInput fieldPath="inference.from_file" value={s.from_file || ''} oninput={(e) => update('from_file', e.target.value)} showFiles tooltip="Text file with prompts (one per line, overrides prompt field)" />
						{#if $advancedMode}
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="inference.use_precached_sample_prompts" checked={s.use_precached_sample_prompts ?? false} onchange={(e) => update('use_precached_sample_prompts', e.target.checked)} tooltip="Read prompt embeddings from cache instead of encoding text prompts" />
							</div>
							<PathInput fieldPath="inference.sample_prompts_cache" value={s.sample_prompts_cache || ''} oninput={(e) => update('sample_prompts_cache', e.target.value)} tooltip="Path to precached prompt embeddings" />
						{/if}
					</div>
				</FormGroup>
			</div>

			<!-- Right column -->
			<div class="space-y-3">
				<FormGroup title="Generation">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="inference.sampling_preset" value={s.sampling_preset || 'defaults'} options={[
								{ value: 'legacy', label: 'Legacy' },
								{ value: 'defaults', label: 'Defaults' },
								{ value: 'ltx20', label: 'LTX 2.0' },
								{ value: 'ltx23', label: 'LTX 2.3' },
								{ value: 'ltx23_hq', label: 'LTX 2.3 HQ' },
								{ value: 'distilled_two_stage', label: 'Distilled Two-Stage' }
							]} onchange={(e) => update('sampling_preset', e.target.value)} tooltip="Generation preset. Blank numeric fields below inherit from this preset." />
							<FormSelect fieldPath="inference.use_default_negative_prompt" value={s.use_default_negative_prompt === true ? 'true' : s.use_default_negative_prompt === false ? 'false' : ''} options={[
								{ value: '', label: 'Auto' },
								{ value: 'true', label: 'On' },
								{ value: 'false', label: 'Off' }
							]} onchange={(e) => update('use_default_negative_prompt', e.target.value === '' ? null : e.target.value === 'true')} tooltip="Use the built-in negative prompt when the preset enables CFG and no negative prompt is typed." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="inference.sample_sampler" value={s.sample_sampler || 'auto'} options={[
								{ value: 'auto', label: 'Auto' },
								{ value: 'res_2s', label: 'RES 2S' },
								{ value: 'euler', label: 'Euler' }
							]} onchange={(e) => update('sample_sampler', e.target.value)} tooltip="Denoising sampler. Auto uses RES 2S for full LTX presets and Euler for distilled two-stage." />
							<FormSelect fieldPath="inference.sample_sigma_schedule" value={s.sample_sigma_schedule || 'auto'} options={[
								{ value: 'auto', label: 'Auto' },
								{ value: 'ltx', label: 'LTX Shifted' },
								{ value: 'ltx23_distilled', label: 'LTX 2.3 Distilled' }
							]} onchange={(e) => update('sample_sigma_schedule', e.target.value)} tooltip="Sigma schedule. Auto uses latent-aware LTX sigmas, with the distilled schedule for distilled presets." />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="inference.width" value={s.width ?? ''} oninput={(e) => update('width', e.target.value ? Number(e.target.value) : null)} min={64} step={64} placeholder="Preset" tooltip="Output width override" />
							<FormField type="number" fieldPath="inference.height" value={s.height ?? ''} oninput={(e) => update('height', e.target.value ? Number(e.target.value) : null)} min={64} step={64} placeholder="Preset" tooltip="Output height override" />
							<FormField type="number" fieldPath="inference.frame_count" value={s.frame_count ?? ''} oninput={(e) => update('frame_count', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Preset" tooltip="Frame count override" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="inference.frame_rate" value={s.frame_rate ?? ''} oninput={(e) => update('frame_rate', e.target.value ? Number(e.target.value) : null)} min={1} step={1} placeholder="Preset" tooltip="Frame rate override" />
							<FormField type="number" fieldPath="inference.sample_steps" value={s.sample_steps ?? ''} oninput={(e) => update('sample_steps', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Preset" tooltip="Sampling steps override" />
							<FormField type="number" fieldPath="inference.seed" value={s.seed ?? ''} oninput={(e) => update('seed', e.target.value ? Number(e.target.value) : null)} placeholder="Random" tooltip="Random seed" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="inference.guidance_scale" value={s.guidance_scale ?? ''} oninput={(e) => update('guidance_scale', e.target.value ? Number(e.target.value) : null)} step="0.1" min={0} placeholder="Preset" tooltip="Guidance scale override" />
							<FormField type="number" fieldPath="inference.cfg_scale" value={s.cfg_scale ?? ''} oninput={(e) => update('cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.1" tooltip="Classifier-free guidance (optional)" />
							<FormField type="number" fieldPath="inference.discrete_flow_shift" value={s.discrete_flow_shift ?? 5.0} oninput={(e) => update('discrete_flow_shift', Number(e.target.value))} step="0.1" tooltip="Discrete flow shift" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="inference.video_cfg_scale" value={s.video_cfg_scale ?? ''} oninput={(e) => update('video_cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video CFG scale override" />
							<FormField type="number" fieldPath="inference.audio_cfg_scale" value={s.audio_cfg_scale ?? ''} oninput={(e) => update('audio_cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio CFG scale override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="inference.stg_scale" value={s.stg_scale ?? ''} oninput={(e) => update('stg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Spatiotemporal guidance scale override" />
							<FormSelect fieldPath="inference.stg_mode" value={s.stg_mode || ''} options={[{ value: '', label: 'Preset' }, 'video', 'audio', 'both']} onchange={(e) => update('stg_mode', e.target.value || null)} tooltip="STG application mode override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="inference.stg_blocks" value={s.stg_blocks || ''} oninput={(e) => update('stg_blocks', e.target.value)} placeholder="Comma or space separated" tooltip="Blocks targeted by STG" />
							<FormField type="number" fieldPath="inference.rescale_scale" value={s.rescale_scale ?? ''} oninput={(e) => update('rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Guidance rescale override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="inference.video_rescale_scale" value={s.video_rescale_scale ?? ''} oninput={(e) => update('video_rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video CFG rescale override" />
							<FormField type="number" fieldPath="inference.audio_rescale_scale" value={s.audio_rescale_scale ?? ''} oninput={(e) => update('audio_rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio CFG rescale override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="inference.video_modality_scale" value={s.video_modality_scale ?? ''} oninput={(e) => update('video_modality_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video A2V modality guidance override" />
							<FormField type="number" fieldPath="inference.audio_modality_scale" value={s.audio_modality_scale ?? ''} oninput={(e) => update('audio_modality_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio V2A modality guidance override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="inference.av_bimodal_cfg" value={s.av_bimodal_cfg === true ? 'true' : s.av_bimodal_cfg === false ? 'false' : ''} options={[{ value: '', label: 'Preset' }, { value: 'true', label: 'On' }, { value: 'false', label: 'Off' }]} onchange={(e) => update('av_bimodal_cfg', e.target.value === '' ? null : e.target.value === 'true')} tooltip="Cross-modal CFG mode override" />
							<FormField type="number" fieldPath="inference.av_bimodal_scale" value={s.av_bimodal_scale ?? ''} oninput={(e) => update('av_bimodal_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="AV bimodal CFG scale override" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Memory">
					<div class="space-y-2 pt-2">
						<FormToggle fieldPath="inference.offloading" checked={s.offloading ?? false} onchange={(e) => update('offloading', e.target.checked)} tooltip="Emit `--sample_with_offloading` to reduce VRAM pressure during inference" />
						<FormField type="number" fieldPath="inference.blocks_to_swap" value={s.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0-40" min={0} max={40} tooltip="Number of DiT blocks swapped to CPU" />
						{#if $advancedMode}
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="inference.use_pinned_memory_for_block_swap" checked={s.use_pinned_memory_for_block_swap ?? false} onchange={(e) => update('use_pinned_memory_for_block_swap', e.target.checked)} tooltip="Use pinned CPU memory for block swapping" />
								<FormToggle fieldPath="inference.sample_disable_flash_attn" checked={s.sample_disable_flash_attn ?? false} onchange={(e) => update('sample_disable_flash_attn', e.target.checked)} tooltip="Disable FlashAttention in VAE decode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField fieldPath="inference.split_attn_target" value={s.split_attn_target || ''} oninput={(e) => update('split_attn_target', e.target.value)} placeholder="Comma or space separated" tooltip="Attention modules to chunk" />
								<FormField fieldPath="inference.split_attn_mode" value={s.split_attn_mode || ''} oninput={(e) => update('split_attn_mode', e.target.value)} placeholder="row or batch" tooltip="Split attention mode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="inference.split_attn_chunk_size" value={s.split_attn_chunk_size ?? 0} oninput={(e) => update('split_attn_chunk_size', Number(e.target.value || 0))} min={0} tooltip="Attention chunk size" />
								<FormField fieldPath="inference.ffn_chunk_target" value={s.ffn_chunk_target || ''} oninput={(e) => update('ffn_chunk_target', e.target.value)} placeholder="Comma or space separated" tooltip="FFN modules to chunk" />
							</div>
							<FormField type="number" fieldPath="inference.ffn_chunk_size" value={s.ffn_chunk_size ?? 0} oninput={(e) => update('ffn_chunk_size', Number(e.target.value || 0))} min={0} tooltip="FFN chunk size" />
						{/if}
					</div>
				</FormGroup>

				{#if $advancedMode}
					<FormGroup title="Reference Conditioning">
						<div class="space-y-2 pt-2">
							<div class="grid grid-cols-2 gap-2">
								<PathInput fieldPath="inference.reference_image" value={s.reference_image || ''} oninput={(e) => update('reference_image', e.target.value)} showFiles tooltip="Global I2V reference image" />
								<PathInput fieldPath="inference.reference_video" value={s.reference_video || ''} oninput={(e) => update('reference_video', e.target.value)} showFiles tooltip="Global V2V reference video" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="inference.reference_downscale" value={s.reference_downscale ?? 1} oninput={(e) => update('reference_downscale', Number(e.target.value || 1))} min={1} tooltip="Downscale factor for reference conditioning" />
								<FormField type="number" fieldPath="inference.reference_frames" value={s.reference_frames ?? 1} oninput={(e) => update('reference_frames', Number(e.target.value || 1))} min={1} tooltip="Number of V2V reference frames" />
							</div>
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="inference.sample_i2v_token_timestep_mask" checked={s.sample_i2v_token_timestep_mask ?? true} onchange={(e) => update('sample_i2v_token_timestep_mask', e.target.checked)} tooltip="Enable I2V token timestep masking" />
								<FormToggle fieldPath="inference.sample_include_reference" checked={s.sample_include_reference ?? false} onchange={(e) => update('sample_include_reference', e.target.checked)} tooltip="Append reference media to output preview" />
							</div>
						</div>
					</FormGroup>
				{/if}

				{#if $advancedMode}
					<FormGroup title="Decode">
						<div class="space-y-2 pt-2">
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="inference.sample_disable_audio" checked={s.sample_disable_audio ?? false} onchange={(e) => update('sample_disable_audio', e.target.checked)} tooltip="Skip audio decode in AV mode" />
								<FormToggle fieldPath="inference.sample_audio_only" checked={s.sample_audio_only ?? false} onchange={(e) => update('sample_audio_only', e.target.checked)} tooltip="Output audio only" />
								<FormToggle fieldPath="inference.sample_merge_audio" checked={s.sample_merge_audio ?? false} onchange={(e) => update('sample_merge_audio', e.target.checked)} tooltip="Mux generated audio into video output" />
								<FormToggle fieldPath="inference.sample_audio_subprocess" checked={s.sample_audio_subprocess ?? true} onchange={(e) => update('sample_audio_subprocess', e.target.checked)} tooltip="Decode audio in a subprocess" />
								<FormToggle fieldPath="inference.sample_two_stage" checked={s.sample_two_stage ?? false} onchange={(e) => update('sample_two_stage', e.target.checked)} tooltip="Enable spatial upsampler second stage" />
								<FormToggle fieldPath="inference.sample_tiled_vae" checked={s.sample_tiled_vae ?? false} onchange={(e) => update('sample_tiled_vae', e.target.checked)} tooltip="Enable tiled VAE decode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<PathInput fieldPath="inference.spatial_upsampler_path" value={s.spatial_upsampler_path || ''} oninput={(e) => update('spatial_upsampler_path', e.target.value)} showFiles tooltip="Spatial upsampler checkpoint for two-stage decode" />
								<PathInput fieldPath="inference.distilled_lora_path" value={s.distilled_lora_path || ''} oninput={(e) => update('distilled_lora_path', e.target.value)} showFiles tooltip="Distilled LoRA for stage two" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="inference.sample_stage2_steps" value={s.sample_stage2_steps ?? 3} oninput={(e) => update('sample_stage2_steps', Number(e.target.value || 3))} min={1} tooltip="Second-stage denoising steps" />
								<FormField type="number" fieldPath="inference.sample_stage1_distilled_lora_multiplier" value={s.sample_stage1_distilled_lora_multiplier ?? ''} oninput={(e) => update('sample_stage1_distilled_lora_multiplier', e.target.value ? Number(e.target.value) : null)} min={0} step="0.05" placeholder="Auto" tooltip="Distilled LoRA multiplier for stage 1. Auto uses 0.25 with RES 2S." />
								<FormField type="number" fieldPath="inference.sample_stage2_distilled_lora_multiplier" value={s.sample_stage2_distilled_lora_multiplier ?? ''} oninput={(e) => update('sample_stage2_distilled_lora_multiplier', e.target.value ? Number(e.target.value) : null)} min={0} step="0.05" placeholder="Auto" tooltip="Distilled LoRA multiplier for stage 2. Auto uses 0.5 with RES 2S." />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="inference.sample_vae_tile_size" value={s.sample_vae_tile_size ?? 512} oninput={(e) => update('sample_vae_tile_size', Number(e.target.value || 512))} min={1} tooltip="Spatial tile size" />
								<FormField type="number" fieldPath="inference.sample_vae_tile_overlap" value={s.sample_vae_tile_overlap ?? 64} oninput={(e) => update('sample_vae_tile_overlap', Number(e.target.value || 64))} min={0} tooltip="Spatial tile overlap" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="inference.sample_vae_temporal_tile_size" value={s.sample_vae_temporal_tile_size ?? 0} oninput={(e) => update('sample_vae_temporal_tile_size', Number(e.target.value || 0))} min={0} tooltip="Temporal tile size, 0 disables temporal tiling" />
								<FormField type="number" fieldPath="inference.sample_vae_temporal_tile_overlap" value={s.sample_vae_temporal_tile_overlap ?? 8} oninput={(e) => update('sample_vae_temporal_tile_overlap', Number(e.target.value || 8))} min={0} tooltip="Temporal tile overlap" />
							</div>
						</div>
					</FormGroup>
				{/if}

				<FormGroup title="Output">
					<div class="space-y-2 pt-2">
						<PathInput fieldPath="inference.output_dir" value={s.output_dir || 'output'} oninput={(e) => update('output_dir', e.target.value)} tooltip="Output directory" />
						<FormField fieldPath="inference.output_name" value={s.output_name || 'ltx2_sample'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix" />
						{#if $advancedMode}
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="inference.use_precached_sample_latents" checked={s.use_precached_sample_latents ?? false} onchange={(e) => update('use_precached_sample_latents', e.target.checked)} tooltip="Use precached latent tensors for inference" />
							</div>
							<PathInput fieldPath="inference.sample_latents_cache" value={s.sample_latents_cache || ''} oninput={(e) => update('sample_latents_cache', e.target.value)} tooltip="Path to precached latent tensors" />
							<FormField fieldPath="inference.extra_args" value={s.extra_args || ''} oninput={(e) => update('extra_args', e.target.value)} placeholder="--flag value --other_flag" tooltip="Extra arguments appended to the inference command. Use this for any CLI option without a dedicated dashboard control." />
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
