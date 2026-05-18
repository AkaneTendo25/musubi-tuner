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
	import { startModelDownload, getModelDownloadStatus, cancelModelDownload, formatModelDownloadStatus, getModelDownloadTone, isActiveModelDownload, getModelDownloadPresets, getModelDownloadPreflight, checkPathExists, scanCheckpointsWithProgress, cancelCheckpointScan, formatCheckpointScanStatus, modelDownloadTooltip, formatModelPreflightStatus } from '$lib/utils/modelDownloads.js';
	import { projectConfig, projectLoaded, updateSection, saveProjectNow } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, preloadLogsIfActive, startLogPolling, refreshStatuses } from '$lib/stores/processes.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import { goto } from '$app/navigation';
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
	let cacheThenTrainRunning = $state(false);
	let cacheThenTrainMessage = $state('');
	let cacheThenTrainError = $state('');
	let downloadPollTimer = null;

	class CacheThenTrainCancelled extends Error {
		constructor(message) {
			super(message);
			this.name = 'CacheThenTrainCancelled';
		}
	}

	onMount(() => {
		fetch('/api/fs/cwd').then((res) => res.ok ? res.json() : null).then((data) => { cwd = data?.cwd || ''; }).catch(() => {});
		getModelDownloadPresets().then((presets) => { downloadPresets = presets; }).catch(() => {});
		preloadLogsIfActive(['cache_latents', 'cache_text', 'cache_dino']);
		const logInterval = startLogPolling(['cache_latents', 'cache_text', 'cache_dino'], 1000);
		return () => {
			clearInterval(logInterval);
			if (downloadPollTimer) clearTimeout(downloadPollTimer);
			if (ltxScanJobId) cancelCheckpointScan(ltxScanJobId).catch(() => {});
			if (gemmaScanJobId) cancelCheckpointScan(gemmaScanJobId).catch(() => {});
			if (gemmaSafetensorsScanJobId) cancelCheckpointScan(gemmaSafetensorsScanJobId).catch(() => {});
		};
	});

	function updateCaching(key, value) { updateSection('caching', key, value); }

	let caching = $derived($projectConfig?.caching || {});
	let latentStatus = $derived($processStatuses.cache_latents || { state: 'idle', exit_code: null, stop_requested: false });
	let textStatus = $derived($processStatuses.cache_text || { state: 'idle', exit_code: null, stop_requested: false });
	let dinoStatus = $derived($processStatuses.cache_dino || { state: 'idle', exit_code: null, stop_requested: false });
	let trainingStatus = $derived($processStatuses.training || { state: 'idle', exit_code: null, stop_requested: false });
	let latentLogs = $derived($processLogs.cache_latents || []);
	let textLogs = $derived($processLogs.cache_text || []);
	let dinoLogs = $derived($processLogs.cache_dino || []);
	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let resolvedLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, caching.ltx2_checkpoint || ''));
	let activeGemmaSafetensors = $derived(effectiveGemmaSafetensors($projectConfig, caching.gemma_safetensors || ''));
	let gemmaRootDisabled = $derived(Boolean(activeGemmaSafetensors));
	let resolvedGemma = $derived(effectiveGemmaRoot(cwd, $projectConfig, caching.gemma_root || '', activeGemmaSafetensors));
	let scanTargetGemmaRoot = $derived(effectiveGemmaRoot(cwd, $projectConfig, caching.gemma_root || '', ''));
	let hasActiveDownload = $derived(Boolean(downloadJobId) && ['queued', 'running', 'cancelling'].includes(downloadState));
	let cacheThenTrainDisabled = $derived(
		cacheThenTrainRunning ||
		isProcessActive(latentStatus) ||
		isProcessActive(textStatus) ||
		isProcessActive(trainingStatus)
	);

	function relatedScanTargets() {
		return {
			ltx2: resolvedLtx,
			gemma: scanTargetGemmaRoot,
			gemma_safetensors: activeGemmaSafetensors
		};
	}

	function isProcessActive(status) {
		return status?.state === 'running' || status?.state === 'stopping';
	}

	const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

	async function waitForProcessSuccess(type, label) {
		let missingPolls = 0;
		let idlePolls = 0;

		while (true) {
			const statuses = await refreshStatuses();
			const status = statuses?.[type];

			if (!status) {
				missingPolls += 1;
				if (missingPolls >= 20) {
					throw new Error(`Lost status while waiting for ${label}`);
				}
				await sleep(1500);
				continue;
			}

			missingPolls = 0;
			const state = status.state || 'idle';
			const exitCode = status.exit_code ?? null;

			if (status.stop_requested) {
				throw new CacheThenTrainCancelled(`${label} was stopped. Cache then train cancelled.`);
			}
			if (state === 'finished' && exitCode === 0) return status;
			if (state === 'error' || (exitCode !== null && exitCode !== 0)) {
				throw new Error(`${label} failed${exitCode !== null ? ` with exit code ${exitCode}` : ''}`);
			}

			if (state === 'idle') {
				idlePolls += 1;
				if (idlePolls >= 6) {
					throw new Error(`${label} did not start`);
				}
			} else {
				idlePolls = 0;
			}

			await sleep(1500);
		}
	}

	async function cacheThenTrain() {
		if (cacheThenTrainDisabled) return;

		cacheThenTrainRunning = true;
		cacheThenTrainError = '';

		try {
			cacheThenTrainMessage = 'Caching latents...';
			await startProcess('cache_latents');
			await waitForProcessSuccess('cache_latents', 'Latent caching');

			cacheThenTrainMessage = 'Caching text embeddings...';
			await startProcess('cache_text');
			await waitForProcessSuccess('cache_text', 'Text embedding caching');

			cacheThenTrainMessage = 'Starting training...';
			await startProcess('training');
			await goto('/training/dashboard');
		} catch (e) {
			if (e instanceof CacheThenTrainCancelled) {
				cacheThenTrainError = '';
			} else {
				cacheThenTrainError = e?.message || 'Cache then train failed';
			}
		} finally {
			cacheThenTrainRunning = false;
			cacheThenTrainMessage = '';
		}
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
	<div class="space-y-5">
		<div class="flex items-start justify-between gap-3">
			<div>
				<h2 class="text-base font-semibold" style="color: var(--text-primary);">Caching</h2>
				<p class="text-[12px]" style="color: var(--text-muted);">Cache latents and text encoder outputs before training.</p>
			</div>
			<div class="flex flex-col items-end gap-2">
				<button
					type="button"
					onclick={cacheThenTrain}
					disabled={cacheThenTrainDisabled}
					class="px-4 py-2 text-[12px] font-semibold disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
					style="background: var(--accent); color: var(--bg-base); border-radius: var(--radius-sm); box-shadow: var(--shadow-sm), var(--glow-accent); font-family: var(--font-label);"
					onmouseenter={(e) => { if (!e.currentTarget.disabled) e.currentTarget.style.filter = 'brightness(1.1)'; }}
					onmouseleave={(e) => { e.currentTarget.style.filter = ''; }}
				>
					{cacheThenTrainRunning ? (cacheThenTrainMessage || 'Running...') : 'Cache then train'}
				</button>
				{#if cacheThenTrainError}
					<div class="max-w-[28rem] text-right text-[12px] px-2.5 py-1" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">
						{cacheThenTrainError}
					</div>
				{/if}
			</div>
		</div>
		<!-- Shared Settings -->
		<div class="p-4 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="grid grid-cols-2 xl:grid-cols-3 gap-3">
				<div class="space-y-2">
					<CheckpointInput fieldPath="caching.ltx2_checkpoint" label="LTX-2 Checkpoint" value={caching.ltx2_checkpoint || ''} onchange={(v) => updateCaching('ltx2_checkpoint', v)} showFiles tooltip="Path to the LTX-2 model checkpoint file" actionLabel="D" actionBusyLabel="..." actionDisabled={hasActiveDownload || ltxDownloadExists} actionTooltip={modelDownloadTooltip(downloadPresets, 'ltxav', resolvedLtx, ltxDownloadExists)} onaction={() => downloadModel('ltxav')} />
					<ModelPathStatus exists={ltxDownloadExists} foundPath={foundLtxPath} disabled={hasActiveDownload} scanning={scanningLtx} scanMessage={ltxScanMessage} scanTone={ltxScanTone} onscan={scanLtx} oncancel={stopLtxScan} onusefound={(path) => updateCaching('ltx2_checkpoint', path)} />
				</div>
				<div class="space-y-2">
					<CheckpointInput fieldPath="caching.gemma_root" label="Gemma Root" value={caching.gemma_root || ''} onchange={(v) => updateCaching('gemma_root', v)} disabled={gemmaRootDisabled} tooltip={gemmaRootDisabled ? 'Ignored while Gemma Safetensors is set' : 'Root directory containing Gemma text encoder weights'} actionLabel="D" actionBusyLabel="..." actionDisabled={gemmaRootDisabled || hasActiveDownload || gemmaDownloadExists} actionTooltip={gemmaRootDisabled ? 'Gemma Safetensors is active' : modelDownloadTooltip(downloadPresets, 'gemma-unsloth', resolvedGemma, gemmaDownloadExists)} onaction={() => downloadModel('gemma-unsloth')} />
					<ModelPathStatus exists={gemmaRootDisabled || gemmaDownloadExists} foundPath={foundGemmaPath} disabled={gemmaRootDisabled || hasActiveDownload} scanning={scanningGemma} scanMessage={gemmaScanMessage} scanTone={gemmaScanTone} onscan={scanGemma} oncancel={stopGemmaScan} onusefound={(path) => { updateCaching('gemma_root', path); updateCaching('gemma_safetensors', ''); }} />
				</div>
				<FormSelect fieldPath="caching.ltx2_mode" value={caching.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => updateCaching('ltx2_mode', e.target.value)} tooltip="Video: visual only, AV: audio+video, Audio: audio only" />
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
				<div class="space-y-2">
					<PathInput fieldPath="caching.gemma_safetensors" value={caching.gemma_safetensors || ''} oninput={(e) => updateCaching('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
					{#if activeGemmaSafetensors}
						<ModelPathStatus exists={gemmaSafetensorsExists} foundPath={foundGemmaSafetensorsPath} disabled={hasActiveDownload} scanning={scanningGemmaSafetensors} scanMessage={gemmaSafetensorsScanMessage} scanTone={gemmaSafetensorsScanTone} onscan={scanGemmaSafetensors} oncancel={stopGemmaSafetensorsScan} onusefound={(path) => updateCaching('gemma_safetensors', path)} />
					{/if}
				</div>
				<PathInput fieldPath="caching.ltx2_text_encoder_checkpoint" value={caching.ltx2_text_encoder_checkpoint || ''} oninput={(e) => updateCaching('ltx2_text_encoder_checkpoint', e.target.value)} showFiles tooltip="Separate text encoder checkpoint (if different from main)" />
				<FormSelect fieldPath="caching.mixed_precision" value={caching.mixed_precision || 'no'} options={['no', 'fp16', 'bf16']} onchange={(e) => updateCaching('mixed_precision', e.target.value)} tooltip="Mixed precision mode for text encoder caching." />
				<FormField type="number" fieldPath="caching.num_workers" value={caching.num_workers ?? ''} oninput={(e) => updateCaching('num_workers', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" tooltip="Number of data loader workers" />
			</div>
			<div class="grid grid-cols-3 gap-x-4 gap-y-1">
				<FormToggle fieldPath="caching.skip_existing" checked={caching.skip_existing ?? false} onchange={(e) => updateCaching('skip_existing', e.target.checked)} tooltip="Skip files that already have cached outputs" />
			</div>
			{#if $advancedMode}
				<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
					<FormSelect fieldPath="caching.vae_dtype" value={caching.vae_dtype || ''} options={[{ value: '', label: 'bfloat16 (default)' }, 'float16', 'bfloat16', 'float32']} onchange={(e) => updateCaching('vae_dtype', e.target.value || null)} tooltip="VAE dtype for latent caching. Blank uses the default `bfloat16`." />
					<FormField fieldPath="caching.device" value={caching.device || ''} oninput={(e) => updateCaching('device', e.target.value || null)} placeholder="Auto" tooltip="Torch device. Leave blank to auto-select the runtime device." />
					<FormSelect fieldPath="caching.quantize_device" value={caching.quantize_device || ''} options={[{ value: '', label: 'Auto' }, { value: 'cuda', label: 'CUDA' }, { value: 'cpu', label: 'CPU' }]} onchange={(e) => updateCaching('quantize_device', e.target.value || null)} tooltip="Device for quantization-related work. Blank auto-selects." />
					<FormToggle fieldPath="caching.keep_cache" checked={caching.keep_cache ?? false} onchange={(e) => updateCaching('keep_cache', e.target.checked)} tooltip="Keep old cache files when re-caching" />
				</div>
				<PathInput fieldPath="caching.save_dataset_manifest" value={caching.save_dataset_manifest || ''} oninput={(e) => updateCaching('save_dataset_manifest', e.target.value)} showFiles tooltip="Optional path to write a dataset manifest during latent caching." />
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
								<FormField type="number" fieldPath="caching.vae_chunk_size" value={caching.vae_chunk_size ?? ''} oninput={(e) => updateCaching('vae_chunk_size', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Frames per VAE chunk" />
								<FormField type="number" fieldPath="caching.vae_spatial_tile_size" value={caching.vae_spatial_tile_size ?? ''} oninput={(e) => updateCaching('vae_spatial_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="e.g. 512" tooltip="Spatial tile size (reduces VRAM)" />
							</div>
							<div class="grid grid-cols-3 gap-3">
								<FormField type="number" fieldPath="caching.vae_spatial_tile_overlap" value={caching.vae_spatial_tile_overlap ?? ''} oninput={(e) => updateCaching('vae_spatial_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="64" tooltip="Spatial tile overlap" />
								<FormField type="number" fieldPath="caching.vae_temporal_tile_size" value={caching.vae_temporal_tile_size ?? ''} oninput={(e) => updateCaching('vae_temporal_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Temporal tile size" />
								<FormField type="number" fieldPath="caching.vae_temporal_tile_overlap" value={caching.vae_temporal_tile_overlap ?? ''} oninput={(e) => updateCaching('vae_temporal_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="24" tooltip="Temporal tile overlap" />
							</div>
						</div>
					</FormGroup>

					<FormGroup title="Reference (V2V)">
						<div class="grid grid-cols-2 gap-3 pt-2">
							<FormField type="number" fieldPath="caching.reference_frames" value={caching.reference_frames ?? 1} oninput={(e) => updateCaching('reference_frames', Number(e.target.value))} min={1} tooltip="Reference frames for V2V" />
							<FormField type="number" fieldPath="caching.reference_downscale" value={caching.reference_downscale ?? 1} oninput={(e) => updateCaching('reference_downscale', Number(e.target.value))} min={1} tooltip="Reference downscale factor" />
						</div>
					</FormGroup>

					<FormGroup title="Precache I2V Latents">
						<div class="space-y-2 pt-2">
							<FormToggle fieldPath="caching.precache_sample_latents" checked={caching.precache_sample_latents ?? false} onchange={(e) => updateCaching('precache_sample_latents', e.target.checked)} tooltip="Pre-encode I2V conditioning latents from prompts defined on the Samples page." />
							{#if caching.precache_sample_latents}
								<PathInput fieldPath="caching.sample_prompts" value={caching.sample_prompts || ''} oninput={(e) => updateCaching('sample_prompts', e.target.value)} showFiles tooltip="Optional override. Leave blank to use prompts defined on the Samples page." />
								<PathInput fieldPath="caching.sample_latents_cache" value={caching.sample_latents_cache || ''} oninput={(e) => updateCaching('sample_latents_cache', e.target.value)} tooltip="Directory for cached sample conditioning latents." />
							{/if}
						</div>
					</FormGroup>
				{/if}

				{#if caching.ltx2_mode === 'av' || caching.ltx2_mode === 'audio'}
					<FormGroup title="Audio Source" collapsed={false}>
						<div class="space-y-2 pt-2">
							<FormSelect fieldPath="caching.ltx2_audio_source" value={caching.ltx2_audio_source || 'video'} options={['video', 'audio_files']} onchange={(e) => updateCaching('ltx2_audio_source', e.target.value)} tooltip="Extract from video or load separate files" />
							{#if caching.ltx2_audio_source === 'audio_files'}
								<PathInput fieldPath="caching.ltx2_audio_dir" value={caching.ltx2_audio_dir || ''} oninput={(e) => updateCaching('ltx2_audio_dir', e.target.value)} tooltip="Directory with audio files" />
								{#if $advancedMode}
									<FormField fieldPath="caching.ltx2_audio_ext" value={caching.ltx2_audio_ext || '.wav'} oninput={(e) => updateCaching('ltx2_audio_ext', e.target.value)} tooltip="Audio file extension" />
								{/if}
							{/if}
							{#if $advancedMode}
								<div class="grid grid-cols-2 gap-2">
									<FormField fieldPath="caching.ltx2_audio_dtype" value={caching.ltx2_audio_dtype || ''} oninput={(e) => updateCaching('ltx2_audio_dtype', e.target.value)} placeholder="Auto" tooltip="Audio latent dtype" />
									<FormField type="number" fieldPath="caching.audio_only_sequence_resolution" value={caching.audio_only_sequence_resolution ?? 64} oninput={(e) => updateCaching('audio_only_sequence_resolution', Number(e.target.value))} min={1} tooltip="Audio-only sequence resolution" />
								</div>
								<div class="grid grid-cols-2 gap-2">
									<FormField type="number" fieldPath="caching.audio_video_latent_channels" value={caching.audio_video_latent_channels ?? ''} oninput={(e) => updateCaching('audio_video_latent_channels', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" min={1} tooltip="Override video latent channels when caching audio-only latents" />
									<FormField fieldPath="caching.audio_video_latent_dtype" value={caching.audio_video_latent_dtype || ''} oninput={(e) => updateCaching('audio_video_latent_dtype', e.target.value)} placeholder="Auto" tooltip="Override video latent dtype for audio-only caching" />
								</div>
								<div class="grid grid-cols-2 gap-2">
									<FormField type="number" fieldPath="caching.audio_only_target_resolution" value={caching.audio_only_target_resolution ?? ''} oninput={(e) => updateCaching('audio_only_target_resolution', e.target.value ? Number(e.target.value) : null)} placeholder="Dataset default" min={1} tooltip="Square target resolution used to derive audio-only video latent shapes" />
									<FormField type="number" fieldPath="caching.audio_only_target_fps" value={caching.audio_only_target_fps ?? ''} oninput={(e) => updateCaching('audio_only_target_fps', e.target.value ? Number(e.target.value) : null)} placeholder="Default" min={0} step="0.1" tooltip="Target FPS used to derive frame count for audio-only caching" />
								</div>
							{/if}
						</div>
					</FormGroup>
				{/if}

				{#if $advancedMode}
					<FormGroup title="Cache Latents CLI">
						<div class="space-y-2 pt-2">
							<FormField fieldPath="caching.cache_latents_extra_args" value={caching.cache_latents_extra_args || ''} oninput={(e) => updateCaching('cache_latents_extra_args', e.target.value)} placeholder="--flag value --other_flag" tooltip="Extra arguments appended to the latent cache command. Use this for any CLI option without a dedicated dashboard control." />
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
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="caching.gemma_load_in_8bit" checked={caching.gemma_load_in_8bit ?? false} onchange={(e) => updateCaching('gemma_load_in_8bit', e.target.checked)} tooltip="Load Gemma with 8-bit quantization" />
							<FormToggle fieldPath="caching.gemma_load_in_4bit" checked={caching.gemma_load_in_4bit ?? false} onchange={(e) => updateCaching('gemma_load_in_4bit', e.target.checked)} tooltip="Load Gemma with 4-bit quantization" />
							<FormToggle fieldPath="caching.gemma_bnb_4bit_disable_double_quant" checked={caching.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => updateCaching('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable double quantization" />
							<FormToggle fieldPath="caching.gemma_fp8_weight_offload" checked={caching.gemma_fp8_weight_offload ?? true} onchange={(e) => updateCaching('gemma_fp8_weight_offload', e.target.checked)} tooltip="For FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. Disable this to keep more weights on VRAM and reduce RAM/pagefile pressure." />
						</div>
						{#if caching.gemma_load_in_4bit}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="caching.gemma_bnb_4bit_quant_type" value={caching.gemma_bnb_4bit_quant_type || 'nf4'} options={['nf4', 'fp4']} onchange={(e) => updateCaching('gemma_bnb_4bit_quant_type', e.target.value)} tooltip="NF4 recommended" />
								<FormSelect fieldPath="caching.gemma_bnb_4bit_compute_dtype" value={caching.gemma_bnb_4bit_compute_dtype || 'auto'} options={['auto', 'fp16', 'bf16', 'fp32']} onchange={(e) => updateCaching('gemma_bnb_4bit_compute_dtype', e.target.value)} tooltip="Compute dtype for 4-bit" />
							</div>
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Precache Samples">
					<div class="space-y-2 pt-2">
						<FormToggle fieldPath="caching.precache_sample_prompts" checked={caching.precache_sample_prompts ?? false} onchange={(e) => updateCaching('precache_sample_prompts', e.target.checked)} tooltip="Cache text embeddings for sample prompts" />
						{#if caching.precache_sample_prompts}
							<PathInput fieldPath="caching.sample_prompts" value={caching.sample_prompts || ''} oninput={(e) => updateCaching('sample_prompts', e.target.value)} showFiles tooltip="Optional override. Leave blank to use prompts defined on the Samples page." />
							<PathInput fieldPath="caching.sample_prompts_cache" value={caching.sample_prompts_cache || ''} oninput={(e) => updateCaching('sample_prompts_cache', e.target.value)} tooltip="Output directory for cached embeddings" />
						{/if}
					</div>
				</FormGroup>

				{#if $advancedMode}

					<FormGroup title="Precache Preservation">
						<div class="space-y-2 pt-2">
							<FormToggle fieldPath="caching.precache_preservation_prompts" checked={caching.precache_preservation_prompts ?? false} onchange={(e) => updateCaching('precache_preservation_prompts', e.target.checked)} tooltip="Cache preservation/regularization prompts" />
							{#if caching.precache_preservation_prompts}
								<PathInput fieldPath="caching.preservation_prompts_cache" value={caching.preservation_prompts_cache || ''} oninput={(e) => updateCaching('preservation_prompts_cache', e.target.value)} tooltip="Output directory for cached preservation embeddings" />
								<FormToggle fieldPath="caching.blank_preservation" checked={caching.blank_preservation ?? false} onchange={(e) => updateCaching('blank_preservation', e.target.checked)} tooltip="Use blank prompts" />
								<FormToggle fieldPath="caching.dop" checked={caching.dop ?? false} onchange={(e) => updateCaching('dop', e.target.checked)} tooltip="Differential Output Preservation" />
								{#if caching.dop}
									<FormField fieldPath="caching.dop_class_prompt" value={caching.dop_class_prompt || ''} oninput={(e) => updateCaching('dop_class_prompt', e.target.value)} placeholder="e.g. woman" tooltip="Class word for DOP" />
								{/if}
							{/if}
						</div>
					</FormGroup>

					<FormGroup title="Connector LoRA">
						<div class="space-y-2 pt-2">
							<FormToggle fieldPath="caching.cache_before_connector" checked={caching.cache_before_connector ?? false} onchange={(e) => updateCaching('cache_before_connector', e.target.checked)} tooltip="Save pre-connector text features alongside standard embeddings. Required for --train_connectors during training." />
						</div>
					</FormGroup>

					<FormGroup title="Cache Text CLI">
						<div class="space-y-2 pt-2">
							<FormField fieldPath="caching.cache_text_extra_args" value={caching.cache_text_extra_args || ''} oninput={(e) => updateCaching('cache_text_extra_args', e.target.value)} placeholder="--flag value --other_flag" tooltip="Extra arguments appended to the text cache command. Use this for any CLI option without a dedicated dashboard control." />
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
