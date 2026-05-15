<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormCombobox from '$lib/components/FormCombobox.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import CheckpointInput from '$lib/components/CheckpointInput.svelte';
	import ModelPathStatus from '$lib/components/ModelPathStatus.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { defaultModelDir, describeExactModelScan, effectiveGemmaRoot, effectiveGemmaSafetensors, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { startModelDownload, getModelDownloadStatus, cancelModelDownload, formatModelDownloadStatus, getModelDownloadTone, isActiveModelDownload, getModelDownloadPresets, getModelDownloadPreflight, checkPathExists, scanCheckpointsWithProgress, cancelCheckpointScan, formatCheckpointScanStatus, modelDownloadTooltip, formatModelPreflightStatus } from '$lib/utils/modelDownloads.js';
	import { projectConfig, projectLoaded, updateSection, saveProjectNow } from '$lib/stores/project.js';
	import { processStatuses, processValidation, startProcess, stopProcess, validateProcess } from '$lib/stores/processes.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	const DEFAULT_NETWORK_MODULE = 'networks.lora_ltx2';
	const LOKR_NETWORK_MODULE = 'networks.lokr';
	const DEFAULT_SINKSGD_LEARNING_RATE = '0.001';
	const DEFAULT_SINKSGD_DORA_OFT_LEARNING_RATE = '0.0005';
	const DEFAULT_SINKSGD_OPTIMIZER_ARGS = 'spectral_normalization=True scale_lr_with_effective_batch=True normed_momentum=True momentum=0.995 nesterov=True nesterov_coef=0.8 orthogonal_sinkhorn=True sinkhorn_iterations=3';
	const LOKR_FFN_TARGET_PRESETS = new Set([
		'v2v',
		'video_sa_ff',
		'video_sa_ca_ff',
		'character_training',
		'audio',
		'audio_ref_only_ic',
		'av_ic',
		'video_ref_only_av',
		'full',
	]);

	function parseLokrFactor(raw) {
		const match = String(raw || '').match(/(?:^|\s)factor=("[^"]+"|'[^']+'|\S+)/);
		if (!match) return -1;
		const parsed = Number(match[1].replace(/^['"]|['"]$/g, ''));
		return Number.isFinite(parsed) ? Math.round(parsed) : -1;
	}

	function factorization(dimension, factor = -1) {
		if (factor > 0 && (dimension % factor) === 0) {
			let m = factor;
			let n = dimension / factor;
			if (m > n) [n, m] = [m, n];
			return [m, n];
		}
		if (factor < 0) factor = dimension;
		let m = 1;
		let n = dimension;
		let length = m + n;
		while (m < n) {
			let newM = m + 1;
			while (dimension % newM !== 0) newM += 1;
			let newN = dimension / newM;
			if (newM + newN > length || newM > factor) {
				break;
			} else {
				m = newM;
				n = newN;
			}
		}
		if (m > n) [n, m] = [m, n];
		return [m, n];
	}

	function getLokrDenseThreshold(training) {
		const networkModule = training.network_module || DEFAULT_NETWORK_MODULE;
		if (networkModule !== LOKR_NETWORK_MODULE) return null;

		const dim = Number(training.network_dim);
		if (!Number.isFinite(dim) || dim <= 0) return null;

		const hasGuiFactor = training.lokr_factor !== null && training.lokr_factor !== undefined && training.lokr_factor !== '';
		const guiFactor = hasGuiFactor ? Number(training.lokr_factor) : NaN;
		const factor = Number.isFinite(guiFactor) ? Math.round(guiFactor) : parseLokrFactor(training.network_args || '');
		const includeFfn = LOKR_FFN_TARGET_PRESETS.has(training.lora_target_preset || 't2v');
		const pairs = [
			[4096, 4096],
			[2048, 2048],
			[4096, 2048],
			[2048, 4096],
		];
		if (includeFfn) {
			pairs.push(
				[4096, 16384],
				[16384, 4096],
				[2048, 8192],
				[8192, 2048],
			);
		}

		let threshold = 0;
		for (const [inDim, outDim] of pairs) {
			const [, inN] = factorization(inDim, factor);
			const [, outK] = factorization(outDim, factor);
			threshold = Math.max(threshold, Math.max(outK, inN) / 2);
		}
		if (dim < threshold) return null;

		return {
			dim,
			threshold,
			factor,
			includeFfn,
		};
	}

	function update(key, value) { updateSection('training', key, value); }
	function automaticLearningRate(training) {
		return training?.use_dora_oft ? `Auto: ${DEFAULT_SINKSGD_DORA_OFT_LEARNING_RATE}` : `Auto: ${DEFAULT_SINKSGD_LEARNING_RATE}`;
	}
	function normalizeOptimizerType(value) {
		return String(value || 'SinkSGD_adv').replace(/[-_]/g, '').toLowerCase();
	}
	function isSinkSgdOptimizer(value) {
		const normalized = normalizeOptimizerType(value);
		return normalized === 'sinksgd' || normalized === 'sinksgdadv';
	}
	function updateOptimizerType(value) {
		update('optimizer_type', value);
		if (!isSinkSgdOptimizer(value)) return;
		update('learning_rate', null);
		update('optimizer_args', DEFAULT_SINKSGD_OPTIMIZER_ARGS);
		update('sinksgd_orthogonal_sinkhorn', true);
		update('sinksgd_compiled_optimizer', false);
		update('lr_scheduler', 'constant');
	}
	function updateNetworkMode(mode) {
		if (mode === 'lokr') {
			update('network_module', LOKR_NETWORK_MODULE);
			update('use_lokr', true);
			update('use_dora', false);
			update('use_dokr', false);
			update('use_dora_oft', false);
			return;
		}
		if (mode === 'dora') {
			update('network_module', DEFAULT_NETWORK_MODULE);
			update('use_lokr', false);
			update('use_dora', true);
			update('use_dokr', false);
			update('use_dora_oft', false);
			return;
		}
		if (mode === 'dokr') {
			update('network_module', LOKR_NETWORK_MODULE);
			update('use_lokr', false);
			update('use_dora', false);
			update('use_dokr', true);
			update('use_dora_oft', false);
			return;
		}
		if (mode === 'dora_oft') {
			update('network_module', DEFAULT_NETWORK_MODULE);
			update('use_lokr', false);
			update('use_dora', false);
			update('use_dokr', false);
			update('use_dora_oft', true);
			update('scaled_oft', true);
			return;
		}
		update('network_module', DEFAULT_NETWORK_MODULE);
		update('use_lokr', false);
		update('use_dora', false);
		update('use_dokr', false);
		update('use_dora_oft', false);
	}
	function updateUseLokr(enabled) {
		updateNetworkMode(enabled ? 'lokr' : 'lora');
	}
	function updateUseDora(enabled) {
		updateNetworkMode(enabled ? 'dora' : 'lora');
	}
	function updateUseDokr(enabled) {
		updateNetworkMode(enabled ? 'dokr' : 'lora');
	}
	function updateUseDoraOft(enabled) {
		updateNetworkMode(enabled ? 'dora_oft' : 'lora');
	}
	function updateScaledOft(enabled) {
		update('scaled_oft', t.use_dora_oft ? true : enabled);
	}
	async function startTraining() {
		await startProcess('training');
		await goto('/training/dashboard');
	}

	// Common optimizer presets
	const optimizerOptions = [
		'SinkSGD_adv',
		'SinkSGD',
		'adamw8bit',
		'came8bit',
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
		'came8bit',
		'torchao_adamw8bit',
		'torchao_adamw4bit',
		'torchao_adamwfp8',
		'torchao_adamw',
		'optimi_stableadamw',
		'optimi_adamw',
		'optimi_lion',
		'optimi_adan',
	];

	let t = $derived($projectConfig?.training || {});
	let isLokrBackend = $derived((t.network_module || DEFAULT_NETWORK_MODULE) === LOKR_NETWORK_MODULE);
	let lokrDenseThreshold = $derived(getLokrDenseThreshold(t));
	let trainingStatus = $derived($processStatuses.training || { state: 'idle', exit_code: null });
	let trainingValidation = $derived($processValidation.training || { ok: true, summary: '', errors: [], warnings: [], field_errors: {}, field_warnings: {} });
	let hasValidationIssues = $derived((trainingValidation.errors?.length || 0) > 0 || (trainingValidation.warnings?.length || 0) > 0);
	let hasDatasetValidationErrors = $derived((trainingValidation.errors || []).some((issue) => issue.page === 'dataset'));
	let validationTimer = null;
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
	let downloadPollTimer = null;

	onMount(async () => {
		try {
			const res = await fetch('/api/fs/cwd');
			if (res.ok) cwd = (await res.json()).cwd || '';
		} catch {}
		try {
			downloadPresets = await getModelDownloadPresets();
		} catch {}
		return () => {
			clearTimeout(downloadPollTimer);
			if (ltxScanJobId) cancelCheckpointScan(ltxScanJobId).catch(() => {});
			if (gemmaScanJobId) cancelCheckpointScan(gemmaScanJobId).catch(() => {});
			if (gemmaSafetensorsScanJobId) cancelCheckpointScan(gemmaSafetensorsScanJobId).catch(() => {});
		};
	});

	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let resolvedLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, t.ltx2_checkpoint || ''));
	let activeGemmaSafetensors = $derived(effectiveGemmaSafetensors($projectConfig, t.gemma_safetensors || ''));
	let gemmaRootDisabled = $derived(Boolean(activeGemmaSafetensors));
	let resolvedGemma = $derived(effectiveGemmaRoot(cwd, $projectConfig, t.gemma_root || '', activeGemmaSafetensors));
	let scanTargetGemmaRoot = $derived(effectiveGemmaRoot(cwd, $projectConfig, t.gemma_root || '', ''));
	let hasActiveDownload = $derived(Boolean(downloadJobId) && ['queued', 'running', 'cancelling'].includes(downloadState));

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

	function setModelStatus(status) {
		modelStatus = formatModelDownloadStatus(status);
		modelStatusTone = getModelDownloadTone(status);
		downloadState = status?.state || '';
	}

	async function finalizeDownload(status) {
		if (status.state === 'completed' && status.path) {
			update('ltx2_checkpoint', downloading === 'ltxav' ? status.path : t.ltx2_checkpoint || '');
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
		clearTimeout(downloadPollTimer);
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

	function fieldError(field) {
		return trainingValidation.field_errors?.[field]?.[0] || '';
	}

	function fieldInvalid(field) {
		return Boolean(fieldError(field));
	}

	$effect(() => {
		if (!$projectLoaded || !$projectConfig) return;

		clearTimeout(validationTimer);
		const configSnapshot = $projectConfig;
		validationTimer = setTimeout(() => {
			validateProcess('training', configSnapshot).catch(() => {});
		}, 250);

		return () => clearTimeout(validationTimer);
	});
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
						<CheckpointInput fieldPath="training.ltx2_checkpoint" label="LTX-2 Checkpoint" value={t.ltx2_checkpoint || ''} onchange={(v) => update('ltx2_checkpoint', v)} showFiles tooltip="Path to LTX-2 checkpoint" invalid={fieldInvalid('training.ltx2_checkpoint')} error={fieldError('training.ltx2_checkpoint')} actionLabel="D" actionBusyLabel="..." actionDisabled={hasActiveDownload || ltxDownloadExists} actionTooltip={modelDownloadTooltip(downloadPresets, 'ltxav', resolvedLtx, ltxDownloadExists)} onaction={() => downloadModel('ltxav')} />
						<ModelPathStatus exists={ltxDownloadExists} foundPath={foundLtxPath} disabled={hasActiveDownload} scanning={scanningLtx} scanMessage={ltxScanMessage} scanTone={ltxScanTone} onscan={scanLtx} oncancel={stopLtxScan} onusefound={(path) => update('ltx2_checkpoint', path)} />
						<CheckpointInput fieldPath="training.gemma_root" label="Gemma Root" value={t.gemma_root || ''} onchange={(v) => update('gemma_root', v)} disabled={gemmaRootDisabled} tooltip={gemmaRootDisabled ? 'Ignored while Gemma Safetensors is set' : 'Gemma text encoder directory'} invalid={fieldInvalid('training.gemma_root')} error={fieldError('training.gemma_root')} actionLabel="D" actionBusyLabel="..." actionDisabled={gemmaRootDisabled || hasActiveDownload || gemmaDownloadExists} actionTooltip={gemmaRootDisabled ? 'Gemma Safetensors is active' : modelDownloadTooltip(downloadPresets, 'gemma-unsloth', resolvedGemma, gemmaDownloadExists)} onaction={() => downloadModel('gemma-unsloth')} />
						<ModelPathStatus exists={gemmaRootDisabled || gemmaDownloadExists} foundPath={foundGemmaPath} disabled={gemmaRootDisabled || hasActiveDownload} scanning={scanningGemma} scanMessage={gemmaScanMessage} scanTone={gemmaScanTone} onscan={scanGemma} oncancel={stopGemmaScan} onusefound={(path) => { update('gemma_root', path); update('gemma_safetensors', ''); }} />
						<PathInput fieldPath="training.gemma_safetensors" value={t.gemma_safetensors || ''} oninput={(e) => update('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" invalid={fieldInvalid('training.gemma_safetensors')} error={fieldError('training.gemma_safetensors')} />
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
						{#if $advancedMode}
							<PathInput fieldPath="training.dataset_manifest" value={t.dataset_manifest || ''} oninput={(e) => update('dataset_manifest', e.target.value)} showFiles tooltip="Optional manifest file. If set, training uses this instead of regenerating dataset_config.toml." invalid={fieldInvalid('training.dataset_manifest')} error={fieldError('training.dataset_manifest')} />
							<PathInput fieldPath="training.config_file" value={t.config_file || ''} oninput={(e) => update('config_file', e.target.value)} showFiles tooltip="Optional trainer TOML config passed as --config_file. Dashboard values still appear on the command line and can override config-file values." />
							<PathInput fieldPath="training.dataset_config" value={t.dataset_config || ''} oninput={(e) => update('dataset_config', e.target.value)} showFiles tooltip="Optional dataset TOML path passed as --dataset_config instead of the dashboard-generated dataset config. Leave blank to use dashboard datasets." />
						{/if}
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.ltx2_mode" value={t.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} tooltip="Video/AV/Audio" />
							<FormSelect fieldPath="training.mixed_precision" value={t.mixed_precision || 'no'} options={['no', 'fp16', 'bf16']} onchange={(e) => update('mixed_precision', e.target.value)} tooltip="Mixed precision mode" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.ltx_version" value={t.ltx_version || '2.3'} options={['2.0', '2.3']} onchange={(e) => update('ltx_version', e.target.value)} tooltip="Target LTX version behavior" />
							<FormSelect fieldPath="training.ltx_version_check_mode" value={t.ltx_version_check_mode || 'warn'} options={['off', 'warn', 'error']} onchange={(e) => update('ltx_version_check_mode', e.target.value)} tooltip="Behavior when the checkpoint and selected LTX version do not match." />
						</div>
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.fp8_base" checked={t.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision (VRAM savings)" />
							<FormToggle fieldPath="training.fp8_scaled" checked={t.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} tooltip="Scaled FP8 for stability" />
							<FormToggle fieldPath="training.flash_attn" checked={t.flash_attn ?? false} onchange={(e) => update('flash_attn', e.target.checked)} tooltip="Flash Attention 2" />
							<FormToggle fieldPath="training.sdpa" checked={t.sdpa ?? false} onchange={(e) => update('sdpa', e.target.checked)} tooltip="PyTorch SDPA attention" />
							<FormToggle fieldPath="training.gemma_load_in_8bit" checked={t.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
							<FormToggle fieldPath="training.gemma_load_in_4bit" checked={t.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
							<FormToggle fieldPath="training.gemma_fp8_weight_offload" checked={t.gemma_fp8_weight_offload ?? true} onchange={(e) => update('gemma_fp8_weight_offload', e.target.checked)} tooltip="For FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. Disable this to keep more weights on VRAM and reduce RAM/pagefile pressure." />
						</div>
						{#if $advancedMode}
							<FormField fieldPath="training.fp8_keep_blocks" value={t.fp8_keep_blocks || ''} oninput={(e) => update('fp8_keep_blocks', e.target.value)} placeholder="0,1,2,45" tooltip="Transformer block indices to keep in high precision when FP8 Scaled is enabled. Ranges like 0-2,45 are accepted." />
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="training.vae_dtype" value={t.vae_dtype || ''} options={[{ value: '', label: 'Default' }, 'bfloat16', 'float16', 'float32']} onchange={(e) => update('vae_dtype', e.target.value || null)} tooltip="Optional VAE dtype passed to the trainer." />
								<FormSelect fieldPath="training.gemma_bnb_4bit_quant_type" value={t.gemma_bnb_4bit_quant_type || 'nf4'} options={['nf4', 'fp4']} onchange={(e) => update('gemma_bnb_4bit_quant_type', e.target.value)} tooltip="bitsandbytes 4-bit quant type for Gemma." />
							</div>
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="training.flash3" checked={t.flash3 ?? false} onchange={(e) => update('flash3', e.target.checked)} tooltip="FlashAttention 3 backend" />
								<FormToggle fieldPath="training.sage_attn" checked={t.sage_attn ?? false} onchange={(e) => update('sage_attn', e.target.checked)} tooltip="Sage Attention backend" />
								<FormToggle fieldPath="training.xformers" checked={t.xformers ?? false} onchange={(e) => update('xformers', e.target.checked)} tooltip="xFormers attention" />
								<FormToggle fieldPath="training.gemma_bnb_4bit_disable_double_quant" checked={t.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => update('gemma_bnb_4bit_disable_double_quant', e.target.checked)} tooltip="Disable double quantization (4-bit)" />
								<FormToggle fieldPath="training.ltx2_audio_only_model" checked={t.ltx2_audio_only_model ?? false} onchange={(e) => update('ltx2_audio_only_model', e.target.checked)} tooltip="Audio-only model architecture" />
							</div>
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="LoRA">
						<div class="space-y-2 pt-2">
						{#if $advancedMode}
							<FormField fieldPath="training.network_module" value={t.network_module || ''} oninput={(e) => update('network_module', e.target.value || DEFAULT_NETWORK_MODULE)} placeholder={DEFAULT_NETWORK_MODULE} tooltip="LTX-2 LoRA network module. Clearing this resets it to the LTX-2 default." />
						{/if}
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="training.network_dim" value={t.network_dim ?? ''} oninput={(e) => update('network_dim', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="4 for default LoRA" tooltip="LoRA rank. Blank uses the network module default; for the standard LoRA module that is `4`." />
							<FormField type="number" fieldPath="training.network_alpha" value={t.network_alpha ?? ''} oninput={(e) => update('network_alpha', e.target.value ? Number(e.target.value) : null)} min={0} step="0.1" placeholder="Auto: rank" tooltip="LoRA alpha. Blank lets SinkSGD_adv spectral scaling set alpha to rank in the generated command." />
							<FormSelect fieldPath="training.lora_target_preset" value={t.lora_target_preset || 't2v'} options={[
								{ value: 't2v', label: 't2v (all attn)' },
								{ value: 'v2v', label: 'v2v (all attn+FFN)' },
								{ value: 'lycoris', label: 'lycoris (attn)' },
								{ value: 'video_sa', label: 'V:SA' },
								{ value: 'video_sa_ff', label: 'V:SA+FF' },
								{ value: 'video_sa_ca_ff', label: 'V:SA+CA+FF' },
								{ value: 'character_training', label: 'Character Training' },
								{ value: 'audio', label: 'audio' },
								{ value: 'audio_v2a', label: 'audio+V2A' },
								{ value: 'audio_ref_ic', label: 'audio ref IC' },
								{ value: 'av_ic', label: 'AV IC' },
								{ value: 'video_ref_only_av', label: 'AV video-ref' },
								{ value: 'full', label: 'full (all)' }
							]} onchange={(e) => update('lora_target_preset', e.target.value)} tooltip="Target layers" />
						</div>
						{#if lokrDenseThreshold}
							<div class="text-[11px] px-3 py-2" style="color: var(--warning); background: color-mix(in srgb, var(--warning) 10%, transparent); border: 1px solid color-mix(in srgb, var(--warning) 24%, var(--border)); border-radius: var(--radius-sm);">
								For the current LoKr/DoKr target set ({lokrDenseThreshold.includeFfn ? 'attention+FFN' : 'attention-only'}, {lokrDenseThreshold.factor > 0 ? `factor=${lokrDenseThreshold.factor}` : 'balanced factorization'}), dims &gt;= {lokrDenseThreshold.threshold} already use dense LoKr blocks. network_dim={lokrDenseThreshold.dim} will not increase effective LoKr capacity beyond that threshold.
							</div>
						{/if}
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="LoKr" checked={(t.use_lokr ?? false) || (isLokrBackend && !(t.use_dokr ?? false))} onchange={(e) => updateUseLokr(e.target.checked)} tooltip="LoKr uses a Kronecker-structured update, which is usually more parameter-efficient than plain LoRA while keeping higher expressivity at a given size." />
							<FormToggle label="DoRA" checked={(t.use_dora ?? false) && !isLokrBackend} onchange={(e) => updateUseDora(e.target.checked)} tooltip="DoRA adds a separate magnitude vector on top of LoRA. It usually gives cleaner concept separation and tends to preserve more of the base model's original capability." />
							<FormToggle label="DokR" checked={(t.use_dokr ?? false) || (isLokrBackend && (t.use_dora ?? false))} onchange={(e) => updateUseDokr(e.target.checked)} tooltip="DokR combines LoKr efficiency with DoRA's magnitude control. In practice this is the strongest all-around option if you want both efficiency and cleaner adaptation." />
							<FormToggle label="DoRA-OFT" checked={t.use_dora_oft ?? false} onchange={(e) => updateUseDoraOft(e.target.checked)} tooltip="DoRA magnitude learning on top of scaled OFT rotations. Enables use_dora_oft=true and stores OFT tensors plus dora_scale." />
						</div>
						{#if isLokrBackend}
							<div class="grid grid-cols-3 gap-2">
								<FormField label="LoKr Factor" type="number" fieldPath="training.lokr_factor" value={t.lokr_factor ?? ''} oninput={(e) => update('lokr_factor', e.target.value ? Number(e.target.value) : null)} placeholder="Balanced" step="1" tooltip="Kronecker factorization control for LoKr/DokR. Blank or -1 uses balanced factorization; positive values like 1, 2, or 4 force that smaller factor." />
							</div>
						{/if}
						{#if t.use_dora_oft}
							<div class="grid grid-cols-4 gap-2">
								<FormToggle label="Scaled OFT" checked={true} onchange={(e) => updateScaledOft(e.target.checked)} disabled={true} tooltip="Required for DoRA-OFT. Signed optimizers like Adam/CAME need Scaled OFT so the DoRA scale and OFT blocks share a usable step size." />
								<FormField label="OFT Block Size" type="number" value={t.oft_block_size ?? ''} oninput={(e) => update('oft_block_size', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Match LoRA dim" tooltip="Block size for OFT rotations. If it does not divide the input width, the backend adjusts it to the nearest valid divisor." />
								<FormToggle label="Constrained OFT" checked={t.oft_coft ?? false} onchange={(e) => update('oft_coft', e.target.checked)} tooltip="Project OFT rotation parameters into a small neighborhood each step." />
								<FormToggle label="Block Share" checked={t.oft_block_share ?? false} onchange={(e) => update('oft_block_share', e.target.checked)} tooltip="Share one OFT block across all input blocks." />
							</div>
							{#if t.oft_coft}
								<div class="grid grid-cols-4 gap-2">
									<FormField label="COFT Eps" type="number" value={t.coft_eps ?? 0.00006} oninput={(e) => update('coft_eps', Number(e.target.value))} min={0} step="0.00001" tooltip="Constraint epsilon for OFT projection." />
								</div>
							{/if}
							<div class="text-xs" style="color: var(--text-muted);">
								DoRA-OFT checkpoints store OFT tensors plus dora_scale for OFTv2-compatible loaders.
							</div>
						{/if}
						{#if $advancedMode}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="training.ic_lora_strategy" value={t.ic_lora_strategy || 'auto'} options={[
									{ value: 'auto', label: 'auto' },
									{ value: 'none', label: 'none' },
									{ value: 'v2v', label: 'v2v' },
									{ value: 'audio_ref_ic', label: 'audio_ref_ic' },
									{ value: 'av_ic', label: 'av_ic' },
									{ value: 'video_ref_only_av', label: 'video_ref_only_av' },
								]} onchange={(e) => update('ic_lora_strategy', e.target.value)} tooltip="IC-LoRA conditioning strategy. 'auto' follows lora_target_preset; 'audio_ref_ic' = audio-reference ID-LoRA style (requires av or audio mode); 'av_ic' = joint video+audio reference conditioning (requires av mode, with extra AV modifiers below); 'video_ref_only_av' = video reference with target AV generation (requires av mode)" />
							</div>
							{#if t.ic_lora_strategy === 'audio_ref_ic' || t.ic_lora_strategy === 'av_ic' || (t.ic_lora_strategy === 'auto' && (t.lora_target_preset === 'audio_ref_ic' || t.lora_target_preset === 'av_ic'))}
								<div class="p-2 space-y-2" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
									<span class="text-[11px] font-medium" style="color: var(--text-muted);">Audio-Reference IC-LoRA</span>
									{#if t.ic_lora_strategy === 'av_ic' || (t.ic_lora_strategy === 'auto' && t.lora_target_preset === 'av_ic')}
										<div class="grid grid-cols-2 gap-2">
											<FormSelect fieldPath="training.av_cross_attention_mode" value={t.av_cross_attention_mode || 'both'} options={[
												{ value: 'both', label: 'both' },
												{ value: 'a2v_only', label: 'a2v_only' },
												{ value: 'v2a_only', label: 'v2a_only' },
												{ value: 'none', label: 'none' },
											]} onchange={(e) => update('av_cross_attention_mode', e.target.value)} tooltip="AV IC cross-modal direction control. 'both' = default AV IC, 'a2v_only' = audio-to-video only, 'v2a_only' = video-to-audio only, 'none' = disable both cross-modal directions." />
											<FormToggle fieldPath="training.av_multi_ref" checked={t.av_multi_ref ?? false} onchange={(e) => update('av_multi_ref', e.target.checked)} tooltip="Mark this AV IC run as multi-reference. Backend multi-reference support uses the plural dataset reference fields." />
										</div>
									{/if}
									<div class="grid grid-cols-3 gap-x-4 gap-y-1">
										<FormToggle fieldPath="training.audio_ref_use_negative_positions" checked={t.audio_ref_use_negative_positions ?? false} onchange={(e) => update('audio_ref_use_negative_positions', e.target.checked)} tooltip="Place reference-audio token positions in negative time" />
										<FormToggle fieldPath="training.audio_ref_mask_cross_attention_to_reference" checked={t.audio_ref_mask_cross_attention_to_reference ?? false} onchange={(e) => update('audio_ref_mask_cross_attention_to_reference', e.target.checked)} tooltip="Video attends only to target audio, not reference-audio tokens" />
										<FormToggle fieldPath="training.audio_ref_mask_reference_from_text_attention" checked={t.audio_ref_mask_reference_from_text_attention ?? false} onchange={(e) => update('audio_ref_mask_reference_from_text_attention', e.target.checked)} tooltip={t.ic_lora_strategy === 'av_ic' || (t.ic_lora_strategy === 'auto' && t.lora_target_preset === 'av_ic') ? 'Not supported in AV IC Modality-path mode; the trainer warns and ignores this flag.' : 'Block reference-audio tokens from attending to text tokens'} />
									</div>
									<FormField type="number" fieldPath="training.audio_ref_identity_guidance_scale" value={t.audio_ref_identity_guidance_scale ?? 0.0} oninput={(e) => update('audio_ref_identity_guidance_scale', Number(e.target.value))} step="0.1" min={0} tooltip="Extra forward pass without reference to isolate and amplify speaker identity (0 = disabled, recommended: 3.0)" />
									<div class="grid grid-cols-3 items-end gap-x-4 gap-y-1">
										<FormToggle fieldPath="training.av_bimodal_cfg" checked={t.av_bimodal_cfg ?? false} onchange={(e) => update('av_bimodal_cfg', e.target.checked)} tooltip="Extra forward pass with cross-modal attention disabled to strengthen independent audio/video generation" />
										{#if t.av_bimodal_cfg}
											<FormField type="number" fieldPath="training.av_bimodal_scale" value={t.av_bimodal_scale ?? 3.0} oninput={(e) => update('av_bimodal_scale', Number(e.target.value))} step="0.1" min={1} tooltip="Bimodal guidance strength. Applied as (scale-1) × delta. Default: 3.0" />
										{/if}
									</div>
								</div>
							{/if}
						{/if}
						{#if $advancedMode}
						<FormField fieldPath="training.network_args" value={t.network_args || ''} oninput={(e) => update('network_args', e.target.value)} placeholder="key=value ..." tooltip="Extra network args (space-separated key=value)" />
						<div class="grid grid-cols-4 gap-2">
							<FormField type="number" fieldPath="training.network_dropout" value={t.network_dropout ?? ''} oninput={(e) => update('network_dropout', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="LoRA dropout rate" />
							<FormField type="number" fieldPath="training.rank_dropout" value={t.rank_dropout ?? ''} oninput={(e) => update('rank_dropout', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="Adaptive rank dropout passed via --network_args." />
							<FormField type="number" fieldPath="training.module_dropout" value={t.module_dropout ?? ''} oninput={(e) => update('module_dropout', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="Module dropout passed via --network_args." />
							<FormField type="number" fieldPath="training.scale_weight_norms" value={t.scale_weight_norms ?? ''} oninput={(e) => update('scale_weight_norms', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Max norm for weight scaling" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="training.caption_dropout_rate" value={t.caption_dropout_rate ?? 0} oninput={(e) => update('caption_dropout_rate', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Global caption dropout rate for CFG training" />
							<FormField type="number" fieldPath="training.video_caption_dropout_rate" value={t.video_caption_dropout_rate ?? 0} oninput={(e) => update('video_caption_dropout_rate', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Video-caption dropout rate" />
							<FormField type="number" fieldPath="training.audio_caption_dropout_rate" value={t.audio_caption_dropout_rate ?? 0} oninput={(e) => update('audio_caption_dropout_rate', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Audio-caption dropout rate" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.audio_dim" value={t.audio_dim ?? ''} oninput={(e) => update('audio_dim', e.target.value ? Number(e.target.value) : null)} placeholder="Match main dim" tooltip="Optional separate audio LoRA rank." />
							<FormField type="number" fieldPath="training.audio_alpha" value={t.audio_alpha ?? ''} oninput={(e) => update('audio_alpha', e.target.value ? Number(e.target.value) : null)} placeholder="Match main alpha" step="0.1" tooltip="Optional separate audio LoRA alpha." />
						</div>
						<PathInput fieldPath="training.network_weights" value={t.network_weights || ''} oninput={(e) => update('network_weights', e.target.value)} showFiles tooltip="Warm-start from existing LoRA weights" />
						<FormField fieldPath="training.base_weights" value={t.base_weights || ''} oninput={(e) => update('base_weights', e.target.value)} placeholder="path1 path2 ..." tooltip="Space-separated base weights passed through to the trainer." />
						<FormField fieldPath="training.base_weights_multiplier" value={t.base_weights_multiplier || ''} oninput={(e) => update('base_weights_multiplier', e.target.value)} placeholder="1.0 0.5 ..." tooltip="Optional multipliers paired with Base Weights." />
						<PathInput fieldPath="training.lycoris_config" value={t.lycoris_config || ''} oninput={(e) => update('lycoris_config', e.target.value)} showFiles tooltip="Path to LyCORIS TOML config (enables LyCORIS mode)" />
						<FormSelect fieldPath="training.lycoris_quantized_base_check_mode" value={t.lycoris_quantized_base_check_mode || 'warn'} options={['off', 'warn', 'error']} onchange={(e) => update('lycoris_quantized_base_check_mode', e.target.value)} tooltip="Check for quantized base with LyCORIS" />
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.init_lokr_norm" value={t.init_lokr_norm ?? ''} oninput={(e) => update('init_lokr_norm', e.target.value ? Number(e.target.value) : null)} placeholder="Disabled" step="0.1" tooltip="Initial LoKr norm, mainly for LyCORIS/LoKr variants." />
							<FormToggle fieldPath="training.adaptive_rank" checked={t.adaptive_rank ?? false} onchange={(e) => update('adaptive_rank', e.target.checked)} tooltip="Enable adaptive-rank LoRA arguments." />
						</div>
						{#if t.adaptive_rank}
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="training.adaptive_rank_target" value={t.adaptive_rank_target ?? ''} oninput={(e) => update('adaptive_rank_target', e.target.value ? Number(e.target.value) : null)} placeholder="Required" min={1} tooltip="Target rank for adaptive-rank LoRA." />
								<FormField type="number" fieldPath="training.adaptive_rank_min_rank" value={t.adaptive_rank_min_rank ?? ''} oninput={(e) => update('adaptive_rank_min_rank', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" min={1} tooltip="Minimum rank floor." />
								<FormField type="number" fieldPath="training.adaptive_rank_init_rank" value={t.adaptive_rank_init_rank ?? ''} oninput={(e) => update('adaptive_rank_init_rank', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" min={1} tooltip="Optional initial rank." />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="training.adaptive_rank_quantile" value={t.adaptive_rank_quantile ?? ''} oninput={(e) => update('adaptive_rank_quantile', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.01" min={0} max={1} tooltip="Adaptive-rank quantile." />
								<FormField type="number" fieldPath="training.adaptive_rank_weight" value={t.adaptive_rank_weight ?? ''} oninput={(e) => update('adaptive_rank_weight', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.1" min={0} tooltip="Adaptive-rank loss weight." />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField label="Budget" type="number" value={t.adaptive_rank_budget ?? ''} oninput={(e) => update('adaptive_rank_budget', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" min={1} tooltip="Shared target sum of effective ranks. Overrides Budget Ratio." />
								<FormField label="Budget Ratio" type="number" value={t.adaptive_rank_budget_ratio ?? ''} oninput={(e) => update('adaptive_rank_budget_ratio', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.01" min={0} max={1} tooltip="Shared budget as a ratio of total max rank." />
								<FormField label="Budget Weight" type="number" value={t.adaptive_rank_budget_weight ?? ''} oninput={(e) => update('adaptive_rank_budget_weight', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.0001" min={0} tooltip="Loss weight for the shared rank budget." />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<div class="flex items-end pb-0.5">
									<FormToggle label="Estimate" checked={t.adaptive_rank_estimate ?? false} onchange={(e) => update('adaptive_rank_estimate', e.target.checked)} tooltip="Use or generate ltx2_estimate.json for module-wise rank allocation." />
								</div>
								<FormSelect label="Estimate Apply" value={t.adaptive_rank_estimate_apply || ''} options={[{ value: '', label: 'Default target' }, { value: 'target', label: 'Target only' }, { value: 'init', label: 'Init only' }, { value: 'both', label: 'Target + init' }]} onchange={(e) => update('adaptive_rank_estimate_apply', e.target.value || null)} tooltip="How ltx2_estimate.json allocations are applied." />
								<FormField label="Finalize Start" type="number" value={t.adaptive_rank_finalize_start ?? ''} oninput={(e) => update('adaptive_rank_finalize_start', e.target.value ? Number(e.target.value) : null)} placeholder="Off" step="0.01" min={0} max={1} tooltip="Training progress where adaptive modules convert to static ranks." />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField label="Recover Steps" type="number" value={t.adaptive_rank_finalize_recover_steps ?? ''} oninput={(e) => update('adaptive_rank_finalize_recover_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Off" min={1} tooltip="Recovery window after finalize/prune." />
								<FormField label="Recover Warmup" type="number" value={t.adaptive_rank_finalize_recover_warmup_steps ?? ''} oninput={(e) => update('adaptive_rank_finalize_recover_warmup_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Off" min={0} tooltip="Warmup steps inside the recovery window." />
								<div class="flex items-end pb-0.5">
									<FormToggle label="Hard Prune" checked={t.adaptive_rank_hard_prune ?? false} onchange={(e) => update('adaptive_rank_hard_prune', e.target.checked)} tooltip="Periodically rebuild modules as static lower-rank LoRA during training." />
								</div>
							</div>
							{#if t.adaptive_rank_hard_prune}
								<div class="grid grid-cols-3 gap-2">
									<FormField label="Prune Start" type="number" value={t.adaptive_rank_hard_prune_start ?? ''} oninput={(e) => update('adaptive_rank_hard_prune_start', e.target.value ? Number(e.target.value) : null)} placeholder="0.5" step="0.01" min={0} max={1} tooltip="Training progress before hard pruning can start." />
									<FormField label="Prune Interval" type="number" value={t.adaptive_rank_hard_prune_interval ?? ''} oninput={(e) => update('adaptive_rank_hard_prune_interval', e.target.value ? Number(e.target.value) : null)} placeholder="100" min={1} tooltip="Steps between hard-prune checks." />
									<FormField label="Min Delta" type="number" value={t.adaptive_rank_hard_prune_min_delta ?? ''} oninput={(e) => update('adaptive_rank_hard_prune_min_delta', e.target.value ? Number(e.target.value) : null)} placeholder="1" min={1} tooltip="Minimum rank reduction before a module is rebuilt." />
								</div>
							{/if}
						{/if}
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.dim_from_weights" checked={t.dim_from_weights ?? false} onchange={(e) => update('dim_from_weights', e.target.checked)} tooltip="Auto-detect dim/alpha from weights" />
							<FormToggle fieldPath="training.save_original_lora" checked={t.save_original_lora ?? true} onchange={(e) => update('save_original_lora', e.target.checked)} tooltip="Save original LoRA format" />
							<FormToggle fieldPath="training.train_connectors" checked={t.train_connectors ?? false} onchange={(e) => update('train_connectors', e.target.checked)} tooltip="Also apply LoRA to text connector modules. Requires caching with 'Cache Pre-Connector Features' enabled. Not compatible with LyCORIS." />
						</div>
						{/if}
					</div>
				</FormGroup>

				{#if $advancedMode}
					<FormGroup title="Quantization">
						<div class="space-y-2 pt-2">
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="training.nf4_base" checked={t.nf4_base ?? false} onchange={(e) => update('nf4_base', e.target.checked)} tooltip="NF4 4-bit quantization (~75% VRAM savings)" />
								<FormToggle fieldPath="training.loftq_init" checked={t.loftq_init ?? false} onchange={(e) => update('loftq_init', e.target.checked)} tooltip="LoftQ initialization (compensates NF4 error)" />
								<FormToggle fieldPath="training.fp8_w8a8" checked={t.fp8_w8a8 ?? false} onchange={(e) => update('fp8_w8a8', e.target.checked)} tooltip="W8A8 activation quantization (requires FP8 Base and FP8 Scaled)" />
								<FormToggle fieldPath="training.awq_calibration" checked={t.awq_calibration ?? false} onchange={(e) => update('awq_calibration', e.target.checked)} tooltip="Activation-aware calibration for NF4" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="training.nf4_block_size" value={t.nf4_block_size ?? 32} oninput={(e) => update('nf4_block_size', Number(e.target.value))} disabled={!t.nf4_base} tooltip="Block size for NF4 quantization" />
								<FormField type="number" fieldPath="training.loftq_iters" value={t.loftq_iters ?? 2} oninput={(e) => update('loftq_iters', Number(e.target.value))} min={1} disabled={!t.loftq_init} tooltip="LoftQ alternating iterations" />
								<FormSelect fieldPath="training.w8a8_mode" value={t.w8a8_mode || 'int8'} options={['int8', 'fp8']} onchange={(e) => update('w8a8_mode', e.target.value)} disabled={!t.fp8_w8a8} tooltip="int8 (Turing+) or fp8 (Ada+)" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="training.awq_alpha" value={t.awq_alpha ?? 0.25} oninput={(e) => update('awq_alpha', Number(e.target.value))} step="0.05" min={0} max={1} disabled={!t.awq_calibration} tooltip="AWQ scaling strength" />
								<FormField type="number" fieldPath="training.awq_num_batches" value={t.awq_num_batches ?? 8} oninput={(e) => update('awq_num_batches', Number(e.target.value))} min={1} disabled={!t.awq_calibration} tooltip="Calibration batches" />
								<FormSelect fieldPath="training.quantize_device" value={t.quantize_device || ''} options={[{value:'',label:'Auto'},{value:'cuda',label:'CUDA'},{value:'cpu',label:'CPU'}]} onchange={(e) => update('quantize_device', e.target.value || null)} tooltip="Device for quantization math" />
							</div>
						</div>
					</FormGroup>
				{/if}

				<FormGroup title="Optimizer">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.learning_rate" value={t.learning_rate ?? ''} oninput={(e) => update('learning_rate', e.target.value ? Number(e.target.value) : null)} step="any" placeholder={automaticLearningRate(t)} tooltip="Learning rate. Blank uses 0.001 for SinkSGD_adv, or 0.0005 for DoRA-OFT." />
							<FormCombobox fieldPath="training.optimizer_type" value={t.optimizer_type || 'SinkSGD_adv'} oninput={(e) => updateOptimizerType(e.target.value)} options={optimizerOptions} placeholder="SinkSGD_adv" tooltip="Optimizer type. Selecting SinkSGD_adv or SinkSGD applies the current dashboard SinkSGD defaults to this project." />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormSelect fieldPath="training.lr_scheduler" value={t.lr_scheduler || 'constant'} options={['constant', 'cosine', 'warmup_stable_decay', 'constant_with_warmup', 'cosine_with_restarts', 'linear', 'polynomial', 'rex']} onchange={(e) => update('lr_scheduler', e.target.value)} tooltip="LR schedule. Constant matches OneTrainer's default; cosine decays over the whole run and can under-drive short runs." />
							<FormField type="number" fieldPath="training.lr_warmup_steps" value={t.lr_warmup_steps ?? 0} oninput={(e) => update('lr_warmup_steps', Number(e.target.value))} min={0} tooltip="Warmup steps" />
							<FormField type="number" fieldPath="training.gradient_accumulation_steps" value={t.gradient_accumulation_steps ?? 1} oninput={(e) => update('gradient_accumulation_steps', Number(e.target.value))} min={1} tooltip="Gradient accumulation" />
						</div>
						{#if isSinkSgdOptimizer(t.optimizer_type)}
							<div class="grid grid-cols-2 gap-2">
								<FormToggle label="Orthogonal Sinkhorn" fieldPath="training.sinksgd_orthogonal_sinkhorn" checked={t.sinksgd_orthogonal_sinkhorn ?? true} onchange={(e) => update('sinksgd_orthogonal_sinkhorn', e.target.checked)} tooltip="Enable OrthoGrad-style projection inside SinkSGD's Sinkhorn normalization. Generated commands set orthogonal_sinkhorn from this toggle." />
								<FormToggle label="Compiled Optimizer" fieldPath="training.sinksgd_compiled_optimizer" checked={t.sinksgd_compiled_optimizer ?? false} onchange={(e) => update('sinksgd_compiled_optimizer', e.target.checked)} tooltip="Compile SinkSGD's parameter step with torch.compile. First optimizer step may take longer while compiling." />
							</div>
						{/if}
						{#if $advancedMode}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="training.accumulation_group_by" value={t.accumulation_group_by || 'none'} options={[{value:'none',label:'None'},{value:'frames',label:'Frames'},{value:'bucket',label:'Bucket'},{value:'dataset',label:'Dataset'}]} onchange={(e) => update('accumulation_group_by', e.target.value)} tooltip="Keep gradient accumulation windows grouped by frame count, full bucket, or dataset. Bucket is safest for mixed frame lengths." />
								<FormSelect fieldPath="training.accumulation_group_remainder" value={t.accumulation_group_remainder || 'drop'} options={[{value:'drop',label:'Drop'},{value:'pad',label:'Pad'},{value:'allow_mixed',label:'Allow Mixed'}]} onchange={(e) => update('accumulation_group_remainder', e.target.value)} disabled={(t.accumulation_group_by || 'none') === 'none'} tooltip="How to handle incomplete accumulation groups." />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="training.max_grad_norm" value={t.max_grad_norm ?? 1.0} oninput={(e) => update('max_grad_norm', Number(e.target.value))} step="0.1" tooltip="Gradient clipping" />
								<FormField fieldPath="training.optimizer_args" value={t.optimizer_args || ''} oninput={(e) => update('optimizer_args', e.target.value)} placeholder={DEFAULT_SINKSGD_OPTIMIZER_ARGS} tooltip="Extra optimizer args. SinkSGD_adv defaults include normed long momentum, Nesterov coefficient blending, three Sinkhorn iterations, and sqrt effective-batch LR scaling; Orthogonal Sinkhorn is controlled by the visible toggle." />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="training.lr_decay_steps" value={t.lr_decay_steps ?? ''} oninput={(e) => update('lr_decay_steps', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="LR decay steps" />
								<FormField type="number" fieldPath="training.lr_scheduler_timescale" value={t.lr_scheduler_timescale ?? ''} oninput={(e) => update('lr_scheduler_timescale', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="LR scheduler timescale" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="training.lr_scheduler_num_cycles" value={t.lr_scheduler_num_cycles ?? ''} oninput={(e) => update('lr_scheduler_num_cycles', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="Cosine restarts cycles" />
								<FormField type="number" fieldPath="training.lr_scheduler_min_lr_ratio" value={t.lr_scheduler_min_lr_ratio ?? ''} oninput={(e) => update('lr_scheduler_min_lr_ratio', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.01" tooltip="Minimum LR ratio" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="training.lr_scheduler_power" value={t.lr_scheduler_power ?? 1.0} oninput={(e) => update('lr_scheduler_power', e.target.value ? Number(e.target.value) : null)} step="0.1" tooltip="Polynomial scheduler power." />
								<FormField fieldPath="training.lr_scheduler_type" value={t.lr_scheduler_type || ''} oninput={(e) => update('lr_scheduler_type', e.target.value)} placeholder="None" tooltip="Optional custom scheduler module override. Blank means no custom scheduler type." />
								<FormField fieldPath="training.lr_scheduler_args" value={t.lr_scheduler_args || ''} oninput={(e) => update('lr_scheduler_args', e.target.value)} placeholder="key=value ..." tooltip="Extra scheduler arguments passed through to the trainer." />
							</div>
							<FormField type="number" fieldPath="training.audio_lr" value={t.audio_lr ?? ''} oninput={(e) => update('audio_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Same as LR" step="any" tooltip="Separate LR for audio LoRA modules" />
							<FormField fieldPath="training.lr_args" value={t.lr_args || ''} oninput={(e) => update('lr_args', e.target.value)} placeholder="pattern=lr ..." tooltip="Per-module LR overrides (e.g. audio_attn=1e-6)" />
							<FormField fieldPath="training.lr_group_warmup_args" value={t.lr_group_warmup_args || ''} oninput={(e) => update('lr_group_warmup_args', e.target.value)} placeholder="audio=500 video=1500" tooltip="Per-module learning-rate warmup args passed as --lr_group_warmup_args." />
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Schedule">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.max_train_steps" value={t.max_train_steps ?? 1600} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Total training steps" />
							<FormField type="number" fieldPath="training.max_train_epochs" value={t.max_train_epochs ?? ''} oninput={(e) => update('max_train_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Epochs (overrides steps)" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.timestep_sampling" value={t.timestep_sampling || 'sigma'} options={['sigma', 'uniform', 'sigmoid', 'shift', 'shifted_logit_normal', 'logsnr']} onchange={(e) => update('timestep_sampling', e.target.value)} tooltip="Timestep sampling" />
							<FormField type="number" fieldPath="training.validate_every_n_steps" value={t.validate_every_n_steps ?? ''} oninput={(e) => update('validate_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Run validation every N steps" />
						</div>
						{#if $advancedMode}
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="training.discrete_flow_shift" value={t.discrete_flow_shift ?? 1.0} oninput={(e) => update('discrete_flow_shift', Number(e.target.value))} step="0.1" tooltip="Flow matching shift" />
								<FormSelect fieldPath="training.weighting_scheme" value={t.weighting_scheme || 'none'} options={['none', 'logit_normal', 'mode', 'cosmap', 'sigma_sqrt']} onchange={(e) => update('weighting_scheme', e.target.value)} tooltip="Loss weighting" />
								<FormField type="number" fieldPath="training.validate_every_n_epochs" value={t.validate_every_n_epochs ?? ''} oninput={(e) => update('validate_every_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Run validation every N epochs" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormField type="number" fieldPath="training.seed" value={t.seed ?? ''} oninput={(e) => update('seed', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Random seed" />
								<FormField type="number" fieldPath="training.guidance_scale" value={t.guidance_scale ?? ''} oninput={(e) => update('guidance_scale', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Training guidance scale" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormField type="number" fieldPath="training.sigmoid_scale" value={t.sigmoid_scale ?? ''} oninput={(e) => update('sigmoid_scale', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Sigmoid scale for timestep sampling" />
								<FormField type="number" fieldPath="training.logit_mean" value={t.logit_mean ?? ''} oninput={(e) => update('logit_mean', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Logit normal mean" />
								<FormField type="number" fieldPath="training.logit_std" value={t.logit_std ?? ''} oninput={(e) => update('logit_std', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.1" tooltip="Logit normal std" />
							</div>
							<div class="grid grid-cols-3 gap-2">
								<FormSelect fieldPath="training.show_timesteps" value={t.show_timesteps || ''} options={[{value:'',label:'Off'},{value:'image',label:'Image'},{value:'console',label:'Console'}]} onchange={(e) => update('show_timesteps', e.target.value || null)} tooltip="Debug output for sampled timesteps." />
								<FormField type="number" fieldPath="training.log_timestep_distribution_interval" value={t.log_timestep_distribution_interval ?? 100} oninput={(e) => update('log_timestep_distribution_interval', Number(e.target.value))} min={1} tooltip="Optimizer-step interval for TensorBoard timestep distribution logging." />
								<div class="flex items-end pb-0.5">
									<FormToggle fieldPath="training.log_timestep_distribution_tensorboard" checked={t.log_timestep_distribution_tensorboard ?? true} onchange={(e) => update('log_timestep_distribution_tensorboard', e.target.checked)} tooltip="Enable TensorBoard histogram logging for timestep distribution." />
								</div>
							</div>
						{/if}
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.mode_scale" value={t.mode_scale ?? 1.29} oninput={(e) => update('mode_scale', e.target.value ? Number(e.target.value) : null)} step="0.01" tooltip="Mode weighting scale." />
							<FormField type="number" fieldPath="training.shifted_logit_shift" value={t.shifted_logit_shift ?? ''} oninput={(e) => update('shifted_logit_shift', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" step="0.1" tooltip="Optional shifted-logit shift override." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.min_timestep" value={t.min_timestep ?? ''} oninput={(e) => update('min_timestep', e.target.value ? Number(e.target.value) : null)} placeholder="None" step={1} min={0} max={999} tooltip="Minimum CLI timestep value, 0-999" />
							<FormField type="number" fieldPath="training.max_timestep" value={t.max_timestep ?? ''} oninput={(e) => update('max_timestep', e.target.value ? Number(e.target.value) : null)} placeholder="None" step={1} min={1} max={1000} tooltip="Maximum CLI timestep value, 1-1000" />
						</div>
						{#if $advancedMode}
							<div class="grid grid-cols-3 gap-2">
								<div class="flex items-end pb-0.5">
									<FormToggle fieldPath="training.shifted_logit_clamp_auto_shift" checked={t.shifted_logit_clamp_auto_shift ?? false} onchange={(e) => update('shifted_logit_clamp_auto_shift', e.target.checked)} tooltip="Clamp auto-computed shifted-logit shifts to the min/max shift bounds." />
								</div>
								<FormField type="number" fieldPath="training.shifted_logit_min_shift" value={t.shifted_logit_min_shift ?? 0.95} oninput={(e) => update('shifted_logit_min_shift', Number(e.target.value))} step="0.01" tooltip="Lower clamp bound for auto-computed shifted-logit shifts." />
								<FormField type="number" fieldPath="training.shifted_logit_max_shift" value={t.shifted_logit_max_shift ?? 2.05} oninput={(e) => update('shifted_logit_max_shift', Number(e.target.value))} step="0.01" tooltip="Upper clamp bound for auto-computed shifted-logit shifts." />
							</div>
						{/if}
					</div>
				</FormGroup>

				<FormGroup title="Memory">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.blocks_to_swap" value={t.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0-40" min={0} max={40} tooltip="CPU offload blocks" />
							<FormField type="number" fieldPath="training.max_data_loader_n_workers" value={t.max_data_loader_n_workers ?? ''} oninput={(e) => update('max_data_loader_n_workers', e.target.value === '' ? null : Number(e.target.value))} placeholder="CLI default" min={0} tooltip="Dataloader workers" />
						</div>
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.gradient_checkpointing" checked={t.gradient_checkpointing ?? false} onchange={(e) => update('gradient_checkpointing', e.target.checked)} tooltip="Gradient checkpointing" />
							<FormToggle fieldPath="training.separate_audio_buckets" checked={t.separate_audio_buckets ?? false} onchange={(e) => update('separate_audio_buckets', e.target.checked)} tooltip="Separate audio/video buckets" />
							<FormToggle fieldPath="training.persistent_data_loader_workers" checked={t.persistent_data_loader_workers ?? false} onchange={(e) => update('persistent_data_loader_workers', e.target.checked)} tooltip="Keep workers between epochs" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.ltx2_first_frame_conditioning_p" value={t.ltx2_first_frame_conditioning_p ?? 0.1} oninput={(e) => update('ltx2_first_frame_conditioning_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="First frame conditioning prob" />
							{#if t.ltx2_mode === 'audio'}
								<FormField type="number" fieldPath="training.audio_only_sequence_resolution" value={t.audio_only_sequence_resolution ?? 64} oninput={(e) => update('audio_only_sequence_resolution', Number(e.target.value))} min={0} tooltip="Virtual pixel resolution for shifted_logit_normal in audio-only mode (0 = use cached geometry)" />
							{/if}
						</div>

						<!-- Endpoint-keyframe training (orthogonal to IC-LoRA strategy) -->
						<div class="pt-3 space-y-2" style="border-top: 1px solid var(--border-subtle);">
							<div class="flex items-center justify-between">
								<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Endpoint Keyframe Training</span>
							</div>
							<p class="text-[11px]" style="color: var(--text-muted);">
								Extract first / last / random-interior latent frames of the target as clean keyframe tokens at training time.
								Composes with any <code>--ic_lora_strategy</code>. All sub-fields are no-ops while the master toggle is off.
							</p>
							<FormToggle fieldPath="training.keyframe_endpoint_training" checked={t.keyframe_endpoint_training ?? false} onchange={(e) => update('keyframe_endpoint_training', e.target.checked)} tooltip="Master enable for endpoint-keyframe training. When off, the four probabilities below are not emitted to the CLI." />
							{#if t.keyframe_endpoint_training}
								<div class="grid grid-cols-2 gap-2">
									<FormField type="number" fieldPath="training.keyframe_first_frame_p" value={t.keyframe_first_frame_p ?? 1.0} oninput={(e) => update('keyframe_first_frame_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending the first latent frame as a clean keyframe at frame_idx=0 (independent Bernoulli per item in the batch)." />
									<FormField type="number" fieldPath="training.keyframe_last_frame_p" value={t.keyframe_last_frame_p ?? 1.0} oninput={(e) => update('keyframe_last_frame_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending the last latent frame at frame_idx=(T-1)*VIDEO_SCALE_FACTORS.time (pixel-frame units)." />
									<FormField type="number" fieldPath="training.keyframe_random_interior_p" value={t.keyframe_random_interior_p ?? 0.0} oninput={(e) => update('keyframe_random_interior_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending random interior latent frames as keyframes (interior indices are shared across the batch; only the dropout decision is per-sample)." />
									<FormField type="number" fieldPath="training.keyframe_max_random_interior" value={t.keyframe_max_random_interior ?? 0} oninput={(e) => update('keyframe_max_random_interior', Number(e.target.value))} step="1" min={0} tooltip="Cap on number of random interior keyframes per batch when keyframe_random_interior_p triggers." />
								</div>
							{/if}
						</div>
						{#if $advancedMode}
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="training.split_attn_target" value={t.split_attn_target || ''} options={[{value:'',label:'None'},{value:'all',label:'All'},{value:'self',label:'Self'},{value:'cross',label:'Cross'}]} onchange={(e) => update('split_attn_target', e.target.value || null)} tooltip="Split attention target" />
								<FormSelect fieldPath="training.split_attn_mode" value={t.split_attn_mode || ''} options={[{value:'',label:'None'},{value:'batch',label:'Batch'},{value:'query',label:'Query'}]} onchange={(e) => update('split_attn_mode', e.target.value || null)} disabled={!t.split_attn_target} tooltip="Split mode" />
							</div>
							<div class="grid grid-cols-2 gap-2">
								<FormSelect fieldPath="training.ffn_chunk_target" value={t.ffn_chunk_target || ''} options={[{value:'',label:'None'},{value:'all',label:'All'},{value:'video',label:'Video'},{value:'audio',label:'Audio'}]} onchange={(e) => update('ffn_chunk_target', e.target.value || null)} tooltip="Chunks transformer feed-forward layers by sequence tokens to reduce peak activation VRAM. Use 'all' for low-VRAM video/AV runs; use 'video' if audio is already comfortable." />
								<FormField type="number" fieldPath="training.ffn_chunk_size" value={t.ffn_chunk_size ?? 0} oninput={(e) => update('ffn_chunk_size', Number(e.target.value))} disabled={!t.ffn_chunk_target} tooltip="Tokens per FFN chunk. 0 disables. Start with 512 for low VRAM, 1024 for mild savings, 256/128 only if still OOM; smaller is slower." />
							</div>
							<div class="grid grid-cols-3 gap-x-4 gap-y-1">
								<FormToggle fieldPath="training.gradient_checkpointing_cpu_offload" checked={t.gradient_checkpointing_cpu_offload ?? false} onchange={(e) => update('gradient_checkpointing_cpu_offload', e.target.checked)} tooltip="Offload checkpointed activations to CPU" />
								<FormToggle fieldPath="training.split_attn" checked={t.split_attn ?? false} onchange={(e) => update('split_attn', e.target.checked)} tooltip="Legacy --split_attn flag. LTX-2-specific split controls are above." />
								<FormToggle fieldPath="training.blockwise_checkpointing" checked={t.blockwise_checkpointing ?? false} onchange={(e) => update('blockwise_checkpointing', e.target.checked)} tooltip="Per-block checkpointing" />
								<FormToggle fieldPath="training.use_pinned_memory_for_block_swap" checked={t.use_pinned_memory_for_block_swap ?? false} onchange={(e) => update('use_pinned_memory_for_block_swap', e.target.checked)} tooltip="Pinned memory for block swap" />
								<FormToggle fieldPath="training.img_in_txt_in_offloading" checked={t.img_in_txt_in_offloading ?? false} onchange={(e) => update('img_in_txt_in_offloading', e.target.checked)} tooltip="Offload img_in/txt_in to CPU" />
							</div>
							<FormField type="number" fieldPath="training.blocks_to_checkpoint" value={t.blocks_to_checkpoint ?? ''} oninput={(e) => update('blocks_to_checkpoint', e.target.value ? Number(e.target.value) : null)} placeholder="All" disabled={!t.blockwise_checkpointing} tooltip="Number of blocks to checkpoint (default: all)" />
							<FormField type="number" fieldPath="training.split_attn_chunk_size" value={t.split_attn_chunk_size ?? ''} oninput={(e) => update('split_attn_chunk_size', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!t.split_attn_target} tooltip="Split attention chunk size" />
						{/if}
					</div>
				</FormGroup>

			</div>

			<!-- Right column -->
			<div class="space-y-3">
				<FormGroup title="Output">
					<div class="space-y-2 pt-2">
						<PathInput fieldPath="training.output_dir" value={t.output_dir || ''} oninput={(e) => update('output_dir', e.target.value)} tooltip="Checkpoint save directory" invalid={fieldInvalid('training.output_dir')} error={fieldError('training.output_dir')} />
						<FormField fieldPath="training.output_name" value={t.output_name || 'ltx2_lora'} oninput={(e) => update('output_name', e.target.value)} tooltip="Checkpoint filename" />
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.save_every_n_epochs" value={t.save_every_n_epochs ?? ''} oninput={(e) => update('save_every_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Save every N epochs" />
							<FormField type="number" fieldPath="training.save_every_n_steps" value={t.save_every_n_steps ?? ''} oninput={(e) => update('save_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Save every N steps" />
						</div>
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.save_state" checked={t.save_state ?? false} onchange={(e) => update('save_state', e.target.checked)} tooltip="Save optimizer state" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.log_with" value={t.log_with || ''} options={[{value:'',label:'None'},{value:'tensorboard',label:'TensorBoard'},{value:'wandb',label:'W&B'}]} onchange={(e) => update('log_with', e.target.value || null)} tooltip="Logging integration" />
							<PathInput fieldPath="training.logging_dir" value={t.logging_dir || ''} oninput={(e) => update('logging_dir', e.target.value)} disabled={!t.log_with} tooltip="Log directory" invalid={fieldInvalid('training.logging_dir')} error={fieldError('training.logging_dir')} />
						</div>
						{#if $advancedMode}
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.save_last_n_epochs" value={t.save_last_n_epochs ?? ''} oninput={(e) => update('save_last_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N epoch checkpoints" />
							<FormField type="number" fieldPath="training.save_last_n_steps" value={t.save_last_n_steps ?? ''} oninput={(e) => update('save_last_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N step checkpoints" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.save_last_n_epochs_state" value={t.save_last_n_epochs_state ?? ''} oninput={(e) => update('save_last_n_epochs_state', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N epoch optimizer states" />
							<FormField type="number" fieldPath="training.save_last_n_steps_state" value={t.save_last_n_steps_state ?? ''} oninput={(e) => update('save_last_n_steps_state', e.target.value ? Number(e.target.value) : null)} placeholder="All" tooltip="Only keep last N step optimizer states" />
						</div>
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.save_state_on_train_end" checked={t.save_state_on_train_end ?? false} onchange={(e) => update('save_state_on_train_end', e.target.checked)} tooltip="Save state at training end" />
							<FormToggle fieldPath="training.save_checkpoint_metadata" checked={t.save_checkpoint_metadata ?? false} onchange={(e) => update('save_checkpoint_metadata', e.target.checked)} tooltip="Save JSON metadata alongside each checkpoint" />
							<FormToggle fieldPath="training.no_metadata" checked={t.no_metadata ?? false} onchange={(e) => update('no_metadata', e.target.checked)} tooltip="Skip metadata in checkpoint" />
							<FormToggle fieldPath="training.no_convert_to_comfy" checked={t.no_convert_to_comfy ?? false} onchange={(e) => update('no_convert_to_comfy', e.target.checked)} tooltip="Skip ComfyUI format conversion" />
							<FormToggle fieldPath="training.full_fp16" checked={t.full_fp16 ?? false} onchange={(e) => update('full_fp16', e.target.checked)} tooltip="FP16 gradients (stochastic rounding)" />
							<FormToggle fieldPath="training.full_bf16" checked={t.full_bf16 ?? false} onchange={(e) => update('full_bf16', e.target.checked)} tooltip="BF16 gradients (stochastic rounding)" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.loss_type" value={t.loss_type || 'mse'} options={['mse', 'mae', 'l1', 'huber', 'smooth_l1']} onchange={(e) => update('loss_type', e.target.value)} tooltip="Loss function type" />
							<FormField type="number" fieldPath="training.huber_delta" value={t.huber_delta ?? 1.0} oninput={(e) => update('huber_delta', Number(e.target.value))} step="0.1" disabled={!['huber','smooth_l1'].includes(t.loss_type)} tooltip="Delta for Huber/smooth_l1 loss" />
						</div>
						<PathInput fieldPath="training.resume" value={t.resume || ''} oninput={(e) => update('resume', e.target.value)} showFiles tooltip="Resume training from saved state" />
						<div class="grid grid-cols-2 gap-2">
							<FormToggle fieldPath="training.autoresume" checked={t.autoresume ?? false} onchange={(e) => update('autoresume', e.target.checked)} tooltip="Auto-resume from latest state in output_dir" />
							<FormToggle fieldPath="training.reset_dataloader" checked={t.reset_dataloader ?? false} onchange={(e) => update('reset_dataloader', e.target.checked)} tooltip="Skip mid-epoch resume, restart epoch from beginning" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormToggle fieldPath="training.reset_optimizer" checked={t.reset_optimizer ?? false} onchange={(e) => update('reset_optimizer', e.target.checked)} tooltip="Clear optimizer momentum/variance on resume" />
							<FormToggle fieldPath="training.reset_optimizer_params" checked={t.reset_optimizer_params ?? false} onchange={(e) => update('reset_optimizer_params', e.target.checked)} tooltip="Reset lr/weight_decay to CLI values on resume" />
						</div>
						<FormField fieldPath="training.wandb_run_name" value={t.wandb_run_name || ''} oninput={(e) => update('wandb_run_name', e.target.value)} disabled={t.log_with !== 'wandb'} placeholder="Auto" tooltip="Weights & Biases run name" />
						<FormField fieldPath="training.wandb_api_key" value={t.wandb_api_key || ''} oninput={(e) => update('wandb_api_key', e.target.value)} disabled={t.log_with !== 'wandb'} placeholder="Optional" tooltip="W&B API key (or set WANDB_API_KEY env)" />
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="training.log_prefix" value={t.log_prefix || ''} oninput={(e) => update('log_prefix', e.target.value)} placeholder="None" tooltip="Prefix for log metrics" />
							<FormField fieldPath="training.log_tracker_name" value={t.log_tracker_name || ''} oninput={(e) => update('log_tracker_name', e.target.value)} placeholder="Auto" tooltip="Custom tracker/project name" />
						</div>
						<PathInput fieldPath="training.log_tracker_config" value={t.log_tracker_config || ''} oninput={(e) => update('log_tracker_config', e.target.value)} showFiles disabled={!t.log_with} tooltip="Optional tracker config TOML path." />
						<FormToggle fieldPath="training.log_config" checked={t.log_config ?? false} onchange={(e) => update('log_config', e.target.checked)} tooltip="Log the full training configuration at startup." />
						<FormField type="number" fieldPath="training.log_cuda_memory_every_n_steps" value={t.log_cuda_memory_every_n_steps ?? ''} oninput={(e) => update('log_cuda_memory_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Off" tooltip="Log CUDA memory every N steps" />
						<FormField fieldPath="training.training_comment" value={t.training_comment || ''} oninput={(e) => update('training_comment', e.target.value)} placeholder="Optional training comment" tooltip="Saved in checkpoint metadata" />
						{/if}
					</div>
				</FormGroup>

				{#if $advancedMode}
				<FormGroup title="Compile & CUDA">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.compile" checked={t.compile ?? false} onchange={(e) => update('compile', e.target.checked)} tooltip="Enable torch.compile" />
							<FormToggle fieldPath="training.compile_fullgraph" checked={t.compile_fullgraph ?? false} onchange={(e) => update('compile_fullgraph', e.target.checked)} tooltip="Pass --compile_fullgraph." />
							<FormToggle fieldPath="training.cuda_allow_tf32" checked={t.cuda_allow_tf32 ?? false} onchange={(e) => update('cuda_allow_tf32', e.target.checked)} tooltip="Allow TF32 on Ampere+" />
							<FormToggle fieldPath="training.cuda_cudnn_benchmark" checked={t.cuda_cudnn_benchmark ?? false} onchange={(e) => update('cuda_cudnn_benchmark', e.target.checked)} tooltip="cuDNN benchmark mode" />
							<FormToggle fieldPath="training.disable_numpy_memmap" checked={t.disable_numpy_memmap ?? false} onchange={(e) => update('disable_numpy_memmap', e.target.checked)} tooltip="Disable numpy memmap model loading." />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField fieldPath="training.compile_backend" value={t.compile_backend || 'inductor'} oninput={(e) => update('compile_backend', e.target.value)} disabled={!t.compile} tooltip="Compile backend" />
							<FormField fieldPath="training.compile_mode" value={t.compile_mode || ''} oninput={(e) => update('compile_mode', e.target.value)} placeholder="default" disabled={!t.compile} tooltip="Compile mode (default, reduce-overhead, max-autotune)" />
							<FormSelect fieldPath="training.compile_dynamic" value={t.compile_dynamic === true ? 'true' : t.compile_dynamic === false ? '' : t.compile_dynamic || ''} options={[{value:'',label:'Default'},{value:'true',label:'true'},{value:'false',label:'false'},{value:'auto',label:'auto'}]} onchange={(e) => update('compile_dynamic', e.target.value || false)} disabled={!t.compile} tooltip="Value for --compile_dynamic." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.cuda_memory_fraction" value={t.cuda_memory_fraction ?? ''} oninput={(e) => update('cuda_memory_fraction', e.target.value ? Number(e.target.value) : null)} placeholder="None" step="0.05" min={0} max={1} tooltip="Limit CUDA memory fraction" />
							<FormField type="number" fieldPath="training.compile_cache_size_limit" value={t.compile_cache_size_limit ?? ''} oninput={(e) => update('compile_cache_size_limit', e.target.value ? Number(e.target.value) : null)} placeholder="Default" disabled={!t.compile} tooltip="torch.compile cache size limit" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="training.dynamo_backend" value={t.dynamo_backend || 'NO'} oninput={(e) => update('dynamo_backend', e.target.value || 'NO')} tooltip="Accelerate TorchDynamo backend. Default NO disables it." />
							<FormSelect fieldPath="training.dynamo_mode" value={t.dynamo_mode || ''} options={[{value:'',label:'Default'}, 'default', 'reduce-overhead', 'max-autotune']} onchange={(e) => update('dynamo_mode', e.target.value || null)} disabled={(t.dynamo_backend || 'NO').toUpperCase() === 'NO'} tooltip="TorchDynamo mode." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.ddp_timeout" value={t.ddp_timeout ?? ''} oninput={(e) => update('ddp_timeout', e.target.value ? Number(e.target.value) : null)} placeholder="Accelerate default" min={1} tooltip="DDP timeout in minutes." />
						</div>
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.dynamo_fullgraph" checked={t.dynamo_fullgraph ?? false} onchange={(e) => update('dynamo_fullgraph', e.target.checked)} disabled={(t.dynamo_backend || 'NO').toUpperCase() === 'NO'} tooltip="TorchDynamo fullgraph mode." />
							<FormToggle fieldPath="training.dynamo_dynamic" checked={t.dynamo_dynamic ?? false} onchange={(e) => update('dynamo_dynamic', e.target.checked)} disabled={(t.dynamo_backend || 'NO').toUpperCase() === 'NO'} tooltip="TorchDynamo dynamic mode." />
							<FormToggle fieldPath="training.ddp_gradient_as_bucket_view" checked={t.ddp_gradient_as_bucket_view ?? false} onchange={(e) => update('ddp_gradient_as_bucket_view', e.target.checked)} tooltip="Enable DDP gradient_as_bucket_view." />
							<FormToggle fieldPath="training.ddp_static_graph" checked={t.ddp_static_graph ?? false} onchange={(e) => update('ddp_static_graph', e.target.checked)} tooltip="Enable DDP static_graph." />
						</div>
					</div>
				</FormGroup>
				{/if}

				<FormGroup title="Sampling">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.sample_sampling_preset" value={t.sample_sampling_preset || 'defaults'} options={[
								{ value: 'legacy', label: 'Legacy' },
								{ value: 'defaults', label: 'Defaults' },
								{ value: 'ltx20', label: 'LTX 2.0' },
								{ value: 'ltx23', label: 'LTX 2.3' },
								{ value: 'ltx23_hq', label: 'LTX 2.3 HQ' },
								{ value: 'distilled_two_stage', label: 'Distilled Two-Stage' }
							]} onchange={(e) => update('sample_sampling_preset', e.target.value)} tooltip="Validation sample preset. Blank numeric fields below inherit from this preset." />
							<FormSelect fieldPath="training.sample_use_default_negative_prompt" value={t.sample_use_default_negative_prompt === true ? 'true' : t.sample_use_default_negative_prompt === false ? 'false' : ''} options={[
								{ value: '', label: 'Auto' },
								{ value: 'true', label: 'On' },
								{ value: 'false', label: 'Off' }
							]} onchange={(e) => update('sample_use_default_negative_prompt', e.target.value === '' ? null : e.target.value === 'true')} tooltip="Use the built-in negative prompt for validation samples when the preset enables CFG." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.sample_sampler" value={t.sample_sampler || 'auto'} options={[
								{ value: 'auto', label: 'Auto' },
								{ value: 'res_2s', label: 'RES 2S' },
								{ value: 'euler', label: 'Euler' }
							]} onchange={(e) => update('sample_sampler', e.target.value)} tooltip="Validation sampler. Auto uses RES 2S for full LTX presets and Euler for distilled two-stage." />
							<FormSelect fieldPath="training.sample_sigma_schedule" value={t.sample_sigma_schedule || 'auto'} options={[
								{ value: 'auto', label: 'Auto' },
								{ value: 'ltx', label: 'LTX Default' },
								{ value: 'ltx_latent', label: 'LTX Latent Shift' },
								{ value: 'ltx23_distilled', label: 'LTX 2.3 Distilled' }
							]} onchange={(e) => update('sample_sigma_schedule', e.target.value)} tooltip="Validation sigma schedule. Auto uses official validation sigmas, latent-aware sigmas for HQ, and distilled sigmas for distilled presets." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.sample_every_n_steps" value={t.sample_every_n_steps ?? ''} oninput={(e) => update('sample_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Sample every N steps" />
							<FormField type="number" fieldPath="training.sample_every_n_epochs" value={t.sample_every_n_epochs ?? ''} oninput={(e) => update('sample_every_n_epochs', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" tooltip="Sample every N epochs" />
						</div>
						<PathInput fieldPath="training.sample_prompts" value={t.sample_prompts || ''} oninput={(e) => update('sample_prompts', e.target.value)} showFiles tooltip="Optional external prompts file. Leave blank to use prompts defined on the Samples page." invalid={fieldInvalid('training.sample_prompts')} error={fieldError('training.sample_prompts')} />
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.precache_sample_prompts" checked={t.precache_sample_prompts ?? false} onchange={(e) => update('precache_sample_prompts', e.target.checked)} tooltip="Run sample-prompt precaching from the training command." />
							<FormToggle fieldPath="training.use_precached_sample_prompts" checked={t.use_precached_sample_prompts ?? false} onchange={(e) => update('use_precached_sample_prompts', e.target.checked)} tooltip="Use cached text embeddings" />
							<FormToggle fieldPath="training.use_precached_sample_latents" checked={t.use_precached_sample_latents ?? false} onchange={(e) => update('use_precached_sample_latents', e.target.checked)} tooltip="Use cached I2V latents" />
							<FormToggle fieldPath="training.sample_with_offloading" checked={t.sample_with_offloading ?? false} onchange={(e) => update('sample_with_offloading', e.target.checked)} tooltip="Offload during sampling" />
							<FormToggle fieldPath="training.sample_merge_audio" checked={t.sample_merge_audio ?? false} onchange={(e) => update('sample_merge_audio', e.target.checked)} tooltip="Merge audio in samples" />
							<FormToggle fieldPath="training.sample_at_first" checked={t.sample_at_first ?? false} onchange={(e) => update('sample_at_first', e.target.checked)} tooltip="Generate samples before training" />
							<FormToggle fieldPath="training.sample_tiled_vae" checked={t.sample_tiled_vae ?? false} onchange={(e) => update('sample_tiled_vae', e.target.checked)} tooltip="Tiled VAE for sampling (saves VRAM)" />
						</div>
						<PathInput fieldPath="training.sample_prompts_cache" value={t.sample_prompts_cache || ''} oninput={(e) => update('sample_prompts_cache', e.target.value)} showFiles disabled={!t.use_precached_sample_prompts} tooltip="Cached text embeddings dir" />
						<PathInput fieldPath="training.sample_latents_cache" value={t.sample_latents_cache || ''} oninput={(e) => update('sample_latents_cache', e.target.value)} showFiles disabled={!t.use_precached_sample_latents} tooltip="Cached I2V conditioning latents (.pt)" />
						<FormField fieldPath="training.caption_field" value={t.caption_field || ''} oninput={(e) => update('caption_field', e.target.value)} placeholder="caption" tooltip="JSONL metadata field used as sample caption/prompt source." />
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.sample_vae_tile_size" value={t.sample_vae_tile_size ?? ''} oninput={(e) => update('sample_vae_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!t.sample_tiled_vae} tooltip="Tiled VAE spatial tile size" />
							<FormField type="number" fieldPath="training.sample_vae_tile_overlap" value={t.sample_vae_tile_overlap ?? ''} oninput={(e) => update('sample_vae_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="64" disabled={!t.sample_tiled_vae} tooltip="Spatial tile overlap" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.sample_vae_temporal_tile_size" value={t.sample_vae_temporal_tile_size ?? ''} oninput={(e) => update('sample_vae_temporal_tile_size', e.target.value ? Number(e.target.value) : null)} placeholder="Off" disabled={!t.sample_tiled_vae} tooltip="Temporal tile size (0=disabled)" />
							<FormField type="number" fieldPath="training.sample_vae_temporal_tile_overlap" value={t.sample_vae_temporal_tile_overlap ?? ''} oninput={(e) => update('sample_vae_temporal_tile_overlap', e.target.value ? Number(e.target.value) : null)} placeholder="8" disabled={!t.sample_tiled_vae} tooltip="Temporal tile overlap" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.video_cfg_scale" value={t.video_cfg_scale ?? ''} oninput={(e) => update('video_cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video CFG scale override for validation samples" />
							<FormField type="number" fieldPath="training.audio_cfg_scale" value={t.audio_cfg_scale ?? ''} oninput={(e) => update('audio_cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio CFG scale override for validation samples" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.video_rescale_scale" value={t.video_rescale_scale ?? ''} oninput={(e) => update('video_rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video CFG rescale override for validation samples" />
							<FormField type="number" fieldPath="training.audio_rescale_scale" value={t.audio_rescale_scale ?? ''} oninput={(e) => update('audio_rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio CFG rescale override for validation samples" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.video_modality_scale" value={t.video_modality_scale ?? ''} oninput={(e) => update('video_modality_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Video A2V modality guidance override" />
							<FormField type="number" fieldPath="training.audio_modality_scale" value={t.audio_modality_scale ?? ''} oninput={(e) => update('audio_modality_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Audio V2A modality guidance override" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.stg_scale" value={t.stg_scale ?? ''} oninput={(e) => update('stg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Preset" step="0.1" tooltip="Spatio-temporal guidance scale." />
							<FormSelect fieldPath="training.stg_mode" value={t.stg_mode || ''} options={[{value:'',label:'Preset'}, 'video', 'audio', 'both']} onchange={(e) => update('stg_mode', e.target.value || null)} tooltip="Which modality STG perturbs." />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="training.stg_blocks" value={t.stg_blocks || ''} oninput={(e) => update('stg_blocks', e.target.value)} placeholder="0 1 2" tooltip="Transformer block indices for STG. Space separated." />
							<FormField type="number" fieldPath="training.rescale_scale" value={t.rescale_scale ?? ''} oninput={(e) => update('rescale_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Default" step="0.1" tooltip="Shared CFG rescale fallback for video/audio rescale scales." />
						</div>
						{#if $advancedMode}
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.sample_disable_audio" checked={t.sample_disable_audio ?? false} onchange={(e) => update('sample_disable_audio', e.target.checked)} tooltip="Disable audio sampling" />
							<FormToggle fieldPath="training.sample_two_stage" checked={t.sample_two_stage ?? false} onchange={(e) => update('sample_two_stage', e.target.checked)} tooltip="Two-stage spatial upsampling" />
							<FormToggle fieldPath="training.sample_audio_only" checked={t.sample_audio_only ?? false} onchange={(e) => update('sample_audio_only', e.target.checked)} tooltip="Audio-only samples" />
							<FormToggle fieldPath="training.sample_disable_flash_attn" checked={t.sample_disable_flash_attn ?? false} onchange={(e) => update('sample_disable_flash_attn', e.target.checked)} tooltip="Disable flash attention during sampling" />
							<FormToggle fieldPath="training.sample_i2v_token_timestep_mask" checked={t.sample_i2v_token_timestep_mask ?? true} onchange={(e) => update('sample_i2v_token_timestep_mask', e.target.checked)} tooltip="Token timestep mask for I2V sampling" />
							<FormToggle fieldPath="training.sample_audio_subprocess" checked={t.sample_audio_subprocess ?? true} onchange={(e) => update('sample_audio_subprocess', e.target.checked)} tooltip="Run audio sampling in subprocess" />
							<FormToggle fieldPath="training.sample_include_reference" checked={t.sample_include_reference ?? false} onchange={(e) => update('sample_include_reference', e.target.checked)} tooltip="Show V2V reference side-by-side" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.reference_downscale" value={t.reference_downscale ?? 1} oninput={(e) => update('reference_downscale', Number(e.target.value))} min={1} disabled={!t.sample_include_reference} tooltip="V2V reference downscale factor" />
							<FormField type="number" fieldPath="training.reference_frames" value={t.reference_frames ?? 1} oninput={(e) => update('reference_frames', Number(e.target.value))} min={1} disabled={!t.sample_include_reference} tooltip="Number of reference frames" />
						</div>
						{#if t.sample_two_stage}
						<PathInput fieldPath="training.spatial_upsampler_path" value={t.spatial_upsampler_path || ''} oninput={(e) => update('spatial_upsampler_path', e.target.value)} showFiles tooltip="Spatial upsampler model path" invalid={fieldInvalid('training.spatial_upsampler_path')} error={fieldError('training.spatial_upsampler_path')} />
						<PathInput fieldPath="training.distilled_lora_path" value={t.distilled_lora_path || ''} oninput={(e) => update('distilled_lora_path', e.target.value)} showFiles tooltip="Distilled LoRA for stage 2" />
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="training.sample_stage2_steps" value={t.sample_stage2_steps ?? 3} oninput={(e) => update('sample_stage2_steps', Number(e.target.value))} min={1} tooltip="Number of denoising steps for stage 2" />
							<FormField type="number" fieldPath="training.sample_stage1_distilled_lora_multiplier" value={t.sample_stage1_distilled_lora_multiplier ?? ''} oninput={(e) => update('sample_stage1_distilled_lora_multiplier', e.target.value ? Number(e.target.value) : null)} min={0} step="0.05" placeholder="Auto" tooltip="Distilled LoRA multiplier for stage 1. Auto uses 0.25 with RES 2S." />
							<FormField type="number" fieldPath="training.sample_stage2_distilled_lora_multiplier" value={t.sample_stage2_distilled_lora_multiplier ?? ''} oninput={(e) => update('sample_stage2_distilled_lora_multiplier', e.target.value ? Number(e.target.value) : null)} min={0} step="0.05" placeholder="Auto" tooltip="Distilled LoRA multiplier for stage 2. Auto uses 0.5 with RES 2S." />
						</div>
						{/if}
						{/if}
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="training.width" value={t.width ?? ''} oninput={(e) => update('width', e.target.value ? Number(e.target.value) : null)} min={64} step={64} placeholder="Preset" tooltip="Sample width override" />
							<FormField type="number" fieldPath="training.height" value={t.height ?? ''} oninput={(e) => update('height', e.target.value ? Number(e.target.value) : null)} min={64} step={64} placeholder="Preset" tooltip="Sample height override" />
							<FormField type="number" fieldPath="training.sample_num_frames" value={t.sample_num_frames ?? ''} oninput={(e) => update('sample_num_frames', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Preset" tooltip="Sample frame count override" />
						</div>
					</div>
				</FormGroup>

				{#if $advancedMode}
				<FormGroup title="Loss & Misc">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.video_loss_weight" value={t.video_loss_weight ?? 1.0} oninput={(e) => update('video_loss_weight', Number(e.target.value))} step="0.1" tooltip="Video loss multiplier" />
							<FormField type="number" fieldPath="training.audio_loss_weight" value={t.audio_loss_weight ?? 1.0} oninput={(e) => update('audio_loss_weight', Number(e.target.value))} step="0.1" tooltip="Audio loss multiplier" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Metadata">
					<div class="space-y-2 pt-2">
						<FormField fieldPath="training.metadata_title" value={t.metadata_title || ''} oninput={(e) => update('metadata_title', e.target.value)} placeholder="LoRA name" tooltip="Model card title" />
						<FormField fieldPath="training.metadata_author" value={t.metadata_author || ''} oninput={(e) => update('metadata_author', e.target.value)} tooltip="Model card author" />
						<FormField fieldPath="training.metadata_description" value={t.metadata_description || ''} oninput={(e) => update('metadata_description', e.target.value)} tooltip="Model card description" />
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="training.metadata_license" value={t.metadata_license || ''} oninput={(e) => update('metadata_license', e.target.value)} placeholder="e.g. mit" tooltip="SPDX license identifier" />
							<FormField fieldPath="training.metadata_tags" value={t.metadata_tags || ''} oninput={(e) => update('metadata_tags', e.target.value)} placeholder="tag1,tag2" tooltip="Comma-separated tags" />
						</div>
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="training.metadata_reso" value={t.metadata_reso || ''} oninput={(e) => update('metadata_reso', e.target.value)} placeholder="1024,1024" tooltip="Metadata resolution string." />
							<FormField fieldPath="training.metadata_arch" value={t.metadata_arch || ''} oninput={(e) => update('metadata_arch', e.target.value)} placeholder="Optional" tooltip="Custom architecture metadata." />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="HuggingFace Upload">
					<div class="space-y-2 pt-2">
						<FormField fieldPath="training.huggingface_repo_id" value={t.huggingface_repo_id || ''} oninput={(e) => update('huggingface_repo_id', e.target.value)} placeholder="user/model" tooltip="HuggingFace repo id (user/model)" />
						<div class="grid grid-cols-2 gap-2">
							<FormField fieldPath="training.huggingface_repo_type" value={t.huggingface_repo_type || ''} oninput={(e) => update('huggingface_repo_type', e.target.value)} placeholder="model" tooltip="HuggingFace repo type" />
							<FormField fieldPath="training.huggingface_repo_visibility" value={t.huggingface_repo_visibility || ''} oninput={(e) => update('huggingface_repo_visibility', e.target.value)} placeholder="private" tooltip="Repo visibility (public/private)" />
						</div>
						<FormField fieldPath="training.huggingface_path_in_repo" value={t.huggingface_path_in_repo || ''} oninput={(e) => update('huggingface_path_in_repo', e.target.value)} placeholder="/" tooltip="Upload path within repo" />
						<FormField fieldPath="training.huggingface_token" value={t.huggingface_token || ''} oninput={(e) => update('huggingface_token', e.target.value)} placeholder="hf_..." tooltip="HuggingFace API token (or set HF_TOKEN env)" />
						<div class="grid grid-cols-3 gap-x-4 gap-y-1">
							<FormToggle fieldPath="training.save_state_to_huggingface" checked={t.save_state_to_huggingface ?? false} onchange={(e) => update('save_state_to_huggingface', e.target.checked)} tooltip="Also upload optimizer state" />
							<FormToggle fieldPath="training.resume_from_huggingface" checked={t.resume_from_huggingface ?? false} onchange={(e) => update('resume_from_huggingface', e.target.checked)} tooltip="Resume training from HuggingFace" />
							<FormToggle fieldPath="training.async_upload" checked={t.async_upload ?? false} onchange={(e) => update('async_upload', e.target.checked)} tooltip="Upload checkpoints asynchronously" />
						</div>
					</div>
				</FormGroup>
				{/if}
			</div>
		</div>

		{#if $advancedMode}
			<FormGroup title="CLI Passthrough">
				<div class="space-y-2 pt-2">
					<FormField fieldPath="training.accelerate_extra_args" value={t.accelerate_extra_args || ''} oninput={(e) => update('accelerate_extra_args', e.target.value)} placeholder="--num_processes 2 --main_process_port 29501" tooltip="Extra arguments appended to `accelerate launch` before the training script path." />
					<FormField fieldPath="training.extra_args" value={t.extra_args || ''} oninput={(e) => update('extra_args', e.target.value)} placeholder="--flag value --other_flag" tooltip="Extra arguments appended to the LTX-2 training script command. Use this for any CLI option without a dedicated dashboard control." />
				</div>
			</FormGroup>
		{/if}

		<!-- Controls -->
		{#if hasValidationIssues}
			<div class="p-3 space-y-2" style="background: {trainingValidation.errors?.length ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border: 1px solid {trainingValidation.errors?.length ? 'var(--danger)' : 'var(--border)'}; border-radius: var(--radius-sm);">
				<div class="flex items-center justify-between gap-3">
					<div class="text-[12px] font-medium" style="color: {trainingValidation.errors?.length ? 'var(--danger)' : 'var(--text-primary)'};">
						{trainingValidation.summary}
					</div>
					{#if hasDatasetValidationErrors}
						<a href="/dataset" class="text-[12px] font-medium" style="color: var(--accent);">Open Dataset</a>
					{/if}
				</div>
				{#if trainingValidation.errors?.length}
					<div class="space-y-1">
						{#each trainingValidation.errors as issue}
							<div class="text-[12px]" style="color: var(--text-primary);">
								{issue.message}
							</div>
						{/each}
					</div>
				{/if}
				{#if trainingValidation.warnings?.length}
					<div class="space-y-1">
						{#each trainingValidation.warnings as issue}
							<div class="text-[12px]" style="color: var(--text-secondary);">
								{issue.message}
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
		<div class="py-4 flex items-center gap-4">
			<ProcessControls processType="training" status={trainingStatus} onStart={startTraining} onStop={() => stopProcess('training')} />
			<div class="flex-1"></div>
			{#if trainingStatus.state === 'running' || trainingStatus.state === 'stopping' || trainingStatus.state === 'finished'}
				<a href="/training/dashboard" class="px-4 py-2 text-[13px] font-medium" style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);">Dashboard</a>
			{/if}
		</div>

		{#if $advancedMode}
			<CommandPanel processType="training" defaultFilename="train.bat" />
		{/if}
	</div>
{/if}
