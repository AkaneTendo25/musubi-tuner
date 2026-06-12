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
	import { projectConfig, projectLoaded, updateSection, saveProjectNow } from '$lib/stores/project.js';
	import { processStatuses, processValidation, startProcess, stopProcess, validateProcess } from '$lib/stores/processes.js';
	import { defaultModelDir, describeExactModelScan, effectiveGemmaRoot, effectiveGemmaSafetensors, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { getModelDownloadPresets, checkPathExists, scanCheckpointsWithProgress, cancelCheckpointScan, formatCheckpointScanStatus, modelDownloadTooltip } from '$lib/utils/modelDownloads.js';
	import { cancelSharedModelDownload, modelDownloadState, resumeModelDownloadPolling, startSharedModelDownload } from '$lib/stores/modelDownloads.js';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	const processType = 'full_finetune';
	const section = 'full_finetune';

	const optimizerOptions = [
		'Adafactor',
		'QGaLoreAdamW8bit',
		'APOLLOAdamW',
		'QAPOLLOAdamW',
		'BAdam',
		'ProdigyPlusScheduleFree',
		'AdamW8bit',
		'PagedAdamW8bit',
		'AdamW',
		'CAME',
		'CAME8bit',
		'SinkSGD',
		'SinkSGD_adv',
		'torchao_adamw8bit',
		'torchao_adamw4bit',
		'torchao_adamwfp8',
		'optimi_adamw',
		'optimi_stableadamw',
		'optimi_lion',
		'optimi_adan',
		'lion',
		'lion8bit',
		'lion8bitint8',
		'smmf'
	];
	const prodigyPlusOptimizerArgs = 'betas=(0.9,0.99) beta3=None weight_decay=0.0 weight_decay_by_lr=True use_bias_correction=False d0=1e-6 d_coef=1.0 prodigy_steps=0 use_speed=False eps=1e-8 split_groups=True split_groups_mean=False factored=True factored_fp32=True use_stableadamw=True use_cautious=False use_grams=False use_adopt=False d_limiter=True stochastic_rounding=True use_schedulefree=True schedulefree_c=0.0 use_orthograd=False';

	let t = $derived($projectConfig?.full_finetune || {});
	let status = $derived($processStatuses.full_finetune || { state: 'idle', exit_code: null });
	let validation = $derived($processValidation.full_finetune || { ok: true, summary: '', errors: [], warnings: [], field_errors: {}, field_warnings: {} });
	let hasValidationIssues = $derived((validation.errors?.length || 0) > 0 || (validation.warnings?.length || 0) > 0);
	let validationTimer = null;
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

	onMount(async () => {
		try {
			const res = await fetch('/api/fs/cwd');
			if (res.ok) cwd = (await res.json()).cwd || '';
		} catch {}
		try {
			downloadPresets = await getModelDownloadPresets();
		} catch {}
		resumeModelDownloadPolling();
		return () => {
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
	let downloadState = $derived($modelDownloadState.state || '');
	let modelStatus = $derived($modelDownloadState.message || '');
	let modelStatusTone = $derived($modelDownloadState.tone || 'muted');
	let hasActiveDownload = $derived(Boolean($modelDownloadState.jobId) && ['queued', 'running', 'cancelling'].includes(downloadState));

	function update(key, value) {
		updateSection(section, key, value);
	}

	function fieldError(path) {
		return validation.field_errors?.[path]?.[0] || '';
	}

	function fieldInvalid(path) {
		return Boolean(validation.field_errors?.[path]?.length);
	}

	function numberOrNull(value) {
		return value === '' || value == null ? null : Number(value);
	}

	function relatedScanTargets() {
		return {
			ltx2: resolvedLtx,
			gemma: scanTargetGemmaRoot,
			gemma_safetensors: activeGemmaSafetensors
		};
	}

	async function downloadModel(preset) {
		const targetPath = preset === 'ltxav' ? resolvedLtx : resolvedGemma;
		if (!targetPath) return;
		projectConfig.update((config) => config ? { ...config, model_dir: modelDir } : config);
		await saveProjectNow();
		await startSharedModelDownload({ preset, targetPath, modelDir, section });
	}

	async function stopDownload() {
		await cancelSharedModelDownload();
	}

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
		if (ltxScanJobId) await cancelCheckpointScan(ltxScanJobId).catch(() => {});
	}

	async function stopGemmaScan() {
		if (gemmaScanJobId) await cancelCheckpointScan(gemmaScanJobId).catch(() => {});
	}

	async function stopGemmaSafetensorsScan() {
		if (gemmaSafetensorsScanJobId) await cancelCheckpointScan(gemmaSafetensorsScanJobId).catch(() => {});
	}

	function applyAdafactorRecommendation() {
		update('optimizer_type', 'Adafactor');
		update('optimizer_args', 'relative_step=False scale_parameter=False warmup_init=False stochastic_rounding=True');
		update('base_optimizer_args', '');
		update('qgalore_full_ft', false);
		update('learning_rate', 1e-6);
		update('max_grad_norm', 1.0);
		update('fused_backward_pass', true);
		update('full_bf16', true);
		update('full_fp16', false);
	}

	function applyBadamRecommendation() {
		update('optimizer_type', 'BAdam');
		update('optimizer_args', 'base_optimizer_type=AdamW8bit switch_block_every=25 switch_mode=ascending block_group_size=2 include_non_block=True use_fp32_active_copy=True purge_inactive_state=True reset_state_on_switch=True use_gradient_release=True');
		update('base_optimizer_args', '');
		update('qgalore_full_ft', false);
		update('learning_rate', 1e-6);
		update('max_grad_norm', 1.0);
		update('fused_backward_pass', true);
		update('full_bf16', true);
		update('full_fp16', false);
	}

	function applyQgaloreRecommendation() {
		update('qgalore_full_ft', true);
		update('optimizer_type', 'QGaLoreAdamW8bit');
		update('optimizer_args', '');
		update('base_optimizer_args', '');
		update('learning_rate', 1e-5);
		update('max_grad_norm', 0.0);
		update('fused_backward_pass', true);
		update('full_bf16', true);
		update('full_fp16', false);
		update('fp8_base', false);
		update('fp8_scaled', false);
		update('nf4_base', false);
		update('qgalore_targets', 'video');
		update('qgalore_rank', 256);
		update('qgalore_update_proj_gap', 200);
		update('qgalore_scale', 0.25);
		update('qgalore_proj_type', 'std');
		update('qgalore_proj_quant', true);
		update('qgalore_proj_bits', 4);
		update('qgalore_proj_group_size', 256);
		update('qgalore_weight_bits', 8);
		update('qgalore_weight_group_size', 0);
		update('qgalore_stochastic_round', true);
		update('qgalore_load_device', 'cuda');
		update('qgalore_svd_method', 'lowrank');
		update('qgalore_svd_oversampling', 32);
		update('qgalore_svd_niter', 1);
		update('qgalore_dequantize_save', true);
		update('qgalore_streaming_dequantize_save', true);
		update('qgalore_streaming_dequantize_device', 'cpu');
		update('qgalore_cos_threshold', 0.4);
		update('qgalore_gamma_proj', 2.0);
		update('qgalore_queue_size', 5);
	}

	function applyApolloRecommendation(quantized = false) {
		update('optimizer_type', quantized ? 'QAPOLLOAdamW' : 'APOLLOAdamW');
		update('optimizer_args', '');
		update('base_optimizer_args', '');
		update('qgalore_full_ft', quantized);
		update('learning_rate', quantized ? 1e-5 : 1e-6);
		update('max_grad_norm', quantized ? 0.0 : 1.0);
		update('fused_backward_pass', true);
		update('full_bf16', true);
		update('full_fp16', false);
		update('apollo_rank', 256);
		update('apollo_update_proj_gap', 200);
		update('apollo_scale', 1.0);
		update('apollo_proj', 'random');
		update('apollo_proj_type', 'std');
		update('apollo_scale_type', 'channel');
		update('qapollo_optim_bits', 8);
		if (quantized) {
			update('fp8_base', false);
			update('fp8_scaled', false);
			update('nf4_base', false);
			update('qgalore_targets', 'video');
			update('qgalore_load_device', 'cuda');
			update('qgalore_weight_bits', 8);
			update('qgalore_weight_group_size', 0);
			update('qgalore_stochastic_round', true);
			update('qgalore_dequantize_save', true);
			update('qgalore_streaming_dequantize_save', true);
			update('qgalore_streaming_dequantize_device', 'cpu');
		}
	}

	function optimizerAlias(value) {
		return String(value || '').toLowerCase().replace(/[-_\s]/g, '');
	}

	function isProdigyPlusOptimizer(value) {
		const alias = optimizerAlias(value);
		return alias === 'pplus' || alias === 'prodigyplus' || alias === 'prodigyplusschedulefree';
	}

	function applyProdigyPlusRecommendation() {
		update('optimizer_type', 'ProdigyPlusScheduleFree');
		update('optimizer_args', prodigyPlusOptimizerArgs);
		update('base_optimizer_args', '');
		update('qgalore_full_ft', false);
		update('learning_rate', 1.0);
		update('lr_scheduler', 'constant');
		update('lr_warmup_steps', 0);
		update('lr_decay_steps', 0);
		update('max_grad_norm', 0.0);
		update('fused_backward_pass', true);
		update('full_bf16', true);
		update('full_fp16', false);
	}

	function applyRecommendedOptimizerArgs() {
		const optimizer = (t.optimizer_type || '').toLowerCase();
		if (isProdigyPlusOptimizer(optimizer)) {
			applyProdigyPlusRecommendation();
		} else if (optimizer.includes('qapollo')) {
			applyApolloRecommendation(true);
		} else if (optimizer.includes('apollo')) {
			applyApolloRecommendation(false);
		} else if (optimizer.includes('qgalore')) {
			applyQgaloreRecommendation();
		} else if (optimizer.includes('badam')) {
			applyBadamRecommendation();
		} else if (optimizer.includes('adafactor')) {
			applyAdafactorRecommendation();
		} else {
			update('optimizer_args', '');
			update('base_optimizer_args', '');
			update('qgalore_full_ft', false);
			update('fused_backward_pass', true);
			update('full_bf16', true);
			update('full_fp16', false);
		}
	}

	async function startFullFinetune() {
		await startProcess(processType);
		await goto('/training/dashboard');
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

	$effect(() => {
		if (!$projectLoaded || !$projectConfig) return;

		clearTimeout(validationTimer);
		const configSnapshot = $projectConfig;
		validationTimer = setTimeout(() => {
			validateProcess(processType, configSnapshot).catch(() => {});
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
	{#if hasValidationIssues}
		<div class="p-3 space-y-2" style="background: {validation.errors?.length ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border: 1px solid {validation.errors?.length ? 'var(--danger)' : 'var(--border)'}; border-radius: var(--radius-sm);">
			<div class="text-[12px] font-medium" style="color: {validation.errors?.length ? 'var(--danger)' : 'var(--text-primary)'};">
				{validation.summary}
			</div>
			{#each validation.errors || [] as issue}
				<div class="text-[12px]" style="color: var(--text-primary);">{issue.message}</div>
			{/each}
			{#each validation.warnings || [] as issue}
				<div class="text-[12px]" style="color: var(--text-secondary);">{issue.message}</div>
			{/each}
		</div>
	{/if}

	<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
		<div class="space-y-3">
		<FormGroup title="Model">
			<div class="space-y-2 pt-2">
				<CheckpointInput fieldPath="full_finetune.ltx2_checkpoint" label="LTX-2 Checkpoint" value={t.ltx2_checkpoint || ''} onchange={(value) => update('ltx2_checkpoint', value)} showFiles tooltip="Path to LTX-2 checkpoint" invalid={fieldInvalid('full_finetune.ltx2_checkpoint')} error={fieldError('full_finetune.ltx2_checkpoint')} actionLabel="D" actionBusyLabel="..." actionDisabled={hasActiveDownload || ltxDownloadExists} actionTooltip={modelDownloadTooltip(downloadPresets, 'ltxav', resolvedLtx, ltxDownloadExists)} onaction={() => downloadModel('ltxav')} />
				<ModelPathStatus exists={ltxDownloadExists} foundPath={foundLtxPath} disabled={hasActiveDownload} scanning={scanningLtx} scanMessage={ltxScanMessage} scanTone={ltxScanTone} onscan={scanLtx} oncancel={stopLtxScan} onusefound={(path) => update('ltx2_checkpoint', path)} />
				<CheckpointInput fieldPath="full_finetune.gemma_root" label="Gemma Root" value={t.gemma_root || ''} onchange={(value) => update('gemma_root', value)} disabled={gemmaRootDisabled} tooltip={gemmaRootDisabled ? 'Ignored while Gemma Safetensors is set' : 'Gemma text encoder directory'} invalid={fieldInvalid('full_finetune.gemma_root')} error={fieldError('full_finetune.gemma_root')} actionLabel="D" actionBusyLabel="..." actionDisabled={gemmaRootDisabled || hasActiveDownload || gemmaDownloadExists} actionTooltip={gemmaRootDisabled ? 'Gemma Safetensors is active' : modelDownloadTooltip(downloadPresets, 'gemma-unsloth', resolvedGemma, gemmaDownloadExists)} onaction={() => downloadModel('gemma-unsloth')} />
				<ModelPathStatus exists={gemmaRootDisabled || gemmaDownloadExists} foundPath={foundGemmaPath} disabled={gemmaRootDisabled || hasActiveDownload} scanning={scanningGemma} scanMessage={gemmaScanMessage} scanTone={gemmaScanTone} onscan={scanGemma} oncancel={stopGemmaScan} onusefound={(path) => { update('gemma_root', path); update('gemma_safetensors', ''); }} />
				<PathInput fieldPath="full_finetune.gemma_safetensors" value={t.gemma_safetensors || ''} oninput={(e) => update('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" invalid={fieldInvalid('full_finetune.gemma_safetensors')} error={fieldError('full_finetune.gemma_safetensors')} />
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
				<div class="grid grid-cols-2 gap-2">
					<FormSelect fieldPath="full_finetune.ltx2_mode" value={t.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} />
					<FormSelect fieldPath="full_finetune.ltx_version" value={t.ltx_version || '2.3'} options={['2.0', '2.3']} onchange={(e) => update('ltx_version', e.target.value)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormSelect fieldPath="full_finetune.ltx_version_check_mode" value={t.ltx_version_check_mode || 'warn'} options={['off', 'warn', 'error']} onchange={(e) => update('ltx_version_check_mode', e.target.value)} />
					<FormSelect fieldPath="full_finetune.vae_dtype" value={t.vae_dtype || ''} options={[{ value: '', label: 'Default' }, 'bfloat16', 'float16', 'float32']} onchange={(e) => update('vae_dtype', e.target.value || null)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Dataset">
			<div class="space-y-2 pt-2">
				<PathInput fieldPath="full_finetune.config_file" value={t.config_file || ''} oninput={(e) => update('config_file', e.target.value)} showFiles={true} />
				<PathInput fieldPath="full_finetune.dataset_config" value={t.dataset_config || ''} oninput={(e) => update('dataset_config', e.target.value)} showFiles={true} />
				<PathInput fieldPath="full_finetune.dataset_manifest" value={t.dataset_manifest || ''} oninput={(e) => update('dataset_manifest', e.target.value)} showFiles={true} />
				<div class="grid grid-cols-3 gap-2">
					<FormField type="number" fieldPath="full_finetune.gradient_accumulation_steps" value={t.gradient_accumulation_steps ?? 1} oninput={(e) => update('gradient_accumulation_steps', Number(e.target.value))} min={1} />
					<FormSelect fieldPath="full_finetune.accumulation_group_by" value={t.accumulation_group_by || 'none'} options={['none', 'frames', 'bucket', 'dataset']} onchange={(e) => update('accumulation_group_by', e.target.value)} />
					<FormSelect fieldPath="full_finetune.accumulation_group_remainder" value={t.accumulation_group_remainder || 'drop'} options={['drop', 'pad', 'allow_mixed']} onchange={(e) => update('accumulation_group_remainder', e.target.value)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.max_data_loader_n_workers" value={t.max_data_loader_n_workers ?? ''} oninput={(e) => update('max_data_loader_n_workers', numberOrNull(e.target.value))} min={0} />
					<FormToggle fieldPath="full_finetune.persistent_data_loader_workers" checked={t.persistent_data_loader_workers ?? false} onchange={(e) => update('persistent_data_loader_workers', e.target.checked)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Optimizer">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-3 gap-2">
					<div class="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-1.5">
						<FormCombobox fieldPath="full_finetune.optimizer_type" value={t.optimizer_type || ''} options={optimizerOptions} oninput={(e) => update('optimizer_type', e.target.value)} />
						<button
							type="button"
							onclick={applyRecommendedOptimizerArgs}
							data-tooltip="Set recommended settings for the selected optimizer"
							class="mb-0 text-sm font-medium disabled:opacity-40 flex-shrink-0"
							style="height: 38px; min-width: 36px; padding: 0 10px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
							onmouseenter={(e) => e.currentTarget.style.background = 'var(--border)'}
							onmouseleave={(e) => e.currentTarget.style.background = 'var(--bg-elevated)'}
						>Set</button>
					</div>
					<FormField type="number" fieldPath="full_finetune.learning_rate" value={t.learning_rate ?? 1e-6} oninput={(e) => update('learning_rate', Number(e.target.value))} step="0.000001" min={0} />
					<FormField type="number" fieldPath="full_finetune.max_grad_norm" value={t.max_grad_norm ?? 1.0} oninput={(e) => update('max_grad_norm', Number(e.target.value))} step="0.1" min={0} invalid={fieldInvalid('full_finetune.max_grad_norm')} error={fieldError('full_finetune.max_grad_norm')} />
				</div>
				<FormField fieldPath="full_finetune.optimizer_args" value={t.optimizer_args || ''} oninput={(e) => update('optimizer_args', e.target.value)} placeholder="key=value ..." />
				<FormField fieldPath="full_finetune.base_optimizer_args" value={t.base_optimizer_args || ''} oninput={(e) => update('base_optimizer_args', e.target.value)} placeholder="key=value ..." />
				<div class="grid grid-cols-2 gap-2">
					<FormSelect fieldPath="full_finetune.lr_scheduler" value={t.lr_scheduler || 'constant_with_warmup'} options={['constant', 'constant_with_warmup', 'linear', 'cosine', 'cosine_with_restarts', 'polynomial']} onchange={(e) => update('lr_scheduler', e.target.value)} />
					<FormField type="number" fieldPath="full_finetune.lr_warmup_steps" value={t.lr_warmup_steps ?? 500} oninput={(e) => update('lr_warmup_steps', Number(e.target.value))} min={0} />
					<FormField type="number" fieldPath="full_finetune.lr_decay_steps" value={t.lr_decay_steps ?? ''} oninput={(e) => update('lr_decay_steps', numberOrNull(e.target.value))} min={0} />
					<FormField type="number" fieldPath="full_finetune.lr_scheduler_min_lr_ratio" value={t.lr_scheduler_min_lr_ratio ?? ''} oninput={(e) => update('lr_scheduler_min_lr_ratio', numberOrNull(e.target.value))} min={0} step="0.01" />
				</div>
				<FormField fieldPath="full_finetune.lr_scheduler_args" value={t.lr_scheduler_args || ''} oninput={(e) => update('lr_scheduler_args', e.target.value)} placeholder="key=value ..." />
			</div>
		</FormGroup>

		<FormGroup title="Schedule">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-3 gap-2">
					<FormField type="number" fieldPath="full_finetune.max_train_steps" value={t.max_train_steps ?? 50000} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.max_train_epochs" value={t.max_train_epochs ?? ''} oninput={(e) => update('max_train_epochs', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.seed" value={t.seed ?? ''} oninput={(e) => update('seed', numberOrNull(e.target.value))} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormSelect fieldPath="full_finetune.timestep_sampling" value={t.timestep_sampling || 'sigma'} options={['sigma', 'shifted_logit_normal', 'uniform', 'logit_normal', 'mode']} onchange={(e) => update('timestep_sampling', e.target.value)} />
					<FormField type="number" fieldPath="full_finetune.discrete_flow_shift" value={t.discrete_flow_shift ?? 1.0} oninput={(e) => update('discrete_flow_shift', Number(e.target.value))} step="0.1" />
					<FormField type="number" fieldPath="full_finetune.guidance_scale" value={t.guidance_scale ?? ''} oninput={(e) => update('guidance_scale', numberOrNull(e.target.value))} step="0.1" />
					<FormField type="number" fieldPath="full_finetune.mode_scale" value={t.mode_scale ?? ''} oninput={(e) => update('mode_scale', numberOrNull(e.target.value))} step="0.01" />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.logit_mean" value={t.logit_mean ?? ''} oninput={(e) => update('logit_mean', numberOrNull(e.target.value))} step="0.1" />
					<FormField type="number" fieldPath="full_finetune.logit_std" value={t.logit_std ?? ''} oninput={(e) => update('logit_std', numberOrNull(e.target.value))} step="0.1" />
					<FormField type="number" fieldPath="full_finetune.min_timestep" value={t.min_timestep ?? ''} oninput={(e) => update('min_timestep', numberOrNull(e.target.value))} step="0.01" />
					<FormField type="number" fieldPath="full_finetune.max_timestep" value={t.max_timestep ?? ''} oninput={(e) => update('max_timestep', numberOrNull(e.target.value))} step="0.01" />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Memory">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.full_bf16" checked={t.full_bf16 ?? true} onchange={(e) => update('full_bf16', e.target.checked)} invalid={fieldInvalid('full_finetune.full_bf16')} />
					<FormToggle fieldPath="full_finetune.full_fp16" checked={t.full_fp16 ?? false} onchange={(e) => update('full_fp16', e.target.checked)} invalid={fieldInvalid('full_finetune.full_fp16')} />
					<FormToggle fieldPath="full_finetune.gradient_checkpointing" checked={t.gradient_checkpointing ?? true} onchange={(e) => update('gradient_checkpointing', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.fused_backward_pass" checked={t.fused_backward_pass ?? true} onchange={(e) => update('fused_backward_pass', e.target.checked)} invalid={fieldInvalid('full_finetune.fused_backward_pass')} />
				</div>
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.mem_eff_save" checked={t.mem_eff_save ?? true} onchange={(e) => update('mem_eff_save', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.flash_attn" checked={t.flash_attn ?? true} onchange={(e) => update('flash_attn', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.sdpa" checked={t.sdpa ?? false} onchange={(e) => update('sdpa', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.xformers" checked={t.xformers ?? false} onchange={(e) => update('xformers', e.target.checked)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.blocks_to_swap" value={t.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', numberOrNull(e.target.value))} min={0} max={47} />
					<FormSelect fieldPath="full_finetune.ltx2_finetune_block_swap_mode" value={t.ltx2_finetune_block_swap_mode || 'default'} options={['default', 'linear', 'full']} onchange={(e) => update('ltx2_finetune_block_swap_mode', e.target.value)} />
					<FormField fieldPath="full_finetune.ltx2_finetune_block_swap_mask" value={t.ltx2_finetune_block_swap_mask || 'all'} oninput={(e) => update('ltx2_finetune_block_swap_mask', e.target.value)} placeholder="all, ff, attn" />
					<FormField type="number" fieldPath="full_finetune.ffn_chunk_size" value={t.ffn_chunk_size ?? 0} oninput={(e) => update('ffn_chunk_size', Number(e.target.value))} min={0} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Quantization">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.fp8_base" checked={t.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} invalid={fieldInvalid('full_finetune.fp8_base')} />
					<FormToggle fieldPath="full_finetune.fp8_scaled" checked={t.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} invalid={fieldInvalid('full_finetune.fp8_scaled')} />
					<FormToggle fieldPath="full_finetune.nf4_base" checked={t.nf4_base ?? false} onchange={(e) => update('nf4_base', e.target.checked)} invalid={fieldInvalid('full_finetune.nf4_base')} />
					<FormToggle fieldPath="full_finetune.gemma_fp8_weight_offload" checked={t.gemma_fp8_weight_offload ?? true} onchange={(e) => update('gemma_fp8_weight_offload', e.target.checked)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField fieldPath="full_finetune.fp8_keep_blocks" value={t.fp8_keep_blocks || ''} oninput={(e) => update('fp8_keep_blocks', e.target.value)} placeholder="0-2,45" />
					<FormField type="number" fieldPath="full_finetune.nf4_block_size" value={t.nf4_block_size ?? 32} oninput={(e) => update('nf4_block_size', Number(e.target.value))} min={1} />
					<FormSelect fieldPath="full_finetune.gemma_bnb_4bit_quant_type" value={t.gemma_bnb_4bit_quant_type || 'nf4'} options={['nf4', 'fp4']} onchange={(e) => update('gemma_bnb_4bit_quant_type', e.target.value)} />
					<FormToggle fieldPath="full_finetune.gemma_bnb_4bit_disable_double_quant" checked={t.gemma_bnb_4bit_disable_double_quant ?? false} onchange={(e) => update('gemma_bnb_4bit_disable_double_quant', e.target.checked)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Q-GaLore">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.qgalore_full_ft" checked={t.qgalore_full_ft ?? false} onchange={(e) => update('qgalore_full_ft', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.qgalore_proj_quant" checked={t.qgalore_proj_quant ?? true} onchange={(e) => update('qgalore_proj_quant', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.qgalore_stochastic_round" checked={t.qgalore_stochastic_round ?? true} onchange={(e) => update('qgalore_stochastic_round', e.target.checked)} />
					<FormSelect fieldPath="full_finetune.qgalore_load_device" value={t.qgalore_load_device || 'cuda'} options={['cuda', 'cpu']} onchange={(e) => update('qgalore_load_device', e.target.value)} />
					<FormToggle fieldPath="full_finetune.qgalore_dequantize_save" checked={t.qgalore_dequantize_save ?? true} onchange={(e) => update('qgalore_dequantize_save', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.qgalore_streaming_dequantize_save" checked={t.qgalore_streaming_dequantize_save ?? false} onchange={(e) => update('qgalore_streaming_dequantize_save', e.target.checked)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField fieldPath="full_finetune.qgalore_targets" value={t.qgalore_targets || 'video'} oninput={(e) => update('qgalore_targets', e.target.value)} />
					<FormField type="number" fieldPath="full_finetune.qgalore_rank" value={t.qgalore_rank ?? 256} oninput={(e) => update('qgalore_rank', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.qgalore_update_proj_gap" value={t.qgalore_update_proj_gap ?? 200} oninput={(e) => update('qgalore_update_proj_gap', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.qgalore_scale" value={t.qgalore_scale ?? 0.25} oninput={(e) => update('qgalore_scale', Number(e.target.value))} step="0.01" min={0} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormSelect fieldPath="full_finetune.qgalore_svd_method" value={t.qgalore_svd_method || 'full'} options={['full', 'lowrank']} onchange={(e) => update('qgalore_svd_method', e.target.value)} />
					<FormSelect fieldPath="full_finetune.qgalore_streaming_dequantize_device" value={t.qgalore_streaming_dequantize_device || 'cpu'} options={['cpu', 'cuda']} onchange={(e) => update('qgalore_streaming_dequantize_device', e.target.value)} disabled={!(t.qgalore_streaming_dequantize_save ?? false)} />
					<FormSelect fieldPath="full_finetune.qgalore_proj_type" value={t.qgalore_proj_type || 'std'} options={['std']} onchange={(e) => update('qgalore_proj_type', e.target.value)} />
					<FormField type="number" fieldPath="full_finetune.qgalore_proj_bits" value={t.qgalore_proj_bits ?? 4} oninput={(e) => update('qgalore_proj_bits', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.qgalore_proj_group_size" value={t.qgalore_proj_group_size ?? 256} oninput={(e) => update('qgalore_proj_group_size', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.qgalore_weight_bits" value={t.qgalore_weight_bits ?? 8} oninput={(e) => update('qgalore_weight_bits', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.qgalore_weight_group_size" value={t.qgalore_weight_group_size ?? 0} oninput={(e) => update('qgalore_weight_group_size', Number(e.target.value))} min={0} tooltip="0 = row-wise (one scale per output channel, recommended); >0 = flattened groups." />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.qgalore_min_weight_numel" value={t.qgalore_min_weight_numel ?? 16384} oninput={(e) => update('qgalore_min_weight_numel', Number(e.target.value))} min={0} />
					<FormField type="number" fieldPath="full_finetune.qgalore_max_modules" value={t.qgalore_max_modules ?? ''} oninput={(e) => update('qgalore_max_modules', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.qgalore_svd_oversampling" value={t.qgalore_svd_oversampling ?? 32} oninput={(e) => update('qgalore_svd_oversampling', Number(e.target.value))} min={0} />
					<FormField type="number" fieldPath="full_finetune.qgalore_svd_niter" value={t.qgalore_svd_niter ?? 1} oninput={(e) => update('qgalore_svd_niter', Number(e.target.value))} min={0} />
					<FormField type="number" fieldPath="full_finetune.qgalore_cos_threshold" value={t.qgalore_cos_threshold ?? 0.4} oninput={(e) => update('qgalore_cos_threshold', Number(e.target.value))} min={0} max={1} step="0.01" />
					<FormField type="number" fieldPath="full_finetune.qgalore_gamma_proj" value={t.qgalore_gamma_proj ?? 2.0} oninput={(e) => update('qgalore_gamma_proj', Number(e.target.value))} min={0} step="0.1" />
					<FormField type="number" fieldPath="full_finetune.qgalore_queue_size" value={t.qgalore_queue_size ?? 5} oninput={(e) => update('qgalore_queue_size', Number(e.target.value))} min={1} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="FP8 GEMM (full-FT, sm_89+)">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.fp8_gemm" checked={t.fp8_gemm ?? false} onchange={(e) => update('fp8_gemm', e.target.checked)} tooltip="Replace attention/FFN Linear layers with FP8 GEMMs. Mutually exclusive with LoRA / qgalore_full_ft / fp8_scaled." />
					<FormToggle fieldPath="full_finetune.fp8_gemm_compile" checked={t.fp8_gemm_compile ?? true} onchange={(e) => update('fp8_gemm_compile', e.target.checked)} tooltip="Region-compile the FP8 GEMM to fuse per-tensor scaling (~halves step time)." />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField fieldPath="full_finetune.fp8_gemm_targets" value={t.fp8_gemm_targets || 'video'} oninput={(e) => update('fp8_gemm_targets', e.target.value)} placeholder="video" tooltip="Which LTX-2 Linear layers to run in FP8 (video/audio/attn/ff/blocks/all)." />
					<FormSelect fieldPath="full_finetune.fp8_gemm_grad_dtype" value={t.fp8_gemm_grad_dtype || 'e4m3'} options={['e4m3', 'e5m2']} onchange={(e) => update('fp8_gemm_grad_dtype', e.target.value)} tooltip="FP8 format for gradients. e4m3 = more accurate; e5m2 = wider range." />
					<FormField type="number" fieldPath="full_finetune.fp8_gemm_min_numel" value={t.fp8_gemm_min_numel ?? 16384} oninput={(e) => update('fp8_gemm_min_numel', Number(e.target.value))} min={0} tooltip="Skip Linear layers with fewer than this many weight elements." />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Int8 weight-only (full-FT, Ampere+)">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.int8_weights" checked={t.int8_weights ?? false} onchange={(e) => update('int8_weights', e.target.checked)} tooltip="Store trainable Linear weights as int8 with stochastic-rounding updates. Requires fused_backward_pass and a factored optimizer. Mutually exclusive with fp8_gemm / qgalore_full_ft / fp8_scaled." />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField fieldPath="full_finetune.int8_weights_targets" value={t.int8_weights_targets || 'video'} oninput={(e) => update('int8_weights_targets', e.target.value)} placeholder="video" tooltip="Which LTX-2 Linear layers to store in int8 (video/audio/attn/ff/blocks/all)." />
					<FormField type="number" fieldPath="full_finetune.int8_weights_group_size" value={t.int8_weights_group_size ?? 0} oninput={(e) => update('int8_weights_group_size', Number(e.target.value))} min={0} tooltip="0 = row-wise (one scale per output channel); >0 = group-wise along input dim (128/256 = finer scales, less drift)." />
					<FormField type="number" fieldPath="full_finetune.int8_weights_min_numel" value={t.int8_weights_min_numel ?? 16384} oninput={(e) => update('int8_weights_min_numel', Number(e.target.value))} min={0} tooltip="Skip Linear layers with fewer than this many weight elements." />
					<FormField type="number" fieldPath="full_finetune.int8_weights_outlier_quantile" value={t.int8_weights_outlier_quantile ?? 1.0} oninput={(e) => update('int8_weights_outlier_quantile', Number(e.target.value))} min={0} max={1} step="0.001" tooltip="Set the int8 scale from a per-row/group quantile of |w| (default 1.0 = absmax). 0.999 clips the top 0.1% to give the bulk a tighter grid." />
					<FormField type="number" fieldPath="full_finetune.int8_weights_sparse_ratio" value={t.int8_weights_sparse_ratio ?? 0.0} oninput={(e) => update('int8_weights_sparse_ratio', Number(e.target.value))} min={0} max={1} step="0.001" tooltip="Keep the top fraction of |w| as an exact fp32 side-vector, excluded from the int8 grid so outliers don't coarsen the bulk weights. Default 0.0 = off; try 0.01. Alternative to outlier quantile." />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="APOLLO">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.apollo_rank" value={t.apollo_rank ?? 256} oninput={(e) => update('apollo_rank', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.apollo_update_proj_gap" value={t.apollo_update_proj_gap ?? 200} oninput={(e) => update('apollo_update_proj_gap', Number(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.apollo_scale" value={t.apollo_scale ?? 1.0} oninput={(e) => update('apollo_scale', Number(e.target.value))} step="0.01" min={0} />
					<FormSelect fieldPath="full_finetune.apollo_scale_type" value={t.apollo_scale_type || 'channel'} options={['channel', 'tensor']} onchange={(e) => update('apollo_scale_type', e.target.value)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormSelect fieldPath="full_finetune.apollo_proj" value={t.apollo_proj || 'random'} options={['random', 'svd']} onchange={(e) => update('apollo_proj', e.target.value)} />
					<FormSelect fieldPath="full_finetune.apollo_proj_type" value={t.apollo_proj_type || 'std'} options={['std', 'reverse_std', 'left', 'right']} onchange={(e) => update('apollo_proj_type', e.target.value)} />
					<FormSelect label="QAPOLLO State Bits" fieldPath="full_finetune.qapollo_optim_bits" value={String(t.qapollo_optim_bits ?? 8)} options={['8', '32']} onchange={(e) => update('qapollo_optim_bits', Number(e.target.value))} tooltip="bitsandbytes optimizer-state width for QAPOLLOAdamW. 8 is the low-VRAM default; 32 is a diagnostic/stability fallback." />
				</div>
			</div>
		</FormGroup>

		</div>

		<div class="space-y-3">
		<FormGroup title="Scope">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.freeze_early_blocks" value={t.freeze_early_blocks ?? 0} oninput={(e) => update('freeze_early_blocks', Number(e.target.value))} min={0} max={48} />
					<FormField fieldPath="full_finetune.freeze_block_indices" value={t.freeze_block_indices || ''} oninput={(e) => update('freeze_block_indices', e.target.value)} placeholder="0-7,12" />
					<FormField type="number" fieldPath="full_finetune.non_block_lr_scale" value={t.non_block_lr_scale ?? 1.0} oninput={(e) => update('non_block_lr_scale', Number(e.target.value))} step="0.1" min={0} />
					<FormField type="number" fieldPath="full_finetune.attn_geometry_lr_scale" value={t.attn_geometry_lr_scale ?? 1.0} oninput={(e) => update('attn_geometry_lr_scale', Number(e.target.value))} step="0.1" min={0} />
				</div>
				<FormField fieldPath="full_finetune.block_lr_scales" value={t.block_lr_scales || ''} oninput={(e) => update('block_lr_scales', e.target.value)} placeholder="0-11:0.1 12-23:0.4 24-:1.0" />
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.freeze_attn_geometry" checked={t.freeze_attn_geometry ?? false} onchange={(e) => update('freeze_attn_geometry', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.freeze_audio_params" checked={t.freeze_audio_params ?? false} onchange={(e) => update('freeze_audio_params', e.target.checked)} />
					<FormField type="number" fieldPath="full_finetune.audio_param_lr_scale" value={t.audio_param_lr_scale ?? 1.0} oninput={(e) => update('audio_param_lr_scale', Number(e.target.value))} step="0.1" min={0} />
					<FormToggle fieldPath="full_finetune.preserve_audio_timing" checked={t.preserve_audio_timing ?? false} onchange={(e) => update('preserve_audio_timing', e.target.checked)} tooltip="Preserve original audio duration by skipping audio time-stretching and duration alignment." />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Sampling">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.sample_every_n_steps" value={t.sample_every_n_steps ?? ''} oninput={(e) => update('sample_every_n_steps', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.sample_every_n_epochs" value={t.sample_every_n_epochs ?? ''} oninput={(e) => update('sample_every_n_epochs', numberOrNull(e.target.value))} min={1} />
					<FormSelect fieldPath="full_finetune.sample_sampling_preset" value={t.sample_sampling_preset || 'defaults'} options={['legacy', 'defaults', 'ltx20', 'ltx23', 'ltx23_hq', 'distilled_two_stage']} onchange={(e) => update('sample_sampling_preset', e.target.value)} />
					<FormSelect fieldPath="full_finetune.sample_sigma_schedule" value={t.sample_sigma_schedule || 'auto'} options={['auto', 'ltx', 'ltx23_distilled']} onchange={(e) => update('sample_sigma_schedule', e.target.value)} />
				</div>
				<PathInput fieldPath="full_finetune.sample_prompts" value={t.sample_prompts || ''} oninput={(e) => update('sample_prompts', e.target.value)} showFiles={true} />
				<FormField fieldPath="full_finetune.sample_prompts_text" value={t.sample_prompts_text || ''} oninput={(e) => update('sample_prompts_text', e.target.value)} placeholder="prompt --n negative prompt --w 768 --h 512 --f 33" />
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.height" value={t.height ?? ''} oninput={(e) => update('height', numberOrNull(e.target.value))} min={64} step="8" />
					<FormField type="number" fieldPath="full_finetune.width" value={t.width ?? ''} oninput={(e) => update('width', numberOrNull(e.target.value))} min={64} step="8" />
					<FormField type="number" fieldPath="full_finetune.sample_num_frames" value={t.sample_num_frames ?? ''} oninput={(e) => update('sample_num_frames', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.video_cfg_scale" value={t.video_cfg_scale ?? ''} oninput={(e) => update('video_cfg_scale', numberOrNull(e.target.value))} step="0.1" />
				</div>
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.sample_at_first" checked={t.sample_at_first ?? false} onchange={(e) => update('sample_at_first', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.sample_with_offloading" checked={t.sample_with_offloading ?? false} onchange={(e) => update('sample_with_offloading', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.sample_tiled_vae" checked={t.sample_tiled_vae ?? false} onchange={(e) => update('sample_tiled_vae', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.sample_include_reference" checked={t.sample_include_reference ?? false} onchange={(e) => update('sample_include_reference', e.target.checked)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Saving">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-2">
					<PathInput fieldPath="full_finetune.output_dir" value={t.output_dir || ''} oninput={(e) => update('output_dir', e.target.value)} />
					<FormField fieldPath="full_finetune.output_name" value={t.output_name || 'ltx2_full_ft'} oninput={(e) => update('output_name', e.target.value)} />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.save_every_n_steps" value={t.save_every_n_steps ?? ''} oninput={(e) => update('save_every_n_steps', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.save_last_n_steps" value={t.save_last_n_steps ?? ''} oninput={(e) => update('save_last_n_steps', numberOrNull(e.target.value))} min={0} />
					<FormField type="number" fieldPath="full_finetune.save_every_n_epochs" value={t.save_every_n_epochs ?? ''} oninput={(e) => update('save_every_n_epochs', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.save_last_n_epochs" value={t.save_last_n_epochs ?? ''} oninput={(e) => update('save_last_n_epochs', numberOrNull(e.target.value))} min={0} />
				</div>
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.save_state" checked={t.save_state ?? false} onchange={(e) => update('save_state', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.save_state_on_train_end" checked={t.save_state_on_train_end ?? false} onchange={(e) => update('save_state_on_train_end', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.save_merged_checkpoint" checked={t.save_merged_checkpoint ?? false} onchange={(e) => update('save_merged_checkpoint', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.no_final_save" checked={t.no_final_save ?? false} onchange={(e) => update('no_final_save', e.target.checked)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Logging">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-3 gap-2">
					<FormSelect fieldPath="full_finetune.log_with" value={t.log_with || ''} options={[{ value: '', label: 'None' }, 'tensorboard', 'wandb', 'all']} onchange={(e) => update('log_with', e.target.value || null)} />
					<PathInput fieldPath="full_finetune.logging_dir" value={t.logging_dir || ''} oninput={(e) => update('logging_dir', e.target.value)} invalid={fieldInvalid('full_finetune.logging_dir')} error={fieldError('full_finetune.logging_dir')} />
					<FormField type="number" fieldPath="full_finetune.log_cuda_memory_every_n_steps" value={t.log_cuda_memory_every_n_steps ?? ''} oninput={(e) => update('log_cuda_memory_every_n_steps', numberOrNull(e.target.value))} min={1} />
				</div>
				<div class="grid grid-cols-3 gap-2">
					<FormField fieldPath="full_finetune.log_tracker_name" value={t.log_tracker_name || ''} oninput={(e) => update('log_tracker_name', e.target.value)} />
					<FormField fieldPath="full_finetune.wandb_run_name" value={t.wandb_run_name || ''} oninput={(e) => update('wandb_run_name', e.target.value)} />
					<FormField fieldPath="full_finetune.training_comment" value={t.training_comment || ''} oninput={(e) => update('training_comment', e.target.value)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Resume">
			<div class="space-y-2 pt-2">
				<PathInput fieldPath="full_finetune.resume" value={t.resume || ''} oninput={(e) => update('resume', e.target.value)} showFiles={true} />
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.autoresume" checked={t.autoresume ?? false} onchange={(e) => update('autoresume', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.reset_optimizer" checked={t.reset_optimizer ?? false} onchange={(e) => update('reset_optimizer', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.reset_optimizer_params" checked={t.reset_optimizer_params ?? false} onchange={(e) => update('reset_optimizer_params', e.target.checked)} />
					<FormToggle fieldPath="full_finetune.reset_dataloader" checked={t.reset_dataloader ?? false} onchange={(e) => update('reset_dataloader', e.target.checked)} />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="Validation">
			<div class="space-y-2 pt-2">
				<PathInput fieldPath="full_finetune.validation_dataset_config" value={t.validation_dataset_config || ''} oninput={(e) => update('validation_dataset_config', e.target.value)} showFiles={true} />
				<FormField fieldPath="full_finetune.validation_extra_configs" value={t.validation_extra_configs || ''} oninput={(e) => update('validation_extra_configs', e.target.value)} placeholder="motion:path.toml night:path.toml" />
				<div class="grid grid-cols-2 gap-2">
					<FormField type="number" fieldPath="full_finetune.validate_every_n_steps" value={t.validate_every_n_steps ?? ''} oninput={(e) => update('validate_every_n_steps', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.validate_every_n_epochs" value={t.validate_every_n_epochs ?? ''} oninput={(e) => update('validate_every_n_epochs', numberOrNull(e.target.value))} min={1} />
					<FormField type="number" fieldPath="full_finetune.num_validation_batches" value={t.num_validation_batches ?? ''} oninput={(e) => update('num_validation_batches', numberOrNull(e.target.value))} min={1} />
					<FormField fieldPath="full_finetune.validation_timesteps" value={t.validation_timesteps || ''} oninput={(e) => update('validation_timesteps', e.target.value)} placeholder="100,300,500,700,900" />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="TREAD">
			<div class="space-y-2 pt-2">
				<div class="grid grid-cols-2 gap-x-4 gap-y-1">
					<FormToggle fieldPath="full_finetune.tread" checked={t.tread ?? false} onchange={(e) => update('tread', e.target.checked)} />
					<FormSelect fieldPath="full_finetune.tread_target" value={t.tread_target || 'video'} options={['video', 'audio', 'both']} onchange={(e) => update('tread_target', e.target.value)} />
					<FormField type="number" fieldPath="full_finetune.tread_selection_ratio" value={t.tread_selection_ratio ?? 0.5} oninput={(e) => update('tread_selection_ratio', Number(e.target.value))} min={0} max={1} step="0.01" />
					<FormField fieldPath="full_finetune.tread_args" value={t.tread_args || ''} oninput={(e) => update('tread_args', e.target.value)} placeholder="route=..." />
				</div>
			</div>
		</FormGroup>

		<FormGroup title="CLI">
			<div class="space-y-2 pt-2">
				<FormField fieldPath="full_finetune.accelerate_extra_args" value={t.accelerate_extra_args || ''} oninput={(e) => update('accelerate_extra_args', e.target.value)} placeholder="--num_processes 1 --gpu_ids 0" />
				<FormField fieldPath="full_finetune.extra_args" value={t.extra_args || ''} oninput={(e) => update('extra_args', e.target.value)} placeholder="--flag value --other_flag" />
			</div>
		</FormGroup>
		</div>
	</div>

	<ProcessControls processType={processType} {status} onStart={startFullFinetune} onStop={() => stopProcess(processType)} />
	<CommandPanel processType={processType} defaultFilename="fine_tuning.bat" />
</div>
{/if}
