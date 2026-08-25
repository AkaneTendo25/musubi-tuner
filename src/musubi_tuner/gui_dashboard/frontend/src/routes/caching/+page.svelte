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
	let cacheStatus = $state(null);
	let cacheStatusLoading = $state(false);
	let cacheStatusError = $state('');

	onMount(() => {
		fetch('/api/fs/cwd').then((res) => res.ok ? res.json() : null).then((data) => { cwd = data?.cwd || ''; }).catch(() => {});
		getModelDownloadPresets().then((presets) => { downloadPresets = presets; }).catch(() => {});
		resumeModelDownloadPolling();
		refreshCacheStatus().catch(() => {});
		preloadLogsIfActive(['cache_latents', 'cache_text', 'cache_dino']);
		const logInterval = startLogPolling(['cache_latents', 'cache_text', 'cache_dino'], 1000);
		return () => {
			clearInterval(logInterval);
			if (ltxScanJobId) cancelCheckpointScan(ltxScanJobId).catch(() => {});
			if (gemmaScanJobId) cancelCheckpointScan(gemmaScanJobId).catch(() => {});
			if (gemmaSafetensorsScanJobId) cancelCheckpointScan(gemmaSafetensorsScanJobId).catch(() => {});
		};
	});

	function updateCaching(key, value) { updateSection('caching', key, value); }
	function updateH3ReferenceShortEdge(value) {
		updateSection('caching', 'h3_reference_image_short_edge', value);
		updateSection('training', 'reference_image_short_edge', value);
		updateSection('inference', 'h3_reference_image_short_edge', value);
	}
	function updateH3ReferenceVideoSizing(key, value) {
		updateSection('caching', `h3_reference_video_${key}`, value);
		updateSection('training', `reference_video_${key}`, value);
		updateSection('inference', `h3_reference_video_${key}`, value);
	}

	let caching = $derived($projectConfig?.caching || {});
	let latentStatus = $derived($processStatuses.cache_latents || { state: 'idle', exit_code: null });
	let textStatus = $derived($processStatuses.cache_text || { state: 'idle', exit_code: null });
	let dinoStatus = $derived($processStatuses.cache_dino || { state: 'idle', exit_code: null });
	let latentLogs = $derived($processLogs.cache_latents || []);
	let textLogs = $derived($processLogs.cache_text || []);
	let dinoLogs = $derived($processLogs.cache_dino || []);
	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let resolvedLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, caching.ltx2_checkpoint || ''));
	let activeGemmaSafetensors = $derived(effectiveGemmaSafetensors($projectConfig, caching.gemma_safetensors || '', caching.gemma_root || ''));
	let gemmaRootDisabled = $derived(Boolean(caching.gemma_safetensors));
	let resolvedGemma = $derived(effectiveGemmaRoot(cwd, $projectConfig, caching.gemma_root || '', caching.gemma_safetensors || ''));
	let scanTargetGemmaRoot = $derived(effectiveGemmaRoot(cwd, $projectConfig, caching.gemma_root || '', ''));
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
		await startSharedModelDownload({ preset, targetPath, modelDir, section: 'caching' });
	}

	async function stopDownload() {
		await cancelSharedModelDownload();
	}

	async function refreshCacheStatus() {
		if (cacheStatusLoading) return;
		cacheStatusLoading = true;
		cacheStatusError = '';
		try {
			const res = await fetch('/api/cache/status', { cache: 'no-store' });
			const data = await res.json();
			if (!res.ok) throw new Error(data?.detail || 'Cache scan failed');
			cacheStatus = data;
		} catch (e) {
			cacheStatusError = e?.message || 'Cache scan failed';
			cacheStatus = null;
		} finally {
			cacheStatusLoading = false;
		}
	}

	function cacheReady(row) {
		return row.source_count > 0 && row.missing_latent === 0 && row.missing_text === 0 && row.missing_audio === 0;
	}

	function cacheIssueCount(row) {
		return (row.missing_latent || 0) + (row.missing_text || 0) + (row.missing_audio || 0) + (row.stale_latent || 0) + (row.stale_text || 0) + (row.stale_audio || 0);
	}

	function cacheTone(row) {
		if (row.warnings?.length) return 'warn';
		if (cacheIssueCount(row) > 0) return 'warn';
		if (cacheReady(row)) return 'ready';
		return 'muted';
	}

	function bucketSummary(row) {
		if (!row.buckets?.length) return '—';
		return row.buckets.slice(0, 3).map((b) => `${b.bucket}:${b.count}`).join('  ');
	}
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-5">
		<!-- Cache Status -->
		<div class="p-4 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="flex items-center justify-between gap-3">
				<div>
					<div class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Cache Status</div>
					{#if cacheStatus?.generated_at}
						<div class="text-[11px]" style="color: var(--text-muted);">Updated {cacheStatus.generated_at}</div>
					{/if}
				</div>
				<button
					type="button"
					onclick={refreshCacheStatus}
					disabled={cacheStatusLoading}
					class="px-2.5 py-1 text-[11px] font-medium disabled:opacity-50"
					style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
				>{cacheStatusLoading ? 'Scanning...' : 'Refresh'}</button>
			</div>

			{#if cacheStatusError}
				<div class="text-[12px] px-3 py-2" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">{cacheStatusError}</div>
			{:else if cacheStatus?.rows?.length}
				<div class="grid grid-cols-3 xl:grid-cols-6 gap-2">
					<div class="px-2 py-1.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
						<div class="text-[10px] uppercase tracking-wider" style="color: var(--text-muted);">Sources</div>
						<div class="text-sm font-semibold" style="color: var(--text-primary);">{cacheStatus.totals.source_count}</div>
					</div>
					<div class="px-2 py-1.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
						<div class="text-[10px] uppercase tracking-wider" style="color: var(--text-muted);">Latents</div>
						<div class="text-sm font-semibold" style="color: var(--text-primary);">{cacheStatus.totals.latent_count}</div>
					</div>
					<div class="px-2 py-1.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
						<div class="text-[10px] uppercase tracking-wider" style="color: var(--text-muted);">Text</div>
						<div class="text-sm font-semibold" style="color: var(--text-primary);">{cacheStatus.totals.text_count}</div>
					</div>
					<div class="px-2 py-1.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
						<div class="text-[10px] uppercase tracking-wider" style="color: var(--text-muted);">Audio</div>
						<div class="text-sm font-semibold" style="color: var(--text-primary);">{cacheStatus.totals.audio_count}</div>
					</div>
					<div class="px-2 py-1.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
						<div class="text-[10px] uppercase tracking-wider" style="color: var(--text-muted);">Missing</div>
						<div class="text-sm font-semibold" style="color: var(--text-primary);">{cacheStatus.totals.missing_latent + cacheStatus.totals.missing_text + cacheStatus.totals.missing_audio}</div>
					</div>
					<div class="px-2 py-1.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
						<div class="text-[10px] uppercase tracking-wider" style="color: var(--text-muted);">Stale</div>
						<div class="text-sm font-semibold" style="color: var(--text-primary);">{cacheStatus.totals.stale_latent + cacheStatus.totals.stale_text + cacheStatus.totals.stale_audio}</div>
					</div>
				</div>

				<div class="overflow-x-auto">
					<table class="w-full text-[11px]">
						<thead>
							<tr style="color: var(--text-muted); border-bottom: 1px solid var(--border-subtle);">
								<th class="text-left font-medium py-2 pr-3">Dataset</th>
								<th class="text-right font-medium py-2 px-2">Sources</th>
								<th class="text-right font-medium py-2 px-2">Latents</th>
								<th class="text-right font-medium py-2 px-2">Text</th>
								<th class="text-right font-medium py-2 px-2">Audio</th>
								<th class="text-right font-medium py-2 px-2">Missing</th>
								<th class="text-right font-medium py-2 px-2">Stale</th>
								<th class="text-left font-medium py-2 pl-3">Buckets</th>
							</tr>
						</thead>
						<tbody>
							{#each cacheStatus.rows as row}
								<tr style="border-bottom: 1px solid var(--border-subtle);">
									<td class="py-2 pr-3 min-w-[220px]">
										<div class="flex items-center gap-2">
											<span class="w-1.5 h-1.5 rounded-full flex-shrink-0" style="background: {cacheTone(row) === 'ready' ? 'var(--success)' : cacheTone(row) === 'warn' ? 'var(--warning)' : 'var(--text-muted)'};"></span>
											<div class="min-w-0">
												<div class="font-medium truncate" style="color: var(--text-primary);">{row.group} {row.index + 1} · {row.type}</div>
												<div class="truncate" style="color: var(--text-muted);" title={row.cache_directory}>{row.cache_directory || 'No cache directory'}</div>
												{#if row.warnings?.length}
													<div style="color: var(--warning);">{row.warnings[0]}</div>
												{/if}
											</div>
										</div>
									</td>
									<td class="text-right py-2 px-2 tabular-nums">{row.source_count}</td>
									<td class="text-right py-2 px-2 tabular-nums">{row.latent_count}</td>
									<td class="text-right py-2 px-2 tabular-nums">{row.text_count}</td>
									<td class="text-right py-2 px-2 tabular-nums">{row.audio_count}</td>
									<td class="text-right py-2 px-2 tabular-nums" style="color: {(row.missing_latent + row.missing_text + row.missing_audio) > 0 ? 'var(--warning)' : 'var(--text-secondary)'};">{row.missing_latent + row.missing_text + row.missing_audio}</td>
									<td class="text-right py-2 px-2 tabular-nums" style="color: {(row.stale_latent + row.stale_text + row.stale_audio) > 0 ? 'var(--warning)' : 'var(--text-secondary)'};">{row.stale_latent + row.stale_text + row.stale_audio}</td>
									<td class="py-2 pl-3 min-w-[120px]" style="color: var(--text-muted);">{bucketSummary(row)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<div class="text-[12px] px-3 py-2" style="color: var(--text-muted); background: var(--bg-elevated); border-radius: var(--radius-sm);">
					{cacheStatusLoading ? 'Scanning cache status...' : 'No datasets configured.'}
				</div>
			{/if}
		</div>

		<!-- Shared Settings -->
		<div class="p-4 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="grid grid-cols-2 xl:grid-cols-3 gap-3">
				<div class="text-[12px] font-semibold" style="color: var(--text-primary);">MiniMax H3</div>
				{#if caching.model_type === 'minimax_h3'}
					<PathInput fieldPath="caching.h3_video_vae" value={caching.h3_video_vae || ''} oninput={(e) => updateCaching('h3_video_vae', e.target.value)} showFiles tooltip="MiniMax H3 video VAE checkpoint" />
					<PathInput fieldPath="caching.h3_audio_vae" value={caching.h3_audio_vae || ''} oninput={(e) => updateCaching('h3_audio_vae', e.target.value)} showFiles tooltip="MiniMax H3 audio VAE checkpoint; optional for image-only datasets" />
				{:else}
					<div class="space-y-2">
						<CheckpointInput fieldPath="caching.ltx2_checkpoint" label="LTX-2 Checkpoint" value={caching.ltx2_checkpoint || ''} onchange={(v) => updateCaching('ltx2_checkpoint', v)} showFiles tooltip="Path to the LTX-2 model checkpoint file" actionLabel="D" actionBusyLabel="..." actionDisabled={hasActiveDownload || ltxDownloadExists} actionTooltip={modelDownloadTooltip(downloadPresets, 'ltxav', resolvedLtx, ltxDownloadExists)} onaction={() => downloadModel('ltxav')} />
						<ModelPathStatus exists={ltxDownloadExists} foundPath={foundLtxPath} disabled={hasActiveDownload} scanning={scanningLtx} scanMessage={ltxScanMessage} scanTone={ltxScanTone} onscan={scanLtx} oncancel={stopLtxScan} onusefound={(path) => updateCaching('ltx2_checkpoint', path)} />
					</div>
					<div class="space-y-2">
						<CheckpointInput fieldPath="caching.gemma_root" label="Gemma Root" value={caching.gemma_root || ''} onchange={(v) => updateCaching('gemma_root', v)} disabled={gemmaRootDisabled} tooltip={gemmaRootDisabled ? 'Ignored while Gemma Safetensors is set' : 'Root directory containing Gemma text encoder weights'} actionLabel="D" actionBusyLabel="..." actionDisabled={gemmaRootDisabled || hasActiveDownload || gemmaDownloadExists} actionTooltip={gemmaRootDisabled ? 'Gemma Safetensors is active' : modelDownloadTooltip(downloadPresets, 'gemma-unsloth', resolvedGemma, gemmaDownloadExists)} onaction={() => downloadModel('gemma-unsloth')} />
						<ModelPathStatus exists={gemmaRootDisabled || gemmaDownloadExists} foundPath={foundGemmaPath} disabled={gemmaRootDisabled || hasActiveDownload} scanning={scanningGemma} scanMessage={gemmaScanMessage} scanTone={gemmaScanTone} onscan={scanGemma} oncancel={stopGemmaScan} onusefound={(path) => { updateCaching('gemma_root', path); updateCaching('gemma_safetensors', ''); }} />
					</div>
					<FormSelect fieldPath="caching.ltx2_mode" value={caching.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => updateCaching('ltx2_mode', e.target.value)} tooltip="Video: visual only, AV: audio+video, Audio: audio only" />
				{/if}
			</div>
			{#if caching.model_type === 'minimax_h3'}
				<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
					<PathInput fieldPath="caching.h3_text_encoder" value={caching.h3_text_encoder || ''} oninput={(e) => updateCaching('h3_text_encoder', e.target.value)} showFiles tooltip="MiniMax H3 Qwen3-VL checkpoint: released BF16 or prequantized Comfy NVFP4/AWQ safetensors." />
					<PathInput fieldPath="caching.h3_tokenizer" value={caching.h3_tokenizer || ''} oninput={(e) => updateCaching('h3_tokenizer', e.target.value)} showFiles tooltip="MiniMax H3 tokenizer and processor directory" />
					<FormSelect fieldPath="caching.h3_task" value={caching.h3_task || 't2va'} options={['t2va', 'i2va', 'fl2va', 'l2va', 'ref2va', 'ref2va_omni']} onchange={(e) => updateCaching('h3_task', e.target.value)} tooltip="FL2VA training accepts T2VA, I2VA, FL2VA, or L2VA caches. Ref2VA and Ref2VA Omni training require their matching cache task." />
					<FormSelect fieldPath="caching.h3_text_encoder_dtype" value={caching.h3_text_encoder_dtype || 'bfloat16'} options={[{ value: 'bfloat16', label: 'BF16' }]} onchange={(e) => updateCaching('h3_text_encoder_dtype', e.target.value)} tooltip="MiniMax H3 Qwen3-VL conditioning and cached layer-50 outputs require BF16." />
					<FormField type="number" fieldPath="caching.cache_batch_size" value={caching.cache_batch_size ?? ''} oninput={(e) => updateCaching('cache_batch_size', e.target.value ? Number(e.target.value) : null)} min={1} placeholder="Automatic" tooltip="Batch size passed to both H3 cache stages" />
					<FormField type="number" fieldPath="caching.h3_reference_image_short_edge" value={caching.h3_reference_image_short_edge ?? 2048} oninput={(e) => updateH3ReferenceShortEdge(Number(e.target.value || 2048))} min={32} step={8} tooltip="Shared reference-image size for H3 caching, training, and inference. Changing it here updates all three stages." />
					<FormField type="number" fieldPath="caching.h3_reference_video_short_edge" value={caching.h3_reference_video_short_edge ?? 768} oninput={(e) => updateH3ReferenceVideoSizing('short_edge', Number(e.target.value || 768))} min={16} step={16} tooltip="Shared reference-video short edge. Lower values reduce Ref2VA compute and VRAM." />
					<FormField type="number" fieldPath="caching.h3_reference_video_max_pixels" value={caching.h3_reference_video_max_pixels ?? 1032192} oninput={(e) => updateH3ReferenceVideoSizing('max_pixels', Number(e.target.value || 1032192))} min={256} step={256} tooltip="Shared maximum pixels per reference-video frame after aspect-preserving resize." />
					<FormToggle fieldPath="caching.h3_cache_guidance_empty" checked={caching.h3_cache_guidance_empty ?? false} onchange={(e) => updateCaching('h3_cache_guidance_empty', e.target.checked)} tooltip="Also cache empty-text conditioning required by guidance training and caption dropout." />
				</div>
			{:else}
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
			{/if}
			<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
				{#if caching.model_type !== 'minimax_h3'}
					<div class="space-y-2">
						<PathInput fieldPath="caching.gemma_safetensors" value={caching.gemma_safetensors || ''} oninput={(e) => updateCaching('gemma_safetensors', e.target.value)} showFiles tooltip="Single safetensors file (alternative to Gemma Root)" />
						{#if activeGemmaSafetensors}
							<ModelPathStatus exists={gemmaSafetensorsExists} foundPath={foundGemmaSafetensorsPath} disabled={hasActiveDownload} scanning={scanningGemmaSafetensors} scanMessage={gemmaSafetensorsScanMessage} scanTone={gemmaSafetensorsScanTone} onscan={scanGemmaSafetensors} oncancel={stopGemmaSafetensorsScan} onusefound={(path) => updateCaching('gemma_safetensors', path)} />
						{/if}
					</div>
					<PathInput fieldPath="caching.ltx2_text_encoder_checkpoint" value={caching.ltx2_text_encoder_checkpoint || ''} oninput={(e) => updateCaching('ltx2_text_encoder_checkpoint', e.target.value)} showFiles tooltip="Separate text encoder checkpoint (if different from main)" />
				{/if}
				<FormSelect fieldPath="caching.mixed_precision" value={caching.mixed_precision || 'no'} options={['no', 'fp16', 'bf16']} onchange={(e) => updateCaching('mixed_precision', e.target.value)} tooltip="Mixed precision mode for text encoder caching." />
				<FormField type="number" fieldPath="caching.num_workers" value={caching.num_workers ?? ''} oninput={(e) => updateCaching('num_workers', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" tooltip="Number of data loader workers" />
			</div>
			<div class="grid grid-cols-2 xl:grid-cols-5 gap-x-4 gap-y-1">
				<FormToggle fieldPath="caching.skip_existing" checked={caching.skip_existing ?? false} onchange={(e) => updateCaching('skip_existing', e.target.checked)} tooltip="Skip files that already have cached outputs" />
				<FormToggle label="Faster checking" fieldPath="caching.faster_check" checked={caching.faster_check ?? false} onchange={(e) => updateCaching('faster_check', e.target.checked)} tooltip="When Skip Existing is enabled, recognize caches by filename instead of opening each one. A sample of caches is still validated in full first, and any mismatch falls back to checking every cache." />
				<FormToggle fieldPath="caching.atomic_cache_writes" checked={caching.atomic_cache_writes ?? false} onchange={(e) => updateCaching('atomic_cache_writes', e.target.checked)} tooltip="Write cache files through a temporary sibling file, then atomically replace the final cache path after a successful save." />
				<FormToggle fieldPath="caching.cache_distributed" checked={caching.cache_distributed ?? false} onchange={(e) => updateCaching('cache_distributed', e.target.checked)} tooltip="Shard caching work across multiple processes (opt-in multi-process cache sharding)." />
				<FormToggle fieldPath="caching.cpu_staged_checkpoint_loading" checked={caching.cpu_staged_checkpoint_loading ?? false} onchange={(e) => updateCaching('cpu_staged_checkpoint_loading', e.target.checked)} tooltip="Stage checkpoint tensors through CPU before moving them to the selected device." />
			</div>
			{#if $advancedMode}
				<div class="grid grid-cols-2 xl:grid-cols-4 gap-3">
					<FormSelect fieldPath="caching.vae_dtype" value={caching.vae_dtype || ''} options={[{ value: '', label: 'bfloat16 (default)' }, 'float16', 'bfloat16', 'float32']} onchange={(e) => updateCaching('vae_dtype', e.target.value || null)} tooltip="VAE dtype for latent caching. Blank uses the default `bfloat16`." />
					<FormField fieldPath="caching.device" value={caching.device || ''} oninput={(e) => updateCaching('device', e.target.value || null)} placeholder="Auto" tooltip="Torch device. Leave blank to auto-select the runtime device." />
					<FormToggle fieldPath="caching.keep_cache" checked={caching.keep_cache ?? false} onchange={(e) => updateCaching('keep_cache', e.target.checked)} tooltip="Keep old cache files when re-caching" />
						<FormSelect fieldPath="caching.video_decode_backend" value={caching.video_decode_backend || ''} options={[{ value: '', label: 'pyav (default)' }, 'decord', 'torchcodec']} onchange={(e) => updateCaching('video_decode_backend', e.target.value || null)} tooltip="Video decode backend for latent caching. pyav keeps the default path. decord and torchcodec batch-decode selected frames and require optional dependencies. Alternate-backend errors are logged before retrying with pyav. Decoded pixels can differ across backends." />
						<FormSelect fieldPath="caching.video_decode_device" value={caching.video_decode_device || ''} options={[{ value: '', label: 'cpu (default)' }, 'cuda']} onchange={(e) => updateCaching('video_decode_device', e.target.value || null)} tooltip="Device passed to torchcodec's VideoDecoder. CUDA decode support depends on the installed torchcodec and FFmpeg build. Ignored by pyav/decord." />
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

					{#if caching.model_type !== 'minimax_h3'}
					<FormGroup title="Reference (V2V)">
						<div class="grid grid-cols-2 gap-3 pt-2">
							<FormField type="number" fieldPath="caching.reference_frames" value={caching.reference_frames ?? 1} oninput={(e) => updateCaching('reference_frames', Number(e.target.value))} min={1} tooltip="Reference frames for V2V" />
							<FormField type="number" fieldPath="caching.reference_downscale" value={caching.reference_downscale ?? 1} oninput={(e) => updateCaching('reference_downscale', Number(e.target.value))} min={1} tooltip="Reference downscale factor" />
						</div>
					</FormGroup>
					{/if}

					{#if caching.model_type !== 'minimax_h3'}
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
								<FormToggle fieldPath="caching.preserve_audio_timing" checked={caching.preserve_audio_timing ?? false} onchange={(e) => updateCaching('preserve_audio_timing', e.target.checked)} tooltip="Preserve original audio duration by skipping audio time-stretching and audio-latent duration alignment." />
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

				<ProcessControls processType="cache_latents" status={latentStatus} onStart={() => startProcess('cache_latents')} onStop={(options) => stopProcess('cache_latents', options)} />
				<ProcessConsole lines={latentLogs} processType="cache_latents" initiallyCollapsed={false} />
				{#if $advancedMode}
					<CommandPanel processType="cache_latents" defaultFilename="cache_latents.bat" />
				{/if}
			</div>

			<!-- Cache Text -->
			<div class="space-y-3">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Cache Text Encoder</span>

				{#if caching.model_type === 'minimax_h3'}
					<FormGroup title="Qwen3-VL Loading" collapsed={false}>
						<div class="space-y-2 pt-2">
							<FormSelect fieldPath="caching.h3_text_encoder_quantization" value={caching.h3_text_encoder_quantization || 'none'} options={[{ value: 'none', label: 'BF16 (~52 GB peak)' }, { value: 'int8', label: 'INT8 (~28 GB peak)' }, { value: 'nf4', label: 'NF4 (~17 GB peak)' }, { value: 'nvfp4', label: 'NVFP4 W4A16 · on the fly (~18 GB)' }, { value: 'nvfp4_awq', label: 'NVFP4/AWQ · pre-quantized (~18 GB)' }]} onchange={(e) => updateCaching('h3_text_encoder_quantization', e.target.value)} tooltip="NVFP4 quantizes an ordinary BF16 checkpoint layer by layer with BF16 activations. NVFP4/AWQ loads the released pre-quantized checkpoint. Peak estimates include encode workspace." />
							<div class="grid grid-cols-3 gap-2">
								<FormField label="Blocks to stream" type="number" fieldPath="caching.h3_text_encoder_blocks_to_stream" value={caching.h3_text_encoder_blocks_to_stream ?? 0} oninput={(e) => updateCaching('h3_text_encoder_blocks_to_stream', Number(e.target.value))} min={0} max={50} tooltip="Stream this many Qwen3-VL blocks from CPU to reduce peak VRAM. 0 keeps all blocks resident." />
								<FormToggle label="NVFP4 scaled GEMM" fieldPath="caching.h3_nvfp4_scaled_mm" checked={caching.h3_nvfp4_scaled_mm ?? false} onchange={(e) => updateCaching('h3_nvfp4_scaled_mm', e.target.checked)} disabled={caching.h3_text_encoder_quantization !== 'nvfp4_awq'} tooltip="Use scaled NVFP4 matrix multiplication for a compatible NVFP4/AWQ text encoder." />
								<FormField label="Visual max pixels" type="number" fieldPath="caching.h3_text_visual_max_pixels" value={caching.h3_text_visual_max_pixels ?? 0} oninput={(e) => updateCaching('h3_text_visual_max_pixels', Number(e.target.value))} min={0} step={1024} tooltip="Cap pixels passed through Qwen3-VL visual conditioning. 0 uses the encoder default." />
							</div>
						</div>
					</FormGroup>
				{:else}
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
				{/if}

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
									<FormSelect fieldPath="caching.dop_mode" value={caching.dop_mode || 'fixed'} onchange={(e) => updateCaching('dop_mode', e.target.value)} options={[{value:'fixed',label:'Fixed prompt'}, {value:'caption_replace',label:'Caption replacement'}]} tooltip="Must match the DOP mode used during training." />
									{#if (caching.dop_mode || 'fixed') === 'caption_replace'}
										<FormField fieldPath="caching.dop_trigger" value={caching.dop_trigger || ''} oninput={(e) => updateCaching('dop_trigger', e.target.value)} placeholder="sks" tooltip="Single trigger token replaced with the class prompt." />
										<FormField fieldPath="caching.dop_replacements" value={caching.dop_replacements || ''} oninput={(e) => updateCaching('dop_replacements', e.target.value)} placeholder="sks=>woman;sksdog=>dog" tooltip="Multi-concept mappings; must match training." />
									{/if}
									<FormField fieldPath="caching.dop_prompt_bank" value={caching.dop_prompt_bank || ''} oninput={(e) => updateCaching('dop_prompt_bank', e.target.value)} placeholder="person;woman outdoors" tooltip="Additional prompts separated by semicolons." />
									<FormField fieldPath="caching.dop_args" value={caching.dop_args || ''} oninput={(e) => updateCaching('dop_args', e.target.value)} placeholder="mode=caption_replace trigger=sks class=woman" tooltip="Additional prompt-related DOP cache values." />
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

				<ProcessControls processType="cache_text" status={textStatus} onStart={() => startProcess('cache_text')} onStop={(options) => stopProcess('cache_text', options)} />
				<ProcessConsole lines={textLogs} processType="cache_text" initiallyCollapsed={false} />
				{#if $advancedMode}
					<CommandPanel processType="cache_text" defaultFilename="cache_text.bat" />
				{/if}
			</div>
		</div>
	</div>
{/if}
