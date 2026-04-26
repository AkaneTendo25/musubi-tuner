<script>
	import { onDestroy, onMount } from 'svelte';
	import { projectConfig, saveProjectNow } from '$lib/stores/project.js';
	import {
		defaultModelDir,
		effectiveGemmaRoot,
		effectiveLtx2Checkpoint
	} from '$lib/utils/modelPaths.js';
	import {
		cancelModelDownload,
		formatModelDownloadStatus,
		getModelDownloadStatus,
		getModelDownloadTone,
		isActiveModelDownload,
		startModelDownload
	} from '$lib/utils/modelDownloads.js';

	let { section = 'caching', title = 'Model Downloads', description = '' } = $props();

	let cwd = $state('');
	let downloading = $state('');
	let status = $state('');
	let statusTone = $state('muted');
	let downloadJobId = $state('');
	let downloadState = $state('');
	let downloadPollTimer = null;

	onMount(async () => {
		try {
			const res = await fetch('/api/fs/cwd');
			if (res.ok) cwd = (await res.json()).cwd || '';
		} catch {}
	});

	onDestroy(() => {
		if (downloadPollTimer) clearTimeout(downloadPollTimer);
	});

	let sectionConfig = $derived($projectConfig?.[section] || {});
	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let resolvedLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, sectionConfig.ltx2_checkpoint || ''));
	let resolvedGemma = $derived(
		effectiveGemmaRoot(
			cwd,
			$projectConfig,
			sectionConfig.gemma_root || '',
			sectionConfig.gemma_safetensors || ''
		)
	);
	let hasActiveDownload = $derived(Boolean(downloadJobId) && ['queued', 'running', 'cancelling'].includes(downloadState));

	function setStatus(downloadStatus) {
		status = formatModelDownloadStatus(downloadStatus);
		statusTone = getModelDownloadTone(downloadStatus);
		downloadState = downloadStatus?.state || '';
	}

	async function finalizeDownload(downloadStatus) {
		if (downloadStatus.state === 'completed' && downloadStatus.path) {
			projectConfig.update((config) => {
				if (!config) return config;
				const next = { ...config, model_dir: modelDir };
				const currentSection = { ...(config[section] || {}) };
				if (downloading === 'ltxav') {
					next.default_ltx2_checkpoint = downloadStatus.path;
					currentSection.ltx2_checkpoint = downloadStatus.path;
				} else if (downloading === 'gemma-unsloth') {
					next.default_gemma_root = downloadStatus.path;
					next.default_gemma_safetensors = '';
					currentSection.gemma_root = downloadStatus.path;
					currentSection.gemma_safetensors = '';
				}
				next[section] = currentSection;
				return next;
			});
			await saveProjectNow();
		}

		downloadJobId = '';
		downloading = '';
	}

	async function pollDownload(jobId) {
		if (downloadPollTimer) clearTimeout(downloadPollTimer);
		try {
			const downloadStatus = await getModelDownloadStatus(jobId);
			setStatus(downloadStatus);
			if (!isActiveModelDownload(downloadStatus)) {
				await finalizeDownload(downloadStatus);
				return;
			}
			downloadPollTimer = setTimeout(() => pollDownload(jobId), 1000);
		} catch (e) {
			setStatus({ state: 'failed', error: e.message || 'Download status failed' });
			downloadJobId = '';
			downloading = '';
		}
	}

	async function handleDownload(preset) {
		if (downloadJobId) return;
		const targetPath = preset === 'ltxav' ? resolvedLtx : resolvedGemma;
		if (!targetPath) return;

		projectConfig.update((config) => {
			if (!config) return config;
			return { ...config, model_dir: modelDir.trim() };
		});
		await saveProjectNow();

		try {
			const job = await startModelDownload(preset, targetPath);
			downloading = preset;
			downloadJobId = job.job_id || '';
			setStatus(job);
			if (downloadJobId) {
				await pollDownload(downloadJobId);
			}
		} catch (e) {
			setStatus({ state: 'failed', error: e.message || 'Download failed' });
		}
	}

	async function handleCancel() {
		if (!downloadJobId) return;
		try {
			const downloadStatus = await cancelModelDownload(downloadJobId);
			setStatus(downloadStatus);
		} catch (e) {
			setStatus({ state: 'failed', error: e.message || 'Cancel failed' });
		}
	}
</script>

<div class="p-4 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
	<div class="flex items-start justify-between gap-3">
		<div>
			<div class="text-[13px] font-semibold" style="color: var(--text-primary);">{title}</div>
			{#if description}
				<div class="text-[11px]" style="color: var(--text-muted);">{description}</div>
			{/if}
			<div class="text-[10px] font-mono mt-1" style="color: var(--text-muted);">Target directory: {modelDir}</div>
		</div>
	</div>

	<div class="grid grid-cols-1 xl:grid-cols-2 gap-3">
		<div class="p-3 space-y-2" style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
			<div class="flex items-center justify-between gap-2">
				<div>
					<div class="text-[12px] font-semibold" style="color: var(--text-primary);">LTX-2 Checkpoint</div>
					<div class="text-[10px]" style="color: var(--text-muted);">Downloads `Lightricks/LTX-2` checkpoint</div>
				</div>
				<button
					onclick={() => handleDownload('ltxav')}
					disabled={hasActiveDownload}
					class="px-3 py-1 text-[11px] font-medium disabled:opacity-40"
					style="background: var(--accent-muted); color: var(--accent); border: 1px solid var(--accent); border-radius: var(--radius-sm);"
				>
					{downloading === 'ltxav' && hasActiveDownload ? 'Downloading...' : 'Download'}
				</button>
			</div>
			<div class="text-[10px] font-mono break-all" style="color: var(--text-muted);">{resolvedLtx}</div>
		</div>

		<div class="p-3 space-y-2" style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
			<div class="flex items-center justify-between gap-2">
				<div>
					<div class="text-[12px] font-semibold" style="color: var(--text-primary);">Gemma Text Encoder</div>
					<div class="text-[10px]" style="color: var(--text-muted);">Downloads `unsloth/gemma-3-12b-it`</div>
				</div>
				<button
					onclick={() => handleDownload('gemma-unsloth')}
					disabled={hasActiveDownload}
					class="px-3 py-1 text-[11px] font-medium disabled:opacity-40"
					style="background: var(--accent-muted); color: var(--accent); border: 1px solid var(--accent); border-radius: var(--radius-sm);"
				>
					{downloading === 'gemma-unsloth' && hasActiveDownload ? 'Downloading...' : 'Download'}
				</button>
			</div>
			<div class="text-[10px] font-mono break-all" style="color: var(--text-muted);">{resolvedGemma}</div>
		</div>
	</div>

	{#if status}
		<div
			class="flex items-center justify-between gap-3 text-[11px] px-3 py-2"
			style="color: {statusTone === 'success' ? 'var(--success)' : statusTone === 'accent' ? 'var(--accent)' : statusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {statusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : statusTone === 'accent' ? 'var(--accent-muted)' : statusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);"
		>
			<span>{status}</span>
			{#if hasActiveDownload}
				<button
					type="button"
					onclick={handleCancel}
					class="px-2 py-1 text-[10px] font-medium"
					style="background: var(--bg-surface); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
				>
					Cancel
				</button>
			{/if}
		</div>
	{/if}
</div>
