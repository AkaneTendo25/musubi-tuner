<script>
	import { onMount } from 'svelte';
	import { projectConfig, saveProjectNow } from '$lib/stores/project.js';
	import {
		defaultModelDir,
		effectiveGemmaRoot,
		effectiveLtx2Checkpoint
	} from '$lib/utils/modelPaths.js';

	let { section = 'caching', title = 'Model Downloads', description = '' } = $props();

	let cwd = $state('');
	let downloading = $state('');
	let status = $state('');

	onMount(async () => {
		try {
			const res = await fetch('/api/fs/cwd');
			if (res.ok) cwd = (await res.json()).cwd || '';
		} catch {}
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

	async function handleDownload(preset) {
		const dest = modelDir.trim();
		if (!dest) return;
		downloading = preset;
		status = 'Downloading...';

		projectConfig.update((config) => {
			if (!config) return config;
			return { ...config, model_dir: dest };
		});
		await saveProjectNow();

		try {
			const res = await fetch('/api/fs/download-model', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ preset, dest_dir: dest })
			});
			const data = await res.json();
			if (!res.ok) {
				throw new Error(data.detail || 'Download failed');
			}

			projectConfig.update((config) => {
				if (!config) return config;
				const next = { ...config, model_dir: dest };
				const currentSection = { ...(config[section] || {}) };
				if (preset === 'ltxav') {
					next.default_ltx2_checkpoint = data.path || resolvedLtx;
					currentSection.ltx2_checkpoint = data.path || resolvedLtx;
				} else if (preset === 'gemma-unsloth') {
					next.default_gemma_root = data.path || resolvedGemma;
					next.default_gemma_safetensors = '';
					currentSection.gemma_root = data.path || resolvedGemma;
					currentSection.gemma_safetensors = '';
				}
				next[section] = currentSection;
				return next;
			});
			await saveProjectNow();
			status = `Done. Saved to ${data.path || dest}`;
		} catch (e) {
			status = e.message || 'Download failed';
		}

		downloading = '';
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
					disabled={!!downloading}
					class="px-3 py-1 text-[11px] font-medium disabled:opacity-40"
					style="background: var(--accent-muted); color: var(--accent); border: 1px solid var(--accent); border-radius: var(--radius-sm);"
				>
					{downloading === 'ltxav' ? 'Downloading...' : 'Download'}
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
					disabled={!!downloading}
					class="px-3 py-1 text-[11px] font-medium disabled:opacity-40"
					style="background: var(--accent-muted); color: var(--accent); border: 1px solid var(--accent); border-radius: var(--radius-sm);"
				>
					{downloading === 'gemma-unsloth' ? 'Downloading...' : 'Download'}
				</button>
			</div>
			<div class="text-[10px] font-mono break-all" style="color: var(--text-muted);">{resolvedGemma}</div>
		</div>
	</div>

	{#if status}
		<div
			class="text-[11px] px-3 py-2"
			style="color: {status.startsWith('Done') ? 'var(--success)' : status === 'Downloading...' ? 'var(--accent)' : 'var(--danger)'}; background: {status.startsWith('Done') ? 'var(--success-muted, rgba(34,197,94,0.1))' : status === 'Downloading...' ? 'var(--accent-muted)' : 'var(--danger-muted)'}; border-radius: var(--radius-sm);"
		>
			{status}
		</div>
	{/if}
</div>
