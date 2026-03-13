<script>
	import FormField from '$lib/components/FormField.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import { projectConfig, projectLoaded, updateSection, saveProjectDebounced } from '$lib/stores/project.js';
	import { onMount } from 'svelte';

	let cwd = $state('');
	let downloading = $state('');
	let downloadStatus = $state('');

	onMount(async () => {
		try {
			const res = await fetch('/api/fs/cwd');
			if (res.ok) cwd = (await res.json()).cwd || '';
		} catch {}
	});

	function updateConfig(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, [key]: value };
		});
		saveProjectDebounced();
	}

	let modelDir = $derived($projectConfig?.model_dir || (cwd ? cwd + '/ltx-2-models' : ''));

	async function downloadModel(preset) {
		const dest = $projectConfig?.model_dir || modelDir;
		if (!dest) return;
		downloading = preset;
		downloadStatus = 'Downloading...';
		try {
			const res = await fetch('/api/fs/download-model', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ preset, dest_dir: dest }),
			});
			if (res.ok) {
				const data = await res.json();
				downloadStatus = 'Done! Saved to ' + (data.path || dest);
			} else {
				const err = await res.json();
				downloadStatus = err.detail || 'Download failed';
			}
		} catch (e) {
			downloadStatus = 'Failed: ' + e.message;
		}
		downloading = '';
		setTimeout(() => downloadStatus = '', 8000);
	}
</script>

<div class="space-y-5">
	<div>
		<h2 class="text-base font-semibold" style="color: var(--text-primary);">Settings</h2>
		<p class="text-[12px]" style="color: var(--text-muted);">Model paths and download configuration.</p>
	</div>

	<!-- Model Directory -->
	<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
		<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>
		<div class="p-5 space-y-3">
			<div class="flex items-center gap-3 mb-2">
				<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
					<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z"/></svg>
				</div>
				<div>
					<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Model Directory</div>
					<div class="text-[11px]" style="color: var(--text-muted);">Where to store and find downloaded model files</div>
				</div>
			</div>
			<PathInput label="Model Directory" value={$projectConfig?.model_dir || modelDir} oninput={(e) => updateConfig('model_dir', e.target.value)} placeholder="{cwd}/ltx-2-models" tooltip="Directory for model checkpoints. Downloaded models are saved here." />
		</div>
	</div>

	<!-- Download Models -->
	<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
		<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>
		<div class="p-5 space-y-4">
			<div class="flex items-center gap-3 mb-2">
				<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
					<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"/></svg>
				</div>
				<div>
					<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Download Models</div>
					<div class="text-[11px]" style="color: var(--text-muted);">Download required models to the model directory via huggingface-cli</div>
				</div>
			</div>

			<!-- LTX-2 Checkpoint -->
			<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
				<div class="flex items-center justify-between mb-1">
					<span class="text-[12px] font-semibold" style="color: var(--text-primary);">LTX-2 Checkpoint</span>
					<button
						onclick={() => downloadModel('ltxav')}
						disabled={!!downloading}
						class="px-3 py-1 text-[11px] font-medium flex items-center gap-1.5 disabled:opacity-40"
						style="background: var(--accent-muted); color: var(--accent); border-radius: var(--radius-sm); border: 1px solid var(--accent);"
						onmouseenter={(e) => { if (!downloading) e.currentTarget.style.background = 'var(--accent)'; e.currentTarget.style.color = 'var(--bg-base)'; }}
						onmouseleave={(e) => { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; }}
					>
						{#if downloading === 'ltxav'}
							<svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
						{:else}
							<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
						{/if}
						Download
					</button>
				</div>
				<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
					LTXAV 19B (audio+video) from <span class="font-mono text-[10px]">Lightricks/LTX-2</span>
				</p>
				<div class="text-[10px] font-mono mt-1 truncate" style="color: var(--text-muted);">ltx-2-19b-dev.safetensors</div>
			</div>

			<!-- Gemma -->
			<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
				<div class="flex items-center justify-between mb-1">
					<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Gemma Text Encoder</span>
					<button
						onclick={() => downloadModel('gemma-unsloth')}
						disabled={!!downloading}
						class="px-3 py-1 text-[11px] font-medium flex items-center gap-1.5 disabled:opacity-40"
						style="background: var(--accent-muted); color: var(--accent); border-radius: var(--radius-sm); border: 1px solid var(--accent);"
						onmouseenter={(e) => { if (!downloading) e.currentTarget.style.background = 'var(--accent)'; e.currentTarget.style.color = 'var(--bg-base)'; }}
						onmouseleave={(e) => { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; }}
					>
						{#if downloading === 'gemma-unsloth'}
							<svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
						{:else}
							<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
						{/if}
						Download
					</button>
				</div>
				<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
					Gemma 3 12B-IT from <span class="font-mono text-[10px]">unsloth/gemma-3-12b-it</span>
				</p>
				<div class="text-[10px] font-mono mt-1 truncate" style="color: var(--text-muted);">Full model directory (tokenizer + weights)</div>
			</div>

			<!-- Status -->
			{#if downloadStatus}
				<div class="text-[11px] px-3 py-2" style="color: {downloadStatus.startsWith('Done') ? 'var(--success)' : downloadStatus === 'Downloading...' ? 'var(--accent)' : 'var(--danger)'}; background: {downloadStatus.startsWith('Done') ? 'var(--success-muted, rgba(34,197,94,0.1))' : downloadStatus === 'Downloading...' ? 'var(--accent-muted)' : 'var(--danger-muted)'}; border-radius: var(--radius-sm);">
					{downloadStatus}
				</div>
			{/if}
		</div>
	</div>
</div>
