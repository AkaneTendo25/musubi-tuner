<script>
	import { projectConfig, replaceProjectConfig, saveProjectNow } from '$lib/stores/project.js';

	let fileInput = $state(null);
	let status = $state('');
	let statusTone = $state('muted');
	let busy = $state('');
	let keepProjectDir = $state(true);

	let canRun = $derived(Boolean($projectConfig) && !busy);

	function setStatus(message, tone = 'muted') {
		status = message;
		statusTone = tone;
	}

	function safeFilename() {
		const name = String($projectConfig?.name || 'musubi-project')
			.trim()
			.replace(/[^a-z0-9._-]+/gi, '-')
			.replace(/^-+|-+$/g, '');
		return `${name || 'musubi-project'}.json`;
	}

	function downloadJson(filename, data) {
		const blob = new Blob([`${JSON.stringify(data, null, 2)}\n`], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = filename;
		document.body.appendChild(link);
		link.click();
		link.remove();
		URL.revokeObjectURL(url);
	}

	async function exportConfig() {
		if (!canRun) return;
		busy = 'export';
		setStatus('');
		try {
			await saveProjectNow();
			downloadJson(safeFilename(), $projectConfig);
			setStatus('Exported', 'success');
		} catch (e) {
			setStatus(e?.message || 'Export failed', 'danger');
		} finally {
			busy = '';
		}
	}

	function openImport() {
		if (!canRun || !fileInput) return;
		fileInput.value = '';
		fileInput.click();
	}

	async function importConfig(event) {
		const file = event.currentTarget.files?.[0];
		if (!file || busy) return;
		busy = 'import';
		setStatus('');
		try {
			const imported = JSON.parse(await file.text());
			if (!imported || typeof imported !== 'object' || Array.isArray(imported)) {
				throw new Error('Config JSON must be an object');
			}
			const nextConfig = { ...imported };
			if (keepProjectDir) {
				nextConfig.project_dir = $projectConfig?.project_dir || '';
			}
			await replaceProjectConfig(nextConfig);
			setStatus('Imported', 'success');
		} catch (e) {
			setStatus(e?.message || 'Import failed', 'danger');
		} finally {
			busy = '';
			event.currentTarget.value = '';
		}
	}
</script>

<div class="p-3.5 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
	<input bind:this={fileInput} type="file" accept="application/json,.json" class="hidden" onchange={importConfig} />

	<div class="flex items-center justify-between gap-3">
		<div>
			<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Project Config</div>
			<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">Settings JSON</div>
		</div>
	</div>

	<div class="grid grid-cols-2 gap-2">
		<button
			type="button"
			onclick={exportConfig}
			disabled={!canRun}
			data-tooltip="Export current project config as JSON"
			class="flex items-center justify-center gap-2 px-3 py-2 text-[12px] font-medium disabled:opacity-40"
			style="background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
		>
			<svg class="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
				<path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" />
			</svg>
			<span>{busy === 'export' ? 'Saving' : 'Export'}</span>
		</button>
		<button
			type="button"
			onclick={openImport}
			disabled={!canRun}
			data-tooltip="Import project config from JSON"
			class="flex items-center justify-center gap-2 px-3 py-2 text-[12px] font-medium disabled:opacity-40"
			style="background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
		>
			<svg class="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
				<path d="M12 21V9m0 0l-4 4m4-4l4 4M5 3h14" />
			</svg>
			<span>{busy === 'import' ? 'Loading' : 'Import'}</span>
		</button>
	</div>

	<label class="flex items-center gap-2 text-[11px]" style="color: var(--text-secondary);">
		<input type="checkbox" bind:checked={keepProjectDir} class="w-3.5 h-3.5" />
		<span data-tooltip="Keep the active project's project_dir when importing">Keep project folder</span>
	</label>

	{#if status}
		<div class="text-[11px] px-3 py-2 truncate" title={status} style="color: {statusTone === 'success' ? 'var(--success)' : statusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {statusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : statusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);">
			{status}
		</div>
	{/if}
</div>
