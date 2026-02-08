<script>
	import PathInput from './PathInput.svelte';
	import { projectConfig } from '$lib/stores/project.js';

	let { label, value = '', onchange, placeholder = '', tooltip = '', disabled = false, showFiles = false, scanType = '' } = $props();

	let scanning = $state(false);
	let scanResults = $state([]);
	let showScanResults = $state(false);
	let downloading = $state(false);
	let downloadStatus = $state('');
	let showDownloadMenu = $state(false);
	let downloadPresets = $state({});

	// Fetch download presets on first interaction
	let presetsFetched = false;
	async function ensurePresets() {
		if (presetsFetched) return;
		presetsFetched = true;
		try {
			const res = await fetch('/api/fs/download-presets');
			if (res.ok) {
				const data = await res.json();
				downloadPresets = data.presets || {};
			}
		} catch {}
	}

	function relevantPresets() {
		const all = Object.entries(downloadPresets);
		if (scanType === 'ltx2') return all.filter(([k]) => k.startsWith('ltx'));
		if (scanType === 'gemma') return all.filter(([k]) => k.startsWith('gemma'));
		return all;
	}

	function fireChange(val) {
		if (onchange) onchange(val);
	}

	async function handleScan() {
		if (!scanType) return;
		scanning = true;
		scanResults = [];
		showScanResults = true;
		try {
			const res = await fetch(`/api/fs/scan-checkpoints?type=${scanType}`);
			if (res.ok) {
				const data = await res.json();
				scanResults = data.results || [];
			}
		} catch {}
		scanning = false;
	}

	function selectResult(path) {
		showScanResults = false;
		fireChange(path);
	}

	async function handleDownload(presetKey) {
		const destDir = $projectConfig?.project_dir;
		if (!destDir) {
			downloadStatus = 'No project directory set';
			return;
		}
		showDownloadMenu = false;
		downloading = true;
		downloadStatus = 'Downloading...';
		try {
			const res = await fetch('/api/fs/download-model', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ preset: presetKey, dest_dir: destDir }),
			});
			if (res.ok) {
				const data = await res.json();
				downloadStatus = 'Done!';
				fireChange(data.path || '');
			} else {
				const err = await res.json();
				downloadStatus = err.detail || 'Download failed';
			}
		} catch (e) {
			downloadStatus = 'Download failed: ' + e.message;
		}
		downloading = false;
		setTimeout(() => { downloadStatus = ''; }, 5000);
	}

	function toggleDownloadMenu() {
		ensurePresets();
		showDownloadMenu = !showDownloadMenu;
		showScanResults = false;
	}

	function handleBrowseSelect(path) {
		fireChange(path);
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="relative" oninput={(e) => { if (e.target?.tagName === 'INPUT') fireChange(e.target.value); }}>
	<div class="flex items-end gap-1.5">
		<div class="flex-1">
			<PathInput {label} {value} {placeholder} {tooltip} {disabled} {showFiles} onselect={handleBrowseSelect} />
		</div>
		{#if scanType}
			<button
				type="button"
				onclick={handleScan}
				disabled={disabled || scanning}
				class="px-2 py-1.5 text-[11px] font-medium flex-shrink-0 flex items-center gap-1 disabled:opacity-40"
				style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
				onmouseenter={(e) => { if (!scanning) e.currentTarget.style.background = 'var(--border)'; }}
				onmouseleave={(e) => e.currentTarget.style.background = 'var(--bg-elevated)'}
				title="Scan filesystem for matching files"
			>
				<svg class="w-3 h-3 {scanning ? 'animate-spin' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
				Find
			</button>
			<div class="relative">
				<button
					type="button"
					onclick={toggleDownloadMenu}
					disabled={disabled || downloading}
					class="px-2 py-1.5 text-[11px] font-medium flex-shrink-0 flex items-center gap-1 disabled:opacity-40"
					style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
					onmouseenter={(e) => { if (!downloading) e.currentTarget.style.background = 'var(--border)'; }}
					onmouseleave={(e) => e.currentTarget.style.background = 'var(--bg-elevated)'}
					title="Download from HuggingFace"
				>
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
					{downloading ? '...' : 'DL'}
				</button>

				{#if showDownloadMenu}
					<!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
					<div class="fixed inset-0 z-40" onclick={() => showDownloadMenu = false}></div>
					<div class="absolute right-0 top-full mt-1 z-50 min-w-[240px] p-1" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); box-shadow: var(--shadow-lg);">
						<div class="px-2 py-1 text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Download to project directory</div>
						{#each relevantPresets() as [key, preset]}
							<button
								onclick={() => handleDownload(key)}
								class="w-full text-left px-2.5 py-2 text-[12px] flex items-center gap-2"
								style="color: var(--text-secondary); border-radius: var(--radius-sm);"
								onmouseenter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
								onmouseleave={(e) => e.currentTarget.style.background = 'transparent'}
							>
								<svg class="w-3.5 h-3.5 flex-shrink-0" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
								<span>{preset.label}</span>
							</button>
						{/each}
						{#if relevantPresets().length === 0}
							<div class="px-2.5 py-2 text-[11px]" style="color: var(--text-muted);">No presets available</div>
						{/if}
					</div>
				{/if}
			</div>
		{/if}
	</div>

	<!-- Download status -->
	{#if downloadStatus}
		<div class="mt-1 text-[11px] px-2 py-1" style="color: {downloadStatus.startsWith('Done') ? 'var(--success)' : downloadStatus === 'Downloading...' ? 'var(--accent)' : 'var(--danger)'}; background: {downloadStatus.startsWith('Done') ? 'var(--success-muted, rgba(34,197,94,0.1))' : downloadStatus === 'Downloading...' ? 'var(--accent-muted)' : 'var(--danger-muted)'}; border-radius: var(--radius-sm);">
			{#if downloading}
				<span class="inline-flex items-center gap-1.5">
					<svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
					{downloadStatus}
				</span>
			{:else}
				{downloadStatus}
			{/if}
		</div>
	{/if}

	<!-- Scan results popup -->
	{#if showScanResults}
		<!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
		<div class="fixed inset-0 z-40" onclick={() => showScanResults = false}></div>
		<div class="absolute left-0 right-0 top-full mt-1 z-50 max-h-[300px] overflow-y-auto" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); box-shadow: var(--shadow-lg);">
			{#if scanning}
				<div class="px-3 py-4 text-center">
					<span class="text-[12px] flex items-center justify-center gap-2" style="color: var(--text-muted);">
						<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
						Scanning...
					</span>
				</div>
			{:else if scanResults.length === 0}
				<div class="px-3 py-4 text-center text-[12px]" style="color: var(--text-muted);">No matching files found</div>
			{:else}
				<div class="px-2 py-1 text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Found {scanResults.length} result{scanResults.length === 1 ? '' : 's'}</div>
				{#each scanResults as result}
					<button
						onclick={() => selectResult(result)}
						class="w-full text-left px-3 py-2 text-[12px] font-mono truncate"
						style="color: var(--text-secondary); border-radius: var(--radius-sm);"
						onmouseenter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
						onmouseleave={(e) => e.currentTarget.style.background = 'transparent'}
					>{result}</button>
				{/each}
			{/if}
		</div>
	{/if}
</div>
