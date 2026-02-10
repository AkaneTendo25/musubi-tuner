<script>
	import { projectConfig } from '$lib/stores/project.js';
	import { onMount, onDestroy } from 'svelte';

	let { processType, defaultFilename = 'command.bat' } = $props();

	let command = $state('');
	let loading = $state(false);
	let initialLoad = $state(true);
	// svelte-ignore state_referenced_locally
	let filename = $state(defaultFilename);
	let saveStatus = $state('');
	let copied = $state(false);
	let showOverwriteConfirm = $state(false);

	// Pending buffer approach
	let lastFetchedSnapshot = '';
	let lastChangeTime = 0;
	let checkInterval = null;

	async function fetchCommand() {
		loading = true;
		try {
			const res = await fetch(`/api/processes/${processType}/command-preview`);
			if (res.ok) {
				const data = await res.json();
				command = data.command;
				// Update last fetched snapshot
				lastFetchedSnapshot = $projectConfig ? JSON.stringify($projectConfig) : '';
			} else {
				const err = await res.json();
				command = `Error: ${err.detail}`;
			}
		} catch (e) {
			command = `Error: ${e.message}`;
		}
		loading = false;
		initialLoad = false;
	}

	function checkForUpdates() {
		const currentSnapshot = $projectConfig ? JSON.stringify($projectConfig) : '';

		// If config changed since last fetch
		if (currentSnapshot && currentSnapshot !== lastFetchedSnapshot) {
			const timeSinceChange = Date.now() - lastChangeTime;

			// Wait at least 500ms since last change before fetching
			if (timeSinceChange >= 500) {
				console.log('[CommandPanel] Config changed, fetching command...');
				fetchCommand();
			} else {
				// Still changing, show loading indicator
				loading = true;
			}
		}
	}

	// Track config changes and mark change time
	$effect(() => {
		const cfg = $projectConfig;
		if (cfg) {
			const currentSnapshot = JSON.stringify(cfg);
			if (currentSnapshot !== lastFetchedSnapshot) {
				lastChangeTime = Date.now();
				loading = true; // Show loading immediately
			}
		}
	});

	onMount(() => {
		// Initial fetch
		fetchCommand();

		// Check for updates every 200ms
		checkInterval = setInterval(checkForUpdates, 200);
	});

	onDestroy(() => {
		if (checkInterval) clearInterval(checkInterval);
	});

	// Build filename with project name
	let effectiveFilename = $derived.by(() => {
		const cfg = $projectConfig;
		const name = cfg?.name;
		if (name) {
			const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
			if (slug && !filename.includes(slug)) {
				return `${slug}_${defaultFilename}`;
			}
		}
		return filename;
	});

	async function saveToFile() {
		const config = $projectConfig;
		if (!config?.project_dir) {
			saveStatus = 'No project directory';
			return;
		}
		const path = `${config.project_dir}/${effectiveFilename}`.replace(/\\/g, '/');

		// Check if file exists
		try {
			const checkRes = await fetch(`/api/fs/exists?path=${encodeURIComponent(path)}`);
			if (checkRes.ok) {
				const info = await checkRes.json();
				if (info.exists) {
					showOverwriteConfirm = true;
					return;
				}
			}
		} catch {}

		await doSave(path);
	}

	async function doSave(path) {
		const config = $projectConfig;
		if (!path) {
			path = `${config.project_dir}/${effectiveFilename}`.replace(/\\/g, '/');
		}
		showOverwriteConfirm = false;
		try {
			const res = await fetch('/api/fs/write-file', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ path, content: command })
			});
			if (res.ok) {
				saveStatus = 'Saved!';
				setTimeout(() => saveStatus = '', 2000);
			} else {
				const err = await res.json();
				saveStatus = `Error: ${err.detail}`;
			}
		} catch (e) {
			saveStatus = `Error: ${e.message}`;
		}
	}

	async function copyToClipboard() {
		try {
			await navigator.clipboard.writeText(command);
			copied = true;
			setTimeout(() => copied = false, 2000);
		} catch {
			const ta = document.createElement('textarea');
			ta.value = command;
			document.body.appendChild(ta);
			ta.select();
			document.execCommand('copy');
			document.body.removeChild(ta);
			copied = true;
			setTimeout(() => copied = false, 2000);
		}
	}
</script>

<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm); position: relative; overflow: hidden;">
	<!-- Accent gradient line -->
	<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>

	<div class="px-3 py-2 flex items-center justify-between" style="border-bottom: 1px solid var(--border-subtle);">
		<span class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Command</span>
		{#if loading}
			<span class="text-[10px] flex items-center gap-1" style="color: var(--text-muted);">
				<svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
				updating...
			</span>
		{/if}
	</div>

	<!-- Command display -->
	{#if initialLoad}
		<div class="p-3" style="background: var(--console-bg);">
			<div class="space-y-2 animate-pulse">
				<div class="h-3 rounded" style="background: var(--border); width: 80%;"></div>
				<div class="h-3 rounded" style="background: var(--border); width: 60%;"></div>
				<div class="h-3 rounded" style="background: var(--border); width: 70%;"></div>
			</div>
		</div>
	{:else}
		<pre
			class="text-[11px] leading-5 p-3 overflow-auto max-h-48 whitespace-pre-wrap break-all font-mono"
			style="background: var(--console-bg); color: var(--console-text); margin: 0; border-radius: 0; box-shadow: inset 0 2px 6px rgba(0,0,0,.3);"
		>{command || 'No command available'}</pre>
	{/if}

	<!-- Footer with save/copy -->
	<div class="px-3 py-2 flex items-center gap-2" style="border-top: 1px solid var(--border-subtle);">
		<input
			type="text"
			bind:value={filename}
			placeholder={effectiveFilename}
			class="text-[11px] px-2 py-1 flex-1 min-w-0 max-w-[200px]"
			style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none; font-family: var(--font-mono, monospace);"
			onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
			onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
		/>
		<button
			onclick={saveToFile}
			class="text-[11px] font-medium px-2.5 py-1"
			style="background: var(--accent-muted); border: 1px solid var(--accent); color: var(--accent); border-radius: var(--radius-sm);"
			onmouseenter={(e) => { e.currentTarget.style.background = 'var(--accent)'; e.currentTarget.style.color = 'var(--bg-base)'; }}
			onmouseleave={(e) => { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; }}
		>Save</button>
		<button
			onclick={copyToClipboard}
			class="text-[11px] font-medium px-2.5 py-1"
			style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
			onmouseenter={(e) => { e.currentTarget.style.background = 'var(--bg-hover, var(--bg-elevated))'; e.currentTarget.style.color = 'var(--text-primary)'; }}
			onmouseleave={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--text-secondary)'; }}
		>{copied ? 'Copied!' : 'Copy'}</button>
		{#if saveStatus}
			<span class="text-[10px] font-medium" style="color: {saveStatus.startsWith('Saved') ? 'var(--success)' : 'var(--danger)'};">{saveStatus}</span>
		{/if}
	</div>

	<!-- Overwrite confirmation -->
	{#if showOverwriteConfirm}
		<div class="px-3 py-2 flex items-center gap-2" style="background: var(--warning-muted, rgba(234,179,8,0.1)); border-top: 1px solid var(--warning);">
			<svg class="w-3.5 h-3.5 flex-shrink-0" style="color: var(--warning);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
			<span class="text-[11px] flex-1" style="color: var(--warning);">File already exists. Overwrite?</span>
			<button
				onclick={() => doSave()}
				class="text-[11px] font-medium px-2 py-0.5"
				style="background: var(--warning); color: var(--bg-base); border-radius: var(--radius-sm);"
			>Overwrite</button>
			<button
				onclick={() => showOverwriteConfirm = false}
				class="text-[11px] font-medium px-2 py-0.5"
				style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
			>Cancel</button>
		</div>
	{/if}
</div>
