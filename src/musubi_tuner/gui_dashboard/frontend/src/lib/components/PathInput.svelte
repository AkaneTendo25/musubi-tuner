<script>
	import { projectConfig } from '$lib/stores/project.js';

	let {
		label,
		value = $bindable(''),
		placeholder = '',
		tooltip = '',
		disabled = false,
		showFiles = false,
		invalid = false,
		error = '',
		onselect,
		oninput
	} = $props();

	let showBrowser = $state(false);
	let browserEntries = $state([]);
	let browserPath = $state('');
	let browserParent = $state(null);
	let serverCwd = $state('');

	async function openBrowser() {
		showBrowser = true;
		let startPath = value || $projectConfig?.project_dir || '';
		if (!startPath) {
			if (!serverCwd) {
				try {
					const res = await fetch('/api/fs/cwd');
					if (res.ok) serverCwd = (await res.json()).cwd || '';
				} catch {}
			}
			startPath = serverCwd;
		}
		await browse(startPath);
	}

	async function browse(path) {
		try {
			const params = new URLSearchParams({ path, show_files: showFiles });
			const res = await fetch(`/api/fs/browse?${params}`);
			if (res.ok) {
				const data = await res.json();
				browserEntries = data.entries || [];
				browserPath = data.path || '';
				browserParent = data.parent;
			}
		} catch {
			// ignore
		}
	}

	function applyPath(path) {
		value = path;
		if (oninput) oninput({ target: { value: path } });
		if (onselect) onselect(path);
	}

	function selectEntry(entry) {
		if (entry.is_dir) {
			browse(entry.path);
		} else {
			applyPath(entry.path);
			showBrowser = false;
		}
	}

	function selectCurrent() {
		applyPath(browserPath);
		showBrowser = false;
	}

	function goUp() {
		if (browserParent !== null) {
			browse(browserParent);
		} else if (browserPath) {
			// At drive root (Windows) or filesystem root — browse empty to show drive list
			browse('');
		}
	}
</script>

<label class="block" data-tooltip={tooltip || undefined}>
	<span class="text-xs font-medium uppercase tracking-wider" style="color: {invalid ? 'var(--danger)' : 'var(--text-muted)'}; font-family: var(--font-label);">{label}</span>
	<div class="mt-1 flex gap-1.5">
		<input
			type="text"
			bind:value
			{placeholder}
			{disabled}
			oninput={oninput}
			class="flex-1 px-3 py-2 text-sm transition-colors disabled:opacity-40"
			style="background: var(--bg-input); border: 1px solid {invalid ? 'var(--danger)' : 'var(--border)'}; color: var(--text-primary); border-radius: var(--radius-sm);"
		/>
		<!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
		<button
			type="button"
			onclick={openBrowser}
			{disabled}
			class="px-3 py-2 text-sm font-medium disabled:opacity-40 flex-shrink-0"
			style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
			onmouseenter={(e) => e.currentTarget.style.background = 'var(--border)'}
			onmouseleave={(e) => e.currentTarget.style.background = 'var(--bg-elevated)'}
		>...</button>
	</div>
	{#if error}
		<div class="mt-1 text-[11px]" style="color: var(--danger);">{error}</div>
	{/if}
</label>

{#if showBrowser}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="fixed inset-0 z-50 flex items-center justify-center" style="background: rgba(0,0,0,0.5); backdrop-filter: blur(4px);" onkeydown={(e) => e.key === 'Escape' && (showBrowser = false)} onclick={() => showBrowser = false}>
		<!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
		<div class="w-[520px] max-h-[70vh] flex flex-col overflow-hidden" style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-lg); box-shadow: var(--shadow-lg);" onclick={(e) => e.stopPropagation()}>
			<div class="px-4 py-3 flex items-center gap-2" style="border-bottom: 1px solid var(--border-subtle);">
				<button
					onclick={goUp}
					disabled={browserParent === null && !browserPath}
					class="px-2.5 py-1 text-[12px] font-medium disabled:opacity-30"
					style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
				>..</button>
				<span class="text-[13px] truncate flex-1" style="color: var(--text-primary);">{browserPath || 'Select path'}</span>
				<button
					onclick={selectCurrent}
					class="px-3 py-1 text-[12px] font-medium"
					style="background: var(--accent); color: white; border-radius: var(--radius-full);"
					onmouseenter={(e) => e.currentTarget.style.background = 'var(--accent-hover)'}
					onmouseleave={(e) => e.currentTarget.style.background = 'var(--accent)'}
				>Select</button>
			</div>
			<div class="flex-1 overflow-y-auto">
				{#each browserEntries as entry}
					<button
						onclick={() => selectEntry(entry)}
						class="w-full text-left px-4 py-2 text-[13px] flex items-center gap-2.5 transition-colors"
						style="color: {entry.is_dir ? 'var(--accent)' : 'var(--text-secondary)'};"
						onmouseenter={(e) => e.currentTarget.style.background = 'var(--bg-elevated)'}
						onmouseleave={(e) => e.currentTarget.style.background = 'transparent'}
					>
						<svg class="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
							{#if entry.is_dir}
								<path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
							{:else}
								<path d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>
							{/if}
						</svg>
						<span class="truncate">{entry.name}</span>
					</button>
				{/each}
				{#if browserEntries.length === 0}
					<div class="px-4 py-10 text-center text-[13px]" style="color: var(--text-muted);">Empty directory</div>
				{/if}
			</div>
			<div class="px-4 py-2.5" style="border-top: 1px solid var(--border-subtle);">
				<button
					onclick={() => showBrowser = false}
					class="text-[12px] font-medium"
					style="color: var(--text-muted);"
					onmouseenter={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
					onmouseleave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
				>Close</button>
			</div>
		</div>
	</div>
{/if}
