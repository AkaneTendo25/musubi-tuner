<script>
	import { page } from '$app/state';
	import { projectConfig, projectLoaded, closeProject } from '$lib/stores/project.js';
	import { processStatuses, processValidation } from '$lib/stores/processes.js';
	import { theme, setTheme, THEMES } from '$lib/stores/theme.js';
	import { uiMode, setUiMode } from '$lib/stores/uiMode.js';
	import { onMount, onDestroy } from 'svelte';

	let showThemePicker = $state(false);
	let systemInfo = $state(null);
	let _sysInfoTimer = null;

	const navItems = [
		{ href: '/', label: 'Overview', group: 'Workflow', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4', always: true, statusTypes: ['cache_latents', 'cache_text', 'training'] },
		{ href: '/dataset', label: 'Dataset', group: 'Workflow', icon: 'M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M4 7c0-2 1-3 3-3h10c2 0 3 1 3 3M4 7h16M9 11h6' },
		{ href: '/caching', label: 'Caching', group: 'Workflow', icon: 'M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M9 3v4M15 3v4M4 11h16', processTypes: ['cache_latents', 'cache_text'] },
		{ href: '/samples', label: 'Samples', group: 'Tools', icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
		{ href: '/training', label: 'Training', group: 'Workflow', icon: 'M13 10V3L4 14h7v7l9-11h-7z', processTypes: ['training', 'remote_stage_launcher', 'remote_stage_server'] },
		{ href: '/training/techniques', label: 'Techniques', group: 'Advanced', icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z', processTypes: ['slider_training'], advancedOnly: true },
		{ href: '/training/dashboard', label: 'Monitor', group: 'Workflow', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z', statusTypes: ['training'] },
		{ href: '/inference', label: 'Inference', group: 'Tools', icon: 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z', processTypes: ['inference'] },
		{ href: '/settings', label: 'Manage', icon: 'M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7 7 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.248a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a7 7 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.282a1.125 1.125 0 00-.645-.869 7 7 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.248a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.991a7 7 0 010-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.248a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28z M15 12a3 3 0 11-6 0 3 3 0 016 0z', always: true },
	];
	const navGroups = ['Workflow', 'Tools', 'Advanced', 'System'];

	let currentTheme = $derived(THEMES.find((t) => t.id === $theme) || THEMES[0]);
	let remoteStageEnabled = $derived(Boolean($projectConfig?.training?.ltx2_remote_stage));

	// Dashboard dimming: dim when training is not active
	let trainingActive = $derived.by(() => {
		const ts = $processStatuses.training;
		return ts && (ts.state === 'running' || ts.state === 'stopping' || ts.state === 'finished');
	});

	function statusDot(processTypes) {
		if (!processTypes) return null;
		const statuses = $processStatuses;
		for (const t of processTypes) {
			const s = statuses[t];
			if (s && s.state === 'running') return 'running';
			if (s && s.state === 'stopping') return 'stopping';
		}
		for (const t of processTypes) {
			const s = statuses[t];
			if (s && s.state === 'error') return 'error';
		}
		for (const t of processTypes) {
			const s = statuses[t];
			if (s && s.state === 'finished') return 'finished';
		}
		return null;
	}

	function validationInfo(processTypes) {
		if (!processTypes) return { count: 0, title: '' };
		const reports = $processValidation;
		const seen = new Set();
		const lines = [];
		let count = 0;
		for (const t of processTypes) {
			const r = reports[t];
			if (!r || !Array.isArray(r.errors)) continue;
			count += r.errors.length;
			for (const err of r.errors) {
				const key = (err.label || err.field || '') + '|' + (err.message || '');
				if (seen.has(key)) continue;
				seen.add(key);
				const prefix = err.label ? `${err.label}: ` : '';
				lines.push(`${prefix}${err.message || ''}`);
			}
		}
		return { count, title: lines.join('\n') };
	}

	function visibleProcessTypes(processTypes) {
		if (!processTypes) return null;
		if (remoteStageEnabled) return processTypes;
		return processTypes.filter((type) => !type.startsWith('remote_stage_'));
	}

	function dotStyle(status) {
		if (status === 'running') return 'background: var(--success); box-shadow: 0 0 6px var(--success);';
		if (status === 'stopping') return 'background: var(--warning);';
		if (status === 'error') return 'background: var(--danger);';
		if (status === 'finished') return 'background: var(--info);';
		return '';
	}

	function navGroup(item) {
		if (item.group) return item.group;
		if (item.href === '/settings') return 'System';
		if (item.href === '/training/techniques') return 'Advanced';
		if (item.href === '/samples' || item.href === '/inference') return 'Tools';
		return 'Workflow';
	}

	function navLabel(item) {
		if (item.href === '/') return 'Overview';
		if (item.href === '/training/dashboard') return 'Monitor';
		if (item.href === '/settings') return 'Setup';
		return item.label;
	}

	let gpu = $derived(systemInfo?.gpus?.[0]);

	onMount(async () => {
		try {
			const res = await fetch('/api/system/info');
			if (res.ok) systemInfo = await res.json();
		} catch {}
		_sysInfoTimer = setInterval(async () => {
			try {
				const res = await fetch('/api/system/info');
				if (res.ok) systemInfo = await res.json();
			} catch {}
		}, 30000);
	});

	onDestroy(() => {
		if (_sysInfoTimer) clearInterval(_sysInfoTimer);
	});
</script>

<nav class="w-52 flex-shrink-0 flex flex-col h-screen sticky top-0" style="background: var(--sidebar-bg); border-right: 1px solid var(--border-subtle);">
	<!-- Logo -->
	<div class="px-4 py-4" style="border-bottom: 1px solid var(--border-subtle);">
		<div class="flex items-center gap-2">
			<div class="w-6 h-6 flex items-center justify-center flex-shrink-0" style="background: var(--logo-bg); box-shadow: var(--logo-shadow); border-radius: var(--logo-radius); clip-path: var(--logo-clip);">
				<span class="text-[10px] font-bold" style="color: var(--bg-base);">M</span>
			</div>
			<div class="text-[12px] font-semibold truncate flex-1" style="color: var(--text-primary);">
				{$projectConfig?.name || 'Musubi Tuner'}
			</div>
			{#if $projectLoaded}
				<button
					onclick={async () => { await closeProject(); window.location.href = '/'; }}
					class="flex-shrink-0 px-2 py-0.5 text-[10px] font-medium"
					style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);"
					onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
					onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
				>
					Close
				</button>
			{/if}
		</div>
	</div>

	<!-- Nav -->
	<div class="flex-1 py-2 px-2 space-y-3 overflow-y-auto">
		{#each navGroups as group}
			{@const visibleItems = navItems.filter((item) => navGroup(item) === group && (item.always || $projectLoaded) && (!item.advancedOnly || $uiMode === 'advanced'))}
			{#if visibleItems.length > 0}
				<div class="space-y-0.5">
					<div class="px-3 pb-1 text-[9px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">
						{group}
					</div>
					{#each visibleItems as item}
						{@const isActive = page.url.pathname === item.href}
						{@const processTypes = visibleProcessTypes(item.processTypes)}
						{@const statusTypes = visibleProcessTypes(item.statusTypes || item.processTypes)}
						{@const dot = statusDot(statusTypes)}
						{@const validation = validationInfo(processTypes)}
						{@const isDimmed = item.dimWhenIdle && !trainingActive && !isActive}
						<a
							href={item.href}
							title={validation.count > 0 ? validation.title : undefined}
							class="flex items-center gap-2 px-3 py-[7px] text-[12px] font-medium"
							style="
								border-radius: var(--radius-sm);
								{isDimmed ? 'opacity: 0.4;' : ''}
								{isActive
									? `background: var(--sidebar-active); color: var(--accent); box-shadow: var(--shadow-sm), inset 0 0 0 1px var(--border-subtle);`
									: `color: var(--text-secondary);`
								}
							"
							onmouseenter={(e) => { if (!isActive) { e.currentTarget.style.background = 'var(--sidebar-hover)'; e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.opacity = '1'; } }}
							onmouseleave={(e) => { if (!isActive) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-secondary)'; if (isDimmed) e.currentTarget.style.opacity = '0.4'; } }}
						>
							<svg class="w-[14px] h-[14px] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
								<path d="{item.icon}"/>
							</svg>
							<span class="flex-1">{navLabel(item)}</span>
							{#if validation.count > 0}
								<span
									class="text-[10px] font-bold tabular-nums flex-shrink-0 inline-flex items-center justify-center"
									style="background: var(--danger); color: var(--bg-base); border-radius: 4px; height: 16px; min-width: 16px; padding: 0 4px;"
								>{validation.count}</span>
							{/if}
							{#if dot}
								<span class="w-[6px] h-[6px] rounded-full flex-shrink-0" style="{dotStyle(dot)}"></span>
							{/if}
						</a>
					{/each}
				</div>
			{/if}
		{/each}

	</div>

	<!-- Hardware stats -->
	{#if systemInfo}
		<div class="flex-shrink-0 px-3 py-2.5 space-y-1" style="border-top: 1px solid var(--border-subtle);">
			{#if gpu}
				<div class="flex items-center gap-1.5 text-[11px]" style="color: var(--text-secondary);">
					<span style="font-size: 12px;">🖥️</span>
					<span class="truncate flex-1">{gpu.name.replace('NVIDIA ','').replace('GeForce ','')}</span>
					<span class="tabular-nums flex-shrink-0 font-medium" style="color: var(--accent);">{(gpu.vram_used_mb / 1024).toFixed(1)}/{(gpu.vram_total_mb / 1024).toFixed(0)}G</span>
				</div>
			{/if}
			{#if systemInfo.ram}
				<div class="flex items-center gap-1.5 text-[11px]" style="color: var(--text-secondary);">
					<span style="font-size: 12px;">💾</span>
					<span class="flex-1">RAM</span>
					<span class="tabular-nums flex-shrink-0">{systemInfo.ram.used_gb}/{systemInfo.ram.total_gb} GB</span>
				</div>
			{/if}
			{#if systemInfo.disk}
				<div class="flex items-center gap-1.5 text-[11px]" style="color: var(--text-secondary);">
					<span style="font-size: 12px;">💽</span>
					<span class="flex-1">Disk</span>
					<span class="tabular-nums flex-shrink-0">{systemInfo.disk.free_gb} GB free</span>
				</div>
			{/if}
			{#if systemInfo.cpu}
				<div class="flex items-center gap-1.5 text-[11px]" style="color: var(--text-secondary);">
					<span style="font-size: 12px;">🧠</span>
					<span class="flex-1">CPU</span>
					<span class="tabular-nums flex-shrink-0">{systemInfo.cpu.cores}c</span>
				</div>
			{/if}
			{#if systemInfo.python}
				<div class="flex items-center gap-1.5 text-[11px]" style="color: var(--text-secondary);">
					<span style="font-size: 12px;">🐍</span>
					<span class="flex-1">Python</span>
					<span class="flex-shrink-0">{systemInfo.python}</span>
				</div>
			{/if}
		</div>
	{/if}

	<div class="flex-shrink-0 px-2 py-2 space-y-1" style="border-top: 1px solid var(--border-subtle);">
		<div class="px-3 text-[10px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">
			UI Mode
		</div>
		<div class="grid grid-cols-2 gap-1">
			<button
				type="button"
				onclick={() => setUiMode('basic')}
				class="px-3 py-[7px] text-[11px] font-medium"
				style="
					border-radius: var(--radius-sm);
					{$uiMode === 'basic'
						? `background: var(--sidebar-active); color: var(--accent); box-shadow: var(--shadow-sm), inset 0 0 0 1px var(--border-subtle);`
						: `background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border-subtle);`
					}
				"
			>
				Basic
			</button>
			<button
				type="button"
				onclick={() => setUiMode('advanced')}
				class="px-3 py-[7px] text-[11px] font-medium"
				style="
					border-radius: var(--radius-sm);
					{$uiMode === 'advanced'
						? `background: var(--sidebar-active); color: var(--accent); box-shadow: var(--shadow-sm), inset 0 0 0 1px var(--border-subtle);`
						: `background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border-subtle);`
					}
				"
			>
				Advanced
			</button>
		</div>
	</div>

	<!-- Theme picker — pinned at bottom -->
	<div class="flex-shrink-0 px-2 py-2 relative" style="border-top: 1px solid var(--border-subtle);">
		<button
			onclick={() => showThemePicker = !showThemePicker}
			class="w-full flex items-center gap-2 px-3 py-[7px] text-[12px] font-medium"
			style="color: var(--text-muted); border-radius: var(--radius-sm);"
			onmouseenter={(e) => { e.currentTarget.style.background = 'var(--sidebar-hover)'; e.currentTarget.style.color = 'var(--text-secondary)'; }}
			onmouseleave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; }}
		>
			<span class="w-3 h-3 rounded-full flex-shrink-0" style="background: {currentTheme.swatch}; box-shadow: 0 0 6px {currentTheme.swatch}40;"></span>
			<span class="flex-1 text-left">{currentTheme.name}</span>
			<svg class="w-3 h-3 transition-transform {showThemePicker ? 'rotate-180' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M19 9l-7 7-7-7"/></svg>
		</button>

		{#if showThemePicker}
			<button
				type="button"
				aria-label="Close theme picker"
				class="fixed inset-0 z-40"
				style="background: transparent;"
				onclick={() => showThemePicker = false}
			></button>
			<div class="absolute bottom-full left-2 right-2 mb-1 p-1.5 z-50" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); box-shadow: var(--shadow-lg);">
				{#each THEMES as t}
					<button
						onclick={() => { setTheme(t.id); showThemePicker = false; }}
						class="w-full flex items-center gap-2.5 px-2.5 py-[7px] text-[11px] font-medium transition-colors"
						style="border-radius: var(--radius-sm); {$theme === t.id ? `background: var(--accent-muted); color: var(--accent);` : `color: var(--text-secondary);`}"
						onmouseenter={(e) => { if ($theme !== t.id) e.currentTarget.style.background = 'var(--bg-hover)'; }}
						onmouseleave={(e) => { if ($theme !== t.id) e.currentTarget.style.background = 'transparent'; }}
					>
						<span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background: {t.swatch}; box-shadow: 0 0 4px {t.swatch}30;"></span>
						<span class="flex-1 text-left">{t.name}</span>
						{#if $theme === t.id}
							<svg class="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M5 13l4 4L19 7"/></svg>
						{/if}
					</button>
				{/each}
			</div>
		{/if}
	</div>
</nav>
