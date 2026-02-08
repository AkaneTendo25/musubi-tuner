<script>
	import { page } from '$app/state';
	import { projectConfig, projectLoaded, closeProject } from '$lib/stores/project.js';
	import { processStatuses } from '$lib/stores/processes.js';
	import { theme, setTheme, THEMES } from '$lib/stores/theme.js';
	import { onMount, onDestroy } from 'svelte';

	let showThemePicker = $state(false);
	let systemInfo = $state(null);
	let _sysInfoTimer = null;

	const navItems = [
		{ href: '/', label: 'Project', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4', always: true },
		{ href: '/dataset', label: 'Dataset', icon: 'M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M4 7c0-2 1-3 3-3h10c2 0 3 1 3 3M4 7h16M9 11h6' },
		{ href: '/caching', label: 'Caching', icon: 'M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M9 3v4M15 3v4M4 11h16', processTypes: ['cache_latents', 'cache_text'] },
		{ href: '/samples', label: 'Samples', icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
		{ href: '/training', label: 'Training', icon: 'M13 10V3L4 14h7v7l9-11h-7z', processTypes: ['training'] },
		{ href: '/training/techniques', label: 'Techniques', icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z', processTypes: ['slider_training'] },
		{ href: '/training/dashboard', label: 'Dashboard', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
		{ href: '/inference', label: 'Inference', icon: 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z', processTypes: ['inference'] }
	];

	let currentTheme = $derived(THEMES.find((t) => t.id === $theme) || THEMES[0]);

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

	function dotStyle(status) {
		if (status === 'running') return 'background: var(--success); box-shadow: 0 0 6px var(--success);';
		if (status === 'stopping') return 'background: var(--warning);';
		if (status === 'error') return 'background: var(--danger);';
		if (status === 'finished') return 'background: var(--info);';
		return '';
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
					class="w-5 h-5 flex items-center justify-center flex-shrink-0"
					style="color: var(--text-muted); border-radius: var(--radius-sm);"
					onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; }}
					onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
					title="Close project"
				>
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
				</button>
			{/if}
		</div>
	</div>

	<!-- Nav -->
	<div class="flex-1 py-2 px-2 space-y-0.5 overflow-y-auto">
		{#each navItems as item}
			{@const isActive = page.url.pathname === item.href}
			{@const dot = statusDot(item.processTypes)}
			{@const isDimmed = item.dimWhenIdle && !trainingActive && !isActive}
			{@const visible = item.always || $projectLoaded}
			{#if visible}
				<a
					href={item.href}
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
					<span class="flex-1">{item.label}</span>
					{#if dot}
						<span class="w-[6px] h-[6px] rounded-full flex-shrink-0" style="{dotStyle(dot)}"></span>
					{/if}
				</a>
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
			<!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
			<div class="fixed inset-0 z-40" onclick={() => showThemePicker = false}></div>
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
						<span class="text-[10px]" style="color: var(--text-muted);">{t.desc}</span>
						{#if $theme === t.id}
							<svg class="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M5 13l4 4L19 7"/></svg>
						{/if}
					</button>
				{/each}
			</div>
		{/if}
	</div>
</nav>
