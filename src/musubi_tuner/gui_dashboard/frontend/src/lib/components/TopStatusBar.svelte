<script>
	import { page } from '$app/state';
	import { projectConfig, projectLoaded } from '$lib/stores/project.js';
	import { processStatuses } from '$lib/stores/processes.js';
	import { getConditioningIssues } from '$lib/utils/conditioningStatus.js';
	import { estimateLatentCaching, estimateTextCaching, estimateTraining } from '$lib/utils/vramEstimate.js';
	import { onMount, onDestroy } from 'svelte';

	let systemInfo = $state(null);
	let _sysInfoTimer = null;

	let gpu = $derived(systemInfo?.gpus?.[0]);
	let gpuName = $derived(gpu ? gpu.name.replace('NVIDIA ', '').replace('GeForce ', '') : '');
	let vramTotal = $derived(gpu ? gpu.vram_total_mb / 1024 : 24);
	let vramUsed = $derived(gpu ? gpu.vram_used_mb / 1024 : null);
	let vramPercent = $derived(percentOf(vramUsed, vramTotal));
	let gpuLoadPercent = $derived(clampPercent(gpu?.utilization ?? vramPercent));
	let ramPercent = $derived(clampPercent(systemInfo?.ram?.percent));
	let diskUsedPercent = $derived(clampPercent(systemInfo?.disk?.percent));
	let cpuPercent = $derived(clampPercent(systemInfo?.cpu?.utilization));
	const pageTitles = {
		'/': 'Overview',
		'/dataset': 'Dataset',
		'/caching': 'Caching',
		'/tools': 'Tools',
		'/samples': 'Samples',
		'/training': 'Training',
		'/training/conditioning': 'Conditioning',
		'/training/techniques': 'Methods',
		'/training/full-finetune': 'Fine-tuning',
		'/training/dashboard': 'Monitor',
		'/inference': 'Inference',
		'/settings': 'Setup',
	};
	let currentPageTitle = $derived(pageTitles[page.url.pathname] || 'Dashboard');
	let conditioningIssues = $derived(getConditioningIssues($projectConfig));
	let hasConditioningIssue = $derived(conditioningIssues.errors.length > 0 || conditioningIssues.warnings.length > 0);
	let conditioningIssueTitle = $derived(conditioningIssues.all.map((issue) => issue.msg).join('\n'));
	let monitorConfigHref = $derived.by(() => {
		const training = $processStatuses.training || { state: 'idle' };
		const full = $processStatuses.full_finetune || { state: 'idle' };
		if (full.state === 'running' || full.state === 'stopping' || full.state === 'finished' || full.state === 'error') return '/training/full-finetune';
		return '/training';
	});

	let latentCachingVram = $derived(estimateLatentCaching($projectConfig));
	let textCachingVram = $derived(estimateTextCaching($projectConfig));
	let trainingVram = $derived(estimateTraining($projectConfig));
	let cachingVram = $derived.by(() => {
		const estimates = [
			latentCachingVram ? { ...latentCachingVram, mode: 'Latents' } : null,
			textCachingVram ? { ...textCachingVram, mode: 'Text' } : null,
		].filter(Boolean);
		if (!estimates.length) return null;
		return estimates.reduce((largest, current) => Number(current.total || 0) > Number(largest.total || 0) ? current : largest, estimates[0]);
	});

	function vramChip(label, estimate, color, note = '') {
		const value = Number(estimate?.total || 0);
		const fits = value <= vramTotal;
		const delta = Math.abs(vramTotal - value);
		const percent = Math.min((value / Math.max(vramTotal, 1)) * 100, 100);
		return { label, value, color, note, fits, delta, percent };
	}

	let vramChips = $derived.by(() => {
		const chips = [];
		if (cachingVram) chips.push(vramChip('Cache', cachingVram, 'var(--info)', cachingVram.mode));
		if (trainingVram) chips.push(vramChip('Train', trainingVram, 'var(--warning)', trainingVram.blockwise ? 'Blockwise' : ''));
		return chips;
	});

	let activeProcess = $derived.by(() => {
		const order = [
			['training', 'Training'],
			['full_finetune', 'Fine-tune'],
			['cache_latents', 'Cache latents'],
			['cache_text', 'Cache text'],
			['cache_dino', 'Cache DINO'],
			['cache_preview', 'Cache preview'],
			['inference', 'Inference'],
			['remote_stage_launcher', 'Remote stage'],
			['remote_stage_server', 'Remote server'],
			['slider_training', 'Slider'],
		];
		for (const [type, label] of order) {
			const status = $processStatuses[type];
			if (status?.state === 'running' || status?.state === 'stopping') return { label, state: status.state };
		}
		for (const [type, label] of order) {
			const status = $processStatuses[type];
			if (status?.state === 'error') return { label, state: 'error' };
		}
		return { label: 'Idle', state: 'idle' };
	});

	function processColor(state) {
		if (state === 'running') return 'var(--success)';
		if (state === 'stopping') return 'var(--warning)';
		if (state === 'error') return 'var(--danger)';
		return 'var(--text-muted)';
	}

	function clampPercent(value) {
		const numeric = Number(value);
		if (!Number.isFinite(numeric)) return null;
		return Math.max(0, Math.min(100, numeric));
	}

	function percentOf(used, total) {
		const usedValue = Number(used);
		const totalValue = Number(total);
		if (!Number.isFinite(usedValue) || !Number.isFinite(totalValue) || totalValue <= 0) return null;
		return clampPercent((usedValue / totalValue) * 100);
	}

	function meterColor(percent) {
		if (percent == null) return 'var(--text-muted)';
		if (percent >= 90) return 'var(--danger)';
		if (percent >= 70) return 'var(--warning)';
		if (percent >= 35) return 'var(--accent)';
		return 'var(--success)';
	}

	function meterStyle(percent) {
		const value = clampPercent(percent) ?? 0;
		const color = meterColor(value);
		return `background: conic-gradient(${color} ${value}%, var(--border) 0);`;
	}

	function meterText(percent) {
		return percent == null ? '--%' : `${Math.round(percent)}%`;
	}

	async function refreshSystemInfo() {
		try {
			const res = await fetch('/api/system/info');
			if (res.ok) systemInfo = await res.json();
		} catch {}
	}

	onMount(async () => {
		await refreshSystemInfo();
		_sysInfoTimer = setInterval(refreshSystemInfo, 3000);
	});

	onDestroy(() => {
		if (_sysInfoTimer) clearInterval(_sysInfoTimer);
	});
</script>

<header class="flex-shrink-0 h-12 px-4 flex items-center gap-3" style="background: var(--bg-base); border-bottom: 1px solid var(--border-subtle);">
	<div class="min-w-0 flex-1 flex items-center gap-2 overflow-hidden">
		<div class="min-w-[120px] max-w-[220px] px-2.5 py-1.5 text-[12px] font-semibold truncate flex-shrink-0" style="color: var(--text-primary); background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
			{currentPageTitle}
		</div>
		{#if hasConditioningIssue}
			<a
				href="/training/conditioning"
				class="app-status-badge flex-shrink-0"
				class:app-status-badge-danger={conditioningIssues.errors.length > 0}
				class:app-status-badge-warning={conditioningIssues.errors.length === 0}
				title={conditioningIssueTitle}
				aria-label="Conditioning recipe needs attention"
			></a>
		{/if}
		{#if page.url.pathname === '/training/dashboard'}
			<a href={monitorConfigHref} class="px-2.5 py-1.5 text-[11px] font-medium flex-shrink-0" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); color: var(--text-secondary); border-radius: var(--radius-sm);">Config</a>
		{/if}
		<div class="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium flex-shrink-0" style="color: var(--text-secondary); background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
			<span class="w-1.5 h-1.5 rounded-full" style="background: {processColor(activeProcess.state)}; box-shadow: {activeProcess.state === 'running' ? `0 0 6px ${processColor(activeProcess.state)}` : 'none'};"></span>
			<span>{activeProcess.label}</span>
		</div>

		{#if $projectLoaded && vramChips.length}
			<div class="flex items-center gap-1.5 min-w-0 overflow-hidden">
				{#each vramChips as chip}
					<div
						class="w-[142px] flex-shrink-0 px-2 py-1"
						style="background: var(--bg-surface); border: 1px solid {chip.fits ? 'var(--border-subtle)' : 'var(--danger)'}; border-radius: var(--radius-sm);"
						data-tooltip={`${chip.label}: ~${chip.value.toFixed(1)}G of ${vramTotal.toFixed(0)}G, ${chip.fits ? `${chip.delta.toFixed(1)}G free` : `${chip.delta.toFixed(1)}G over`}${chip.note ? ` (${chip.note})` : ''}`}
					>
						<div class="flex items-center justify-between gap-1 text-[10px] leading-none">
							<div class="min-w-0 flex items-center gap-1">
								<span class="w-1.5 h-1.5 rounded-full flex-shrink-0" style="background: {chip.color};"></span>
								<span class="font-semibold" style="color: var(--text-primary);">{chip.label}</span>
								{#if chip.note}
									<span class="truncate" style="color: var(--text-muted);">{chip.note}</span>
								{/if}
							</div>
							<span class="font-bold flex-shrink-0" style="color: {chip.fits ? 'var(--text-muted)' : 'var(--danger)'};">{chip.fits ? 'Fits' : 'Over'}</span>
						</div>
						<div class="mt-1 h-1 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
							<div class="h-full" style="width: {chip.percent.toFixed(0)}%; background: {chip.color}; border-radius: var(--radius-full); transition: width 180ms ease;"></div>
						</div>
					</div>
				{/each}
			</div>
		{/if}

		{#if systemInfo}
			<div class="min-w-0 flex-1 px-2.5 py-1.5 text-[11px] font-medium tabular-nums flex items-center gap-2 overflow-hidden whitespace-nowrap" style="color: var(--text-secondary); background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
				{#if gpu}
					<span class="min-w-0 truncate inline-flex items-center gap-1" title={`GPU load ${meterText(gpuLoadPercent)} · VRAM ${meterText(vramPercent)}`}>
						<span class="status-meter" style={meterStyle(vramPercent)} aria-hidden="true"><span></span></span>
						<span><span style="color: var(--text-muted);">GPU</span> {gpuName} {vramUsed.toFixed(1)}/{vramTotal.toFixed(0)}G</span>
					</span>
				{/if}
				{#if systemInfo.ram}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0 inline-flex items-center gap-1" title={`RAM used ${meterText(ramPercent)}`}>
						<span class="status-meter" style={meterStyle(ramPercent)} aria-hidden="true"><span></span></span>
						<span><span style="color: var(--text-muted);">RAM</span> {systemInfo.ram.used_gb}/{systemInfo.ram.total_gb}G</span>
					</span>
				{/if}
				{#if systemInfo.disk}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0 inline-flex items-center gap-1" title={`Disk used ${meterText(diskUsedPercent)}`}>
						<span class="status-meter" style={meterStyle(diskUsedPercent)} aria-hidden="true"><span></span></span>
						<span><span style="color: var(--text-muted);">Disk</span> {systemInfo.disk.free_gb}G free</span>
					</span>
				{/if}
				{#if systemInfo.cpu}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0 inline-flex items-center gap-1" title={`CPU load ${meterText(cpuPercent)}`}>
						<span class="status-meter" style={meterStyle(cpuPercent)} aria-hidden="true"><span></span></span>
						<span><span style="color: var(--text-muted);">CPU</span> {systemInfo.cpu.cores}c</span>
					</span>
				{/if}
				{#if systemInfo.python}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0"><span style="color: var(--text-muted);">Py</span> {systemInfo.python}</span>
				{/if}
			</div>
		{/if}
	</div>
</header>

<style>
	.status-meter {
		width: 10px;
		height: 10px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex: 0 0 auto;
		border-radius: 999px;
	}

	.status-meter > span {
		width: 5px;
		height: 5px;
		border-radius: 999px;
		background: var(--bg-surface);
	}
</style>
