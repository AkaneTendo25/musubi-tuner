<script>
	import { page } from '$app/state';
	import { projectConfig, projectLoaded } from '$lib/stores/project.js';
	import { processStatuses } from '$lib/stores/processes.js';
	import { estimateLatentCaching, estimateTextCaching, estimateTraining } from '$lib/utils/vramEstimate.js';
	import { onMount, onDestroy } from 'svelte';

	let systemInfo = $state(null);
	let _sysInfoTimer = null;

	let gpu = $derived(systemInfo?.gpus?.[0]);
	let gpuName = $derived(gpu ? gpu.name.replace('NVIDIA ', '').replace('GeForce ', '') : '');
	let vramTotal = $derived(gpu ? gpu.vram_total_mb / 1024 : 24);
	let vramUsed = $derived(gpu ? gpu.vram_used_mb / 1024 : null);
	const pageTitles = {
		'/': 'Overview',
		'/dataset': 'Dataset',
		'/caching': 'Caching',
		'/samples': 'Samples',
		'/training': 'Training',
		'/training/techniques': 'Methods',
		'/training/full-finetune': 'Fine-tuning',
		'/training/dashboard': 'Monitor',
		'/inference': 'Inference',
		'/settings': 'Setup',
	};
	let currentPageTitle = $derived(pageTitles[page.url.pathname] || 'Dashboard');
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

	async function refreshSystemInfo() {
		try {
			const res = await fetch('/api/system/info');
			if (res.ok) systemInfo = await res.json();
		} catch {}
	}

	onMount(async () => {
		await refreshSystemInfo();
		_sysInfoTimer = setInterval(refreshSystemInfo, 30000);
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
					<span class="min-w-0 truncate"><span style="color: var(--text-muted);">GPU</span> {gpuName} {vramUsed.toFixed(1)}/{vramTotal.toFixed(0)}G</span>
				{/if}
				{#if systemInfo.ram}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0"><span style="color: var(--text-muted);">RAM</span> {systemInfo.ram.used_gb}/{systemInfo.ram.total_gb}G</span>
				{/if}
				{#if systemInfo.disk}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0"><span style="color: var(--text-muted);">Disk</span> {systemInfo.disk.free_gb}G free</span>
				{/if}
				{#if systemInfo.cpu}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0"><span style="color: var(--text-muted);">CPU</span> {systemInfo.cpu.cores}c</span>
				{/if}
				{#if systemInfo.python}
					<span class="flex-shrink-0" style="color: var(--border);">|</span>
					<span class="flex-shrink-0"><span style="color: var(--text-muted);">Py</span> {systemInfo.python}</span>
				{/if}
			</div>
		{/if}
	</div>
</header>
