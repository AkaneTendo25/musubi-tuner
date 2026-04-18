<script>
	import { status, eta } from '../stores/status.js';
	import { connected } from '../stores/sse.js';
	import MetricCard from './MetricCard.svelte';
	let { active = false } = $props();

	function formatTime(seconds) {
		if (seconds == null || seconds <= 0) return '--';
		const h = Math.floor(seconds / 3600);
		const m = Math.floor((seconds % 3600) / 60);
		const s = Math.floor(seconds % 60);
		if (h > 0) return `${h}h ${m}m`;
		if (m > 0) return `${m}m ${s}s`;
		return `${s}s`;
	}

	function formatSpeed(s) {
		if (!s || s <= 0) return '--';
		if (s >= 1) return s.toFixed(1);
		return (1 / s).toFixed(1) + ' s/step';
	}

	let statusText = $derived(active ? ($status?.status || 'waiting') : 'waiting');
	let progress = $derived(
		active && $status?.max_steps > 0 ? (($status.step / $status.max_steps) * 100).toFixed(1) : 0
	);
let epochValue = $derived(active ? `${$status?.epoch ?? 0}/${$status?.max_epochs ?? 0}` : '0/0');
let statusLabel = $derived(active ? 'Live Status' : 'Inactive');
</script>

<div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-12 gap-2.5">
	<div class="xl:col-span-4 px-3.5 py-2.5 min-h-[72px]" style="background: {active ? 'var(--bg-surface)' : 'color-mix(in srgb, var(--bg-surface) 78%, transparent)'}; border: 1px solid {active ? 'var(--border-subtle)' : 'var(--border)'}; border-radius: var(--radius-md); opacity: {active ? '1' : '0.76'}; box-shadow: {active ? 'var(--shadow-sm)' : 'none'}; transition: opacity 0.2s ease, border-color 0.2s ease;">
		<div class="flex items-center justify-between gap-3">
			<div class="flex items-center gap-2">
			<div
				class="w-1.5 h-1.5 rounded-full flex-shrink-0"
				style="background: {active ? ($connected ? 'var(--success)' : 'var(--danger)') : 'var(--text-muted)'}; box-shadow: 0 0 6px {active ? ($connected ? 'var(--success)' : 'var(--danger)') : 'transparent'};"
			></div>
				<span class="text-[9px] font-medium uppercase tracking-[0.22em]" style="color: var(--text-muted); font-family: var(--font-label);">{statusLabel}</span>
			</div>
			<span class="text-[10px] font-medium tabular-nums" style="color: var(--text-muted);">{active ? progress : 0}%</span>
		</div>
		<div class="flex items-end justify-between gap-3 mt-1.5">
			<div class="text-[17px] leading-none font-semibold capitalize" style="color: {active ? 'var(--text-primary)' : 'var(--text-secondary)'};">{statusText}</div>
			<div class="text-[10px] tabular-nums whitespace-nowrap" style="color: var(--text-muted);">
				Step {active ? ($status?.step?.toLocaleString() ?? 0) : 0} / {active ? ($status?.max_steps?.toLocaleString() ?? 0) : 0}
			</div>
		</div>
		<div class="h-1 mt-2 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
			<div
				class="h-full transition-all duration-500"
				style="width: {active ? progress : 0}%; background: {active ? 'var(--accent)' : 'var(--text-muted)'}; border-radius: var(--radius-full);"
			></div>
		</div>
	</div>

	<div class="xl:col-span-2">
		<MetricCard label="Epoch" value={epochValue} inactive={!active} />
	</div>
	<div class="xl:col-span-2">
		<MetricCard label="Speed" value={active ? formatSpeed($status?.speed_steps_per_sec) : '--'} inactive={!active} />
	</div>
	<div class="xl:col-span-2">
		<MetricCard label="Elapsed" value={active ? formatTime($status?.elapsed_sec) : '--'} inactive={!active} />
	</div>
	<div class="xl:col-span-2">
		<MetricCard label="ETA" value={active ? formatTime($eta) : '--'} inactive={!active} />
	</div>
</div>
