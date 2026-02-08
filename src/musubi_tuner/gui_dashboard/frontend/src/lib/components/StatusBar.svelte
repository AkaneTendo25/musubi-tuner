<script>
	import { status, eta } from '../stores/status.js';
	import { connected } from '../stores/sse.js';
	import MetricCard from './MetricCard.svelte';

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

	let statusText = $derived($status?.status || 'waiting');
	let progress = $derived(
		$status?.max_steps > 0 ? (($status.step / $status.max_steps) * 100).toFixed(1) : 0
	);
</script>

<div class="flex flex-wrap items-center gap-3">
	<div class="flex items-center gap-2 mr-2">
		<div
			class="w-2 h-2 rounded-full"
			style="background: {$connected ? 'var(--success)' : 'var(--danger)'}; box-shadow: 0 0 6px {$connected ? 'var(--success)' : 'var(--danger)'};"
		></div>
		<span class="text-[12px] capitalize" style="color: var(--text-muted);">{statusText}</span>
	</div>

	{#if $status?.max_steps > 0}
		<div class="flex-1 min-w-[200px]">
			<div class="flex justify-between text-[11px] mb-1" style="color: var(--text-muted);">
				<span>Step {$status?.step?.toLocaleString() ?? 0} / {$status?.max_steps?.toLocaleString()}</span>
				<span>{progress}%</span>
			</div>
			<div class="h-1.5 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
				<div
					class="h-full transition-all duration-500"
					style="width: {progress}%; background: var(--accent); border-radius: var(--radius-full);"
				></div>
			</div>
		</div>
	{/if}

	<MetricCard label="Epoch" value="{$status?.epoch ?? 0}/{$status?.max_epochs ?? 0}" />
	<MetricCard label="Speed" value={formatSpeed($status?.speed_steps_per_sec)} />
	<MetricCard label="Elapsed" value={formatTime($status?.elapsed_sec)} />
	<MetricCard label="ETA" value={formatTime($eta)} />
</div>
