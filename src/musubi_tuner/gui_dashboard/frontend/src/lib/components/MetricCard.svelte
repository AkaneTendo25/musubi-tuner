<script>
	let { label, value, unit = '' } = $props();
</script>

<div class="px-4 py-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
	<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">{label}</div>
	<div class="text-xl font-semibold tabular-nums mt-0.5" style="color: var(--text-primary);">
		{value}{#if unit}<span class="text-sm ml-1" style="color: var(--text-muted);">{unit}</span>{/if}
	</div>
</div>
