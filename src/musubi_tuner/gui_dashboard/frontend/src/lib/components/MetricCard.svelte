<script>
	let { label, value, unit = '', inactive = false } = $props();
</script>

<div class="px-3.5 py-2.5 min-h-[72px] flex flex-col justify-between" style="background: {inactive ? 'color-mix(in srgb, var(--bg-surface) 78%, transparent)' : 'var(--bg-surface)'}; border: 1px solid {inactive ? 'var(--border)' : 'var(--border-subtle)'}; border-radius: var(--radius-md); opacity: {inactive ? '0.76' : '1'}; box-shadow: {inactive ? 'none' : 'var(--shadow-sm)'}; transition: opacity 0.2s ease, border-color 0.2s ease, transform 0.2s ease;">
	<div class="text-[9px] font-medium uppercase tracking-[0.22em]" style="color: var(--text-muted); font-family: var(--font-label);">{label}</div>
	<div class="text-[17px] leading-none font-semibold tabular-nums mt-1" style="color: {inactive ? 'var(--text-secondary)' : 'var(--text-primary)'};">
		{value}{#if unit}<span class="text-[11px] ml-1" style="color: var(--text-muted);">{unit}</span>{/if}
	</div>
</div>
