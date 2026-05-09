<script>
	let {
		label,
		value = 0,
		display = null,
		sublabel = '',
		color = 'var(--accent)',
		size = 116,
		stroke = 8,
		valueSize = 18,
		inactive = false
	} = $props();

	const radius = 52;
	const circumference = 2 * Math.PI * radius;
	let percent = $derived(Math.max(0, Math.min(100, Number(value) || 0)));
	let offset = $derived(circumference * (1 - percent / 100));
	let valueText = $derived(display ?? `${percent.toFixed(0)}%`);
</script>

<div class="flex flex-col items-center justify-center text-center min-w-0" style="opacity: {inactive ? 0.55 : 1};">
	<div class="relative flex-shrink-0" style="width: {size}px; height: {size}px;">
		<svg viewBox="0 0 120 120" class="w-full h-full" style="transform: rotate(-90deg);">
			<circle cx="60" cy="60" r={radius} fill="none" stroke="var(--border)" stroke-width={stroke} />
			<circle
				cx="60"
				cy="60"
				r={radius}
				fill="none"
				stroke={color}
				stroke-width={stroke}
				stroke-dasharray={circumference}
				stroke-dashoffset={offset}
				stroke-linecap="round"
				style="transition: stroke-dashoffset 0.6s ease;"
			/>
		</svg>
		<div class="absolute inset-0 flex flex-col items-center justify-center px-2">
			<div class="leading-none font-bold tabular-nums truncate max-w-full" style="color: {color}; font-size: {valueSize}px;">{valueText}</div>
			{#if sublabel}
				<div class="text-[10px] mt-1 truncate max-w-full" style="color: var(--text-muted);">{sublabel}</div>
			{/if}
		</div>
	</div>
	{#if label}
		<div class="text-[11px] font-medium mt-2 truncate max-w-full" style="color: var(--text-secondary);">{label}</div>
	{/if}
</div>
