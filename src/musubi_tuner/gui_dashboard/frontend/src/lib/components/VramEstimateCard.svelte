<script>
	let {
		title,
		subtitle = '',
		estimate,
		total = 24,
		color = 'var(--accent)'
	} = $props();

	let value = $derived(Number(estimate?.total || 0));
	let capacity = $derived(Math.max(Number(total) || 1, 1));
	let percent = $derived(Math.min(value / capacity, 1.2));
	let over = $derived(value > capacity);
	let barColor = $derived(over ? 'var(--danger)' : color);
	let parts = $derived(estimate?.parts ?? []);
	let hasNotes = $derived(!!(estimate?.swap > 0 || estimate?.noGemma || estimate?.samplingSpike || estimate?.preservationGemmaSpike));
</script>

<div class="min-h-[156px] p-4 flex flex-col justify-between" style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
	<div>
		<div class="flex items-start justify-between gap-3">
			<div>
				<div class="text-[12px] font-semibold" style="color: var(--text-secondary);">{title}</div>
				{#if subtitle}
					<div class="text-[11px] mt-1" style="color: var(--text-muted);">{subtitle}</div>
				{/if}
			</div>
			<div class="text-right">
				<div class="text-[26px] leading-none font-bold tabular-nums" style="color: {barColor};">~{value.toFixed(1)}</div>
				<div class="text-[11px] mt-1" style="color: var(--text-muted);">GB</div>
			</div>
		</div>
		<div class="h-2.5 mt-4 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
			<div class="h-full" style="width: {Math.min(percent * 100, 100).toFixed(0)}%; background: {barColor}; border-radius: var(--radius-full); transition: width 0.5s ease;"></div>
		</div>
	</div>

	<div class="flex flex-wrap gap-x-4 gap-y-1 mt-3">
		{#each parts as p}
			<span class="flex items-center gap-1.5 text-[12px]">
				<span class="w-2 h-2 rounded-full" style="background: {p.color};"></span>
				<span style="color: var(--text-muted);">{p.label} {p.value.toFixed(1)}G</span>
			</span>
		{/each}
	</div>

	{#if hasNotes}
		<div class="mt-2 space-y-0.5">
			{#if estimate.swap > 0}
				<div class="text-[11px]" style="color: var(--success);">-{estimate.swap.toFixed(1)}G block swap</div>
			{/if}
			{#if estimate.noGemma}
				<div class="text-[11px]" style="color: var(--info);">steady-state excludes VAE/Gemma</div>
			{/if}
			{#if estimate.samplingSpike}
				<div class="text-[11px]" style="color: var(--text-muted);">sampling can temporarily load VAE/Gemma</div>
			{/if}
			{#if estimate.preservationGemmaSpike}
				<div class="text-[11px]" style="color: var(--text-muted);">preservation encoding loads Gemma if not precached</div>
			{/if}
		</div>
	{/if}
</div>
