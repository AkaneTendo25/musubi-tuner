<script>
	let { label, value = $bindable(''), options = [], tooltip = '', disabled = false, ...rest } = $props();
</script>

<label class="block" data-tooltip={tooltip || undefined}>
	<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">{label}</span>
	<select
		bind:value
		{disabled}
		{...rest}
		class="mt-0.5 block w-full px-2.5 py-1.5 text-[13px] transition-colors disabled:opacity-40"
		style="background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); border-radius: var(--radius-sm);"
	>
		{#each options as opt}
			{#if typeof opt === 'string'}
				<option value={opt}>{opt}</option>
			{:else}
				<option value={opt.value}>{opt.label}</option>
			{/if}
		{/each}
	</select>
</label>
