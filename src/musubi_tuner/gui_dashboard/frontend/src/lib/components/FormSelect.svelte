<script>
	let { label, value = $bindable(''), options = [], tooltip = '', disabled = false, onchange, ...rest } = $props();

	function handleChange(e) {
		value = e.target.value;
		if (onchange) {
			onchange(e);
		}
	}
</script>

<label class="block" data-tooltip={tooltip || undefined}>
	<span class="text-xs font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">{label}</span>
	<select
		{value}
		onchange={handleChange}
		{disabled}
		{...rest}
		class="mt-1 block w-full px-3 py-2 text-sm transition-colors disabled:opacity-40"
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
