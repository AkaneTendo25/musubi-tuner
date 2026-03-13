<script>
	/**
	 * Form field with autocomplete suggestions (datalist).
	 * Allows both selecting from suggestions and typing custom values.
	 */
	let {
		label,
		value = $bindable(''),
		suggestions = [],
		type = 'text',
		placeholder = '',
		tooltip = '',
		disabled = false,
		oninput,
		...rest
	} = $props();

	// Generate unique ID for datalist
	const datalistId = `datalist-${Math.random().toString(36).slice(2,9)}`;

	// Handle input to ensure immediate propagation
	function handleInput(e) {
		value = e.target.value;
		oninput?.(e);
	}
</script>

<label class="block" data-tooltip={tooltip || undefined}>
	<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">{label}</span>
	<input
		{type}
		{value}
		oninput={handleInput}
		{placeholder}
		{disabled}
		{...rest}
		list={datalistId}
		class="mt-0.5 block w-full px-2.5 py-1.5 text-[13px] transition-colors disabled:opacity-40"
		style="background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); border-radius: var(--radius-sm);"
	/>
	{#if suggestions && suggestions.length > 0}
		<datalist id={datalistId}>
			{#each suggestions as option}
				<option value={option}>{option}</option>
			{/each}
		</datalist>
	{/if}
</label>
