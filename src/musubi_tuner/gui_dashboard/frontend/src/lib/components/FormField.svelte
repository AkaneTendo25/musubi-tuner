<script>
	let { label, value = $bindable(''), type = 'text', placeholder = '', tooltip = '', disabled = false, min = undefined, max = undefined, step = undefined, oninput, ...rest } = $props();

	// Handle input to ensure immediate propagation
	function handleInput(e) {
		value = e.target.value;
		if (oninput) {
			oninput(e);
		}
	}
</script>

<label class="block" data-tooltip={tooltip || undefined}>
	<span class="text-xs font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">{label}</span>
	<input
		{type}
		{value}
		oninput={handleInput}
		{placeholder}
		{disabled}
		{min}
		{max}
		{step}
		{...rest}
		class="mt-1 block w-full px-3 py-2 text-sm transition-colors disabled:opacity-40"
		style="background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); border-radius: var(--radius-sm);"
	/>
</label>
