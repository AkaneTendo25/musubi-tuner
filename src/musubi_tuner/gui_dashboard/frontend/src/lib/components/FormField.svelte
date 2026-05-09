<script>
	import FieldResetButton from './FieldResetButton.svelte';
	import { labelFromFieldPath } from '$lib/utils/fieldLabels.js';

	let {
		label = '',
		value = $bindable(''),
		type = 'text',
		placeholder = '',
		tooltip = '',
		disabled = false,
		invalid = false,
		error = '',
		min = undefined,
		max = undefined,
		step = undefined,
		fieldPath = '',
		oninput,
		...rest
	} = $props();

	let displayLabel = $derived(label || labelFromFieldPath(fieldPath));

	// Handle input to ensure immediate propagation
	function handleInput(e) {
		value = e.target.value;
		if (oninput) {
			oninput(e);
		}
	}
</script>

<label class="block">
	<span class="flex items-center gap-1 text-xs font-medium uppercase tracking-wider" style="color: {invalid ? 'var(--danger)' : 'var(--text-muted)'}; font-family: var(--font-label);">
		<span data-tooltip={tooltip || undefined}>{displayLabel}</span>
		<FieldResetButton {fieldPath} {disabled} />
	</span>
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
		style="background: var(--bg-input); border: 1px solid {invalid ? 'var(--danger)' : 'var(--border)'}; color: var(--text-primary); border-radius: var(--radius-sm);"
	/>
	{#if error}
		<div class="mt-1 text-[11px]" style="color: var(--danger);">{error}</div>
	{/if}
</label>
