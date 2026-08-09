<script>
	import FieldResetButton from './FieldResetButton.svelte';
	import { labelFromFieldPath } from '$lib/utils/fieldLabels.js';

	let { label = '', value = $bindable(''), options = [], tooltip = '', disabled = false, fieldPath = '', onchange, ...rest } = $props();
	let displayLabel = $derived(label || labelFromFieldPath(fieldPath));

	function handleChange(e) {
		value = e.target.value;
		if (onchange) {
			onchange(e);
		}
	}
</script>

<label class="block">
	<span class="flex items-center gap-1 text-xs font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">
		<span data-tooltip={tooltip || undefined}>{displayLabel}</span>
		<FieldResetButton {fieldPath} {disabled} />
	</span>
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
