<script>
	import FieldResetButton from './FieldResetButton.svelte';
	import { labelFromFieldPath } from '$lib/utils/fieldLabels.js';

	let { label = '', checked = $bindable(false), tooltip = '', disabled = false, fieldPath = '', ...rest } = $props();
	let displayLabel = $derived(label || labelFromFieldPath(fieldPath));
</script>

<div class="h-full flex items-end min-w-0 w-full max-w-full">
	<label class="grid w-full grid-cols-[40px_minmax(0,1fr)] items-center gap-3 cursor-pointer select-none {disabled ? 'opacity-40 pointer-events-none' : ''}">
		<div class="relative flex-shrink-0">
			<input type="checkbox" bind:checked {disabled} {...rest} class="sr-only peer" />
			<div class="w-10 h-[22px] transition-colors" style="background: var(--toggle-bg); border-radius: var(--toggle-radius);"></div>
			<div class="w-10 h-[22px] absolute inset-0 transition-colors peer-checked:opacity-100 opacity-0" style="background: var(--toggle-active); border-radius: var(--toggle-radius);"></div>
			<div class="absolute left-[3px] top-[3px] w-4 h-4 bg-white transition-transform peer-checked:translate-x-[18px]" style="border-radius: var(--toggle-knob-radius); box-shadow: var(--toggle-knob-shadow);"></div>
		</div>
		<span class="min-w-0 inline-flex max-w-full items-center gap-1 text-sm leading-tight" style="color: var(--text-secondary)">
			<span class="min-w-0 flex-shrink" data-tooltip={tooltip || undefined}>
				<span class="block truncate whitespace-nowrap">{displayLabel}</span>
			</span>
			<FieldResetButton {fieldPath} {disabled} />
		</span>
	</label>
</div>
