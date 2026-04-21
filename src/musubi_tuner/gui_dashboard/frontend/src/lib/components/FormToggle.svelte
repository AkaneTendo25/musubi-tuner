<script>
	let { label, checked = $bindable(false), tooltip = '', disabled = false, ...rest } = $props();
</script>

<div class="h-full flex items-end">
	<label class="inline-flex items-center gap-3 cursor-pointer select-none {disabled ? 'opacity-40 pointer-events-none' : ''}" data-tooltip={tooltip || undefined}>
		<div class="relative flex-shrink-0">
			<input type="checkbox" bind:checked {disabled} {...rest} class="sr-only peer" />
			<div class="w-10 h-[22px] transition-colors" style="background: var(--toggle-bg); border-radius: var(--toggle-radius);"></div>
			<div class="w-10 h-[22px] absolute inset-0 transition-colors peer-checked:opacity-100 opacity-0" style="background: var(--toggle-active); border-radius: var(--toggle-radius);"></div>
			<div class="absolute left-[3px] top-[3px] w-4 h-4 bg-white transition-transform peer-checked:translate-x-[18px]" style="border-radius: var(--toggle-knob-radius); box-shadow: var(--toggle-knob-shadow);"></div>
		</div>
		<span class="text-sm" style="color: var(--text-secondary)">{label}</span>
	</label>
</div>
