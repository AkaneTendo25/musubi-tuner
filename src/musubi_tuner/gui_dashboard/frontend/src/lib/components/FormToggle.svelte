<script>
	let { label, checked = $bindable(false), tooltip = '', disabled = false, ...rest } = $props();
</script>

<label class="inline-flex items-center gap-2.5 cursor-pointer select-none {disabled ? 'opacity-40 pointer-events-none' : ''}" data-tooltip={tooltip || undefined}>
	<div class="relative flex-shrink-0">
		<input type="checkbox" bind:checked {disabled} {...rest} class="sr-only peer" />
		<div class="w-9 h-[20px] transition-colors" style="background: var(--toggle-bg); border-radius: var(--toggle-radius);"></div>
		<div class="w-9 h-[20px] absolute inset-0 transition-colors peer-checked:opacity-100 opacity-0" style="background: var(--toggle-active); border-radius: var(--toggle-radius);"></div>
		<div class="absolute left-[3px] top-[3px] w-[14px] h-[14px] bg-white transition-transform peer-checked:translate-x-[16px]" style="border-radius: var(--toggle-knob-radius); box-shadow: var(--toggle-knob-shadow);"></div>
	</div>
	<span class="text-[13px]" style="color: var(--text-secondary)">{label}</span>
</label>
