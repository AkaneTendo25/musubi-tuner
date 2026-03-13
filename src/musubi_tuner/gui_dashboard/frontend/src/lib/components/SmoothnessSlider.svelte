<script>
	let { value = $bindable(0.95), label = 'Smoothing' } = $props();
</script>

<div class="flex items-center gap-2 text-xs text-gray-500">
	<span>{label}</span>
	<input
		type="range"
		min="0"
		max="0.99"
		step="0.01"
		bind:value
		class="w-24 h-1 accent-blue-500"
	/>
	<span class="tabular-nums w-8">{value.toFixed(2)}</span>
</div>
