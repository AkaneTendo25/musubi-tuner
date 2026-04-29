<script>
	let {
		exists = false,
		foundPath = '',
		disabled = false,
		onusefound
	} = $props();
</script>

<div class="flex items-center gap-2 text-[10px] -mt-1">
	<span
		class="font-medium px-1.5 py-0.5"
		style="background: {exists ? 'var(--success-muted, rgba(34,197,94,0.1))' : 'var(--danger-muted)'}; color: {exists ? 'var(--success)' : 'var(--danger)'}; border-radius: var(--radius-sm);"
	>
		{exists ? 'Found' : 'Missing'}
	</span>
	{#if !exists && foundPath}
		<button
			type="button"
			onclick={() => onusefound?.(foundPath)}
			{disabled}
			data-tooltip={foundPath}
			class="font-medium disabled:opacity-40"
			style="color: var(--accent);"
		>
			Use found
		</button>
		<span class="truncate min-w-0" style="color: var(--text-muted);">{foundPath}</span>
	{/if}
</div>
