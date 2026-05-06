<script>
	let {
		exists = false,
		foundPath = '',
		disabled = false,
		scanning = false,
		scanMessage = '',
		scanTone = 'muted',
		onscan,
		oncancel,
		onusefound
	} = $props();
</script>

<div class="flex items-center gap-2 text-[10px] -mt-1 w-full min-w-0 overflow-hidden">
	<span
		class="font-medium px-1.5 py-0.5 flex-shrink-0 whitespace-nowrap"
		style="background: {exists ? 'var(--success-muted, rgba(34,197,94,0.1))' : 'transparent'}; color: {exists ? 'var(--success)' : 'var(--danger)'}; border-radius: var(--radius-sm);"
	>
		{exists ? 'Found' : 'Missing'}
	</span>
	{#if !exists && onscan}
		<button
			type="button"
			onclick={() => onscan?.()}
			disabled={disabled || scanning}
			class="inline-flex items-center justify-center text-[10px] font-medium disabled:opacity-40 flex-shrink-0 whitespace-nowrap"
			style="padding: 0 6px; background: transparent; border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
			onmouseenter={(e) => { if (!disabled && !scanning) { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--bg-base)'; e.currentTarget.style.background = 'var(--accent)'; } }}
			onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.background = 'transparent'; }}
		>
			{scanning ? 'Scanning...' : 'Scan'}
		</button>
		{#if scanning && oncancel}
			<button
				type="button"
				onclick={() => oncancel?.()}
				disabled={disabled}
				class="inline-flex items-center justify-center text-[10px] font-medium disabled:opacity-40 flex-shrink-0 whitespace-nowrap"
				style="padding: 0 6px; background: transparent; border: 1px solid var(--danger); color: var(--danger); border-radius: var(--radius-sm);"
				onmouseenter={(e) => { if (!disabled) { e.currentTarget.style.background = 'var(--danger)'; e.currentTarget.style.color = 'var(--bg-base)'; } }}
				onmouseleave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--danger)'; }}
			>
				Stop
			</button>
		{/if}
	{/if}
	{#if !exists && foundPath}
		<button
			type="button"
			onclick={() => onusefound?.(foundPath)}
			{disabled}
			data-tooltip={foundPath}
			class="inline-flex items-center justify-center text-[10px] font-semibold disabled:opacity-40 flex-shrink-0 whitespace-nowrap"
			style="padding: 0 10px; background: var(--accent); color: white; border-radius: var(--radius-sm);"
			onmouseenter={(e) => e.currentTarget.style.background = 'var(--accent-hover)'}
			onmouseleave={(e) => e.currentTarget.style.background = 'var(--accent)'}
		>
			Use found
		</button>
		<span class="min-w-0 flex-1 truncate whitespace-nowrap" style="color: var(--text-muted);">{foundPath}</span>
	{/if}
	{#if !exists && scanMessage && !foundPath}
		<span
			class="min-w-0 flex-1 truncate whitespace-nowrap"
			style="color: {scanTone === 'danger' ? 'var(--danger)' : scanTone === 'success' ? 'var(--success)' : 'var(--text-muted)'};"
		>
			{scanMessage}
		</span>
	{/if}
</div>
